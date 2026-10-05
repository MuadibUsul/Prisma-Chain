"""The prisma CLI command groups (B5-01).

Ownership per group (stated honestly, per the roadmap):

    wallet / balance / bond / worker register / receipt / dispute / task (chain side)
        -> chain queries + transactions through `prismad` (the node CLI); no
           protocol logic here, no wire-format knowledge beyond flag names
    faucet            -> HTTP client for the faucet service (B7)
    job post / job get (API side of task query) -> HTTP client for the Job API (B6)
    run worker / run da / run watcher -> exec the product daemons' own CLIs
    version           -> from the release build (B1-01 identity)
    config            -> one operator config file + validation

Nothing here duplicates protocol logic or drifts from the frozen wire
formats: the CLI composes the product packages and the node CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

from prisma_worker import keystore as worker_keystore
from prisma_worker.identity import WorkerIdentity
from prisma_worker.redact import install as install_redaction
from prisma_worker.redact import register_secret

from .common import CliError, GlobalConfig, emit, http_json, prismad


def _config(args: argparse.Namespace) -> GlobalConfig:
    config = GlobalConfig.load()
    for attr in ("chain_id", "node", "api", "prismad"):
        value = getattr(args, attr, None)
        if value:
            setattr(config, attr, value)
    problems = config.validate()
    if problems:
        raise CliError("config invalid: " + "; ".join(problems))
    return config


def _passphrase(args: argparse.Namespace) -> str:
    if getattr(args, "passphrase_stdin", False):
        passphrase = sys.stdin.readline().rstrip()
    elif os.environ.get("PRISMA_CLI_PASSPHRASE"):
        passphrase = os.environ["PRISMA_CLI_PASSPHRASE"]
    else:
        import getpass

        passphrase = getpass.getpass("keystore passphrase: ")
    if not passphrase:
        raise CliError("an empty passphrase is refused")
    return passphrase


def _identity(args: argparse.Namespace):
    config = _config(args)
    path = pathlib.Path(args.keystore or config.keystore or
                        worker_keystore.default_path()).expanduser()
    if not path.exists():
        raise CliError(f"no operator keystore at {path}; run `prisma wallet init`")
    return worker_keystore.load(path, _passphrase(args))


# --- wallet ------------------------------------------------------------------

def cmd_wallet_init(args: argparse.Namespace) -> int:
    install_redaction()
    config = _config(args)
    path = pathlib.Path(args.keystore or config.keystore or
                        worker_keystore.default_path()).expanduser()
    if path.exists() and not args.force:
        raise CliError(f"{path} already exists; pass --force to replace it")
    if args.import_protocol_hex:
        identity = WorkerIdentity.from_protocol_hex(args.import_protocol_hex,
                                                    os.urandom(32))
    else:
        identity = WorkerIdentity.generate()
    for secret in identity.secret_material():
        register_secret(secret)
    worker_keystore.save(path, identity, _passphrase(args))
    record = identity.public_record()
    record["keystore"] = str(path)
    emit(record, args.json)
    return 0


def cmd_wallet_show(args: argparse.Namespace) -> int:
    identity = _identity(args)
    emit(identity.public_record(), args.json)
    return 0


# --- chain queries -----------------------------------------------------------

def cmd_balance(args: argparse.Namespace) -> int:
    config = _config(args)
    address = args.address
    if not address:
        address = _identity(args).account_address
    payload = prismad(config, "query", "bank", "balance", address, "uprsm")
    emit({"address": address, "balance_uprsm": payload.get("balance", {}).get("amount", "0")},
         args.json)
    return 0


def cmd_task_query(args: argparse.Namespace) -> int:
    """Task/job status: chain side here; the API side is `prisma job get` (B6)."""
    config = _config(args)
    payload = prismad(config, "query", "compute", "task", "--task-id", str(args.task_id))
    document = payload.get("task_json")
    if document:
        import base64

        document = json.loads(base64.b64decode(document))
    emit(document or payload, args.json, fallback=f"task {args.task_id} not found")
    return 0


def cmd_receipt_query(args: argparse.Namespace) -> int:
    config = _config(args)
    payload = prismad(config, "query", "compute", "gemm-receipt",
                      "--receipt-id", args.receipt_id)
    emit(payload, args.json, fallback=f"receipt {args.receipt_id} not found")
    return 0


def cmd_da_query(args: argparse.Namespace) -> int:
    """DA status: chain side (per-task availability)."""
    config = _config(args)
    payload = prismad(config, "query", "compute", "gemmda-status",
                      "--task-id", str(args.task_id))
    emit(payload, args.json, fallback=f"no DA status for task {args.task_id}")
    return 0


def cmd_dispute_query(args: argparse.Namespace) -> int:
    config = _config(args)
    payload = prismad(config, "query", "compute", "graph-v2-status",
                      "--task-id", str(args.task_id))
    emit(payload, args.json, fallback=f"no dispute status for task {args.task_id}")
    return 0


# --- chain transactions ------------------------------------------------------

def cmd_bond(args: argparse.Namespace) -> int:
    install_redaction()
    config = _config(args)
    identity = _identity(args)
    for secret in identity.secret_material():
        register_secret(secret)
    from prisma_da.chain_tx import provider_keyring, run_tx

    args_list = ["compute", "bond-worker", "--amount", str(args.amount)]
    if args.register:
        args_list += ["--network-public-key", identity.protocol_public.hex(),
                      "--network-key-proof",
                      identity.network_key_proof_hex(config.chain_id, identity.account_address)]
    with provider_keyring(identity.account_scalar.hex(), prismad=config.prismad,
                          chain_id=config.chain_id, rpc_url=config.node) as keyring:
        txhash = run_tx(keyring, args_list, prismad=config.prismad,
                        chain_id=config.chain_id, rpc_url=config.node)
    emit({"bonded": True, "txhash": txhash, "registered": bool(args.register)}, args.json)
    return 0


def cmd_worker_register(args: argparse.Namespace) -> int:
    """Register (bond with the network key binding) — the dedicated spelling."""
    args.register = True
    return cmd_bond(args)


# --- API clients (honest ownership: HTTP, not chain tx) ----------------------

def cmd_faucet(args: argparse.Namespace) -> int:
    config = _config(args)
    if not config.faucet:
        raise CliError("no faucet endpoint configured; set it with `prisma config set --faucet URL`")
    status, payload = http_json("POST", config.faucet.rstrip("/") + "/fund",
                                body={"address": args.address, "denom": "uprsm"})
    emit({"status": status, **payload}, args.json)
    return 0 if status < 300 else 1


def cmd_job_post(args: argparse.Namespace) -> int:
    config = _config(args)
    if not config.api:
        raise CliError("no Job API configured; set it with `prisma config set --api URL` "
                       "(the API itself is B6)")
    body = {"profile": args.profile, "input": json.loads(args.input) if args.input else None}
    if args.idempotency_key:
        body["idempotency_key"] = args.idempotency_key
    status, payload = http_json("POST", config.api.rstrip("/") + "/v1/jobs", body=body)
    emit({"status": status, **payload}, args.json)
    return 0 if status < 300 else 1


def cmd_job_get(args: argparse.Namespace) -> int:
    config = _config(args)
    if not config.api:
        raise CliError("no Job API configured (B6)")
    status, payload = http_json("GET", config.api.rstrip("/") + f"/v1/jobs/{args.job_id}")
    emit({"status": status, **payload}, args.json)
    return 0 if status < 300 else 1


# --- daemon launchers (exec the components' own CLIs) ------------------------

def _run_component(module: str, args: argparse.Namespace, passthrough: list[str]) -> int:
    python = sys.executable
    os.execv(python, [python, "-m", module, *passthrough])


def cmd_run_worker(args: argparse.Namespace) -> int:
    if shutil.which("prisma-worker"):
        os.execvp("prisma-worker", ["prisma-worker", *args.passthrough])
    return _run_component("prisma_worker", args, args.passthrough)


def cmd_run_da(args: argparse.Namespace) -> int:
    if shutil.which("prisma-da"):
        os.execvp("prisma-da", ["prisma-da", *args.passthrough])
    return _run_component("prisma_da", args, args.passthrough)


def cmd_run_watcher(args: argparse.Namespace) -> int:
    if shutil.which("prisma-watcher"):
        os.execvp("prisma-watcher", ["prisma-watcher", *args.passthrough])
    return _run_component("prisma_watcher", args, args.passthrough)


# --- version / config --------------------------------------------------------

def cmd_version(args: argparse.Namespace) -> int:
    config = _config(args)
    try:
        payload = prismad(config, "version")
    except (CliError, OSError) as exc:
        emit({"error": str(exc)}, args.json, fallback=f"prismad not reachable: {exc}")
        return 1
    emit(payload, args.json)
    return 0


def cmd_config_set(args: argparse.Namespace) -> int:
    config = GlobalConfig.load()
    for attr in ("chain_id", "node", "api", "prismad", "keystore", "faucet"):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(config, attr, value)
    problems = config.validate()
    if problems:
        raise CliError("refusing to save an invalid config: " + "; ".join(problems))
    path = config.save()
    emit({"saved": str(path), **config.to_dict()}, args.json)
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    config = GlobalConfig.load()
    problems = config.validate()
    payload = config.to_dict()
    payload["problems"] = problems
    emit(payload, args.json)
    return 1 if problems else 0


# --- parser ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prisma", description="Prisma Chain operator/developer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chain-id", default="", help="override the configured chain id")
    parser.add_argument("--node", default="", help="override the configured chain RPC")
    parser.add_argument("--api", default="", help="override the configured Job API base")
    parser.add_argument("--prismad", default="", help="override the prismad binary")
    sub = parser.add_subparsers(dest="group", required=True)

    def keystore_flags(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--keystore", default="", help="operator keystore path")
        cmd.add_argument("--passphrase-stdin", action="store_true")

    wallet = sub.add_parser("wallet", help="operator key/account management (chain rules)")
    wallet_sub = wallet.add_subparsers(dest="action", required=True)
    wallet_init = wallet_sub.add_parser("init", help="generate or import the operator identity")
    keystore_flags(wallet_init)
    wallet_init.add_argument("--force", action="store_true")
    wallet_init.add_argument("--import-protocol-hex", default="",
                             help="import an existing ed25519 protocol key (hex)")
    wallet_init.add_argument("--json", action="store_true")
    wallet_init.set_defaults(func=cmd_wallet_init)
    wallet_show = wallet_sub.add_parser("show", help="print the public record")
    keystore_flags(wallet_show)
    wallet_show.add_argument("--json", action="store_true")
    wallet_show.set_defaults(func=cmd_wallet_show)

    balance = sub.add_parser("balance", help="account balance (chain)")
    balance.add_argument("--address", default="", help="default: the operator identity")
    balance.add_argument("--json", action="store_true")
    balance.set_defaults(func=cmd_balance)

    bond = sub.add_parser("bond", help="bond (and optionally register) the operator as a worker (chain)")
    keystore_flags(bond)
    bond.add_argument("--amount", type=int, required=True)
    bond.add_argument("--register", action="store_true",
                      help="also bind the protocol key (network-key proof)")
    bond.add_argument("--json", action="store_true")
    bond.set_defaults(func=cmd_bond)

    register = sub.add_parser("worker-register", help="bond + register the protocol key (chain)")
    keystore_flags(register)
    register.add_argument("--amount", type=int, required=True)
    register.add_argument("--json", action="store_true")
    register.set_defaults(func=cmd_worker_register)

    faucet = sub.add_parser("faucet", help="testnet faucet request (API, B7)")
    faucet.add_argument("--address", required=True)
    faucet.add_argument("--json", action="store_true")
    faucet.set_defaults(func=cmd_faucet)

    job = sub.add_parser("job", help="developer job surface (API, B6)")
    job_sub = job.add_subparsers(dest="action", required=True)
    job_post = job_sub.add_parser("post", help="submit a job (API)")
    job_post.add_argument("--profile", required=True)
    job_post.add_argument("--input", default="", help="JSON input")
    job_post.add_argument("--idempotency-key", default="")
    job_post.add_argument("--json", action="store_true")
    job_post.set_defaults(func=cmd_job_post)
    job_get = job_sub.add_parser("get", help="job status (API)")
    job_get.add_argument("--job-id", required=True)
    job_get.add_argument("--json", action="store_true")
    job_get.set_defaults(func=cmd_job_get)

    task = sub.add_parser("task", help="task status (chain)")
    task.add_argument("--task-id", type=int, required=True)
    task.add_argument("--json", action="store_true")
    task.set_defaults(func=cmd_task_query)

    receipt = sub.add_parser("receipt", help="VWR lookup (chain)")
    receipt.add_argument("--receipt-id", required=True)
    receipt.add_argument("--json", action="store_true")
    receipt.set_defaults(func=cmd_receipt_query)

    da = sub.add_parser("da", help="DA status for a task (chain)")
    da.add_argument("--task-id", type=int, required=True)
    da.add_argument("--json", action="store_true")
    da.set_defaults(func=cmd_da_query)

    dispute = sub.add_parser("dispute", help="dispute status for a task (chain)")
    dispute.add_argument("--task-id", type=int, required=True)
    dispute.add_argument("--json", action="store_true")
    dispute.set_defaults(func=cmd_dispute_query)

    for name, module, help_text in (("run-worker", "prisma_worker", "run the worker daemon"),
                                    ("run-da", "prisma_da", "run the DA daemon"),
                                    ("run-watcher", "prisma_watcher", "run the watcher daemon")):
        run_cmd = sub.add_parser(name, help=help_text + " (delegates to its own CLI)")
        run_cmd.add_argument("passthrough", nargs="*", help=argparse.SUPPRESS)
        run_cmd.set_defaults(module=module, func=lambda a: _run_component(
            a.module, a, a.passthrough))

    version = sub.add_parser("version", help="version metadata (from the release build)")
    version.add_argument("--json", action="store_true")
    version.set_defaults(func=cmd_version)

    config_cmd = sub.add_parser("config", help="operator config management")
    config_sub = config_cmd.add_subparsers(dest="action", required=True)
    config_set = config_sub.add_parser("set", help="set and save the config")
    for attr in ("chain_id", "node", "api", "prismad", "keystore", "faucet"):
        config_set.add_argument(f"--{attr.replace('_', '-')}", default=None)
    config_set.add_argument("--json", action="store_true")
    config_set.set_defaults(func=cmd_config_set)
    config_show = config_sub.add_parser("show", help="print and validate the config")
    config_show.add_argument("--json", action="store_true")
    config_show.set_defaults(func=cmd_config_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (CliError, worker_keystore.KeystoreError, OSError) as exc:
        print(f"prisma: {exc}", file=sys.stderr)
        return 1
