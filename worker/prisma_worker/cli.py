"""``prisma-worker`` command line (B2-01): identity lifecycle.

    prisma-worker identity init   [--keystore PATH] [--json]
    prisma-worker identity import --protocol-key-hex HEX [--account-key-hex HEX | --account-key-file FILE]
    prisma-worker identity show   [--keystore PATH] [--json]
    prisma-worker identity rotate [--keystore PATH] [--json]

Passphrase sources, in order: ``--passphrase-stdin`` (reads one line),
``PRISMA_WORKER_PASSPHRASE``, interactive prompt. Nothing is ever written
to disk in the clear and nothing secret is printed or logged.
"""

from __future__ import annotations

import argparse
import getpass
import json
import pathlib
import sys

from . import chain, gpu, jobs, keystore, model
from .identity import WorkerIdentity
from .join import JoinConfig, JoinError, join as run_join, heartbeat as run_heartbeat
from .redact import install as install_redaction
from .redact import register_secret


class CliError(Exception):
    """User-facing command failure: printed, exit code 1."""


def _passphrase(args: argparse.Namespace, confirm: bool = False) -> str:
    import os

    if getattr(args, "passphrase_stdin", False):
        passphrase = sys.stdin.readline().rstrip("\n")
    elif os.environ.get("PRISMA_WORKER_PASSPHRASE"):
        passphrase = os.environ["PRISMA_WORKER_PASSPHRASE"]
    else:
        passphrase = getpass.getpass("keystore passphrase: ")
        if confirm:
            again = getpass.getpass("confirm passphrase: ")
            if passphrase != again:
                raise CliError("passphrases do not match")
    if not passphrase:
        raise CliError("an empty passphrase is refused")
    return passphrase


def _print_record(record: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(record, indent=1))
        return
    for key, value in record.items():
        print(f"{key:20s} {value}")


def _register(identity: WorkerIdentity) -> None:
    for secret in identity.secret_material():
        register_secret(secret)


def cmd_identity_init(args: argparse.Namespace) -> int:
    install_redaction()
    identity = WorkerIdentity.generate()
    _register(identity)
    passphrase = _passphrase(args, confirm=True)
    path = keystore.save(args.keystore, identity, passphrase)
    record = identity.public_record()
    record["keystore"] = str(path)
    _print_record(record, args.json)
    return 0


def cmd_identity_import(args: argparse.Namespace) -> int:
    install_redaction()
    protocol_hex = (args.protocol_key_hex or "").strip()
    if not protocol_hex:
        raise CliError("--protocol-key-hex is required (32-byte ed25519, hex)")
    if args.account_key_file:
        account_hex = pathlib.Path(args.account_key_file).read_text(encoding="utf-8").strip()
    else:
        account_hex = (args.account_key_hex or "").strip()
    if not account_hex:
        raise CliError("provide --account-key-hex or --account-key-file (32-byte secp256k1, hex)")
    identity = WorkerIdentity.from_protocol_hex(protocol_hex, bytes.fromhex(
        account_hex[2:] if account_hex.startswith("0x") else account_hex))
    _register(identity)
    passphrase = _passphrase(args, confirm=True)
    path = keystore.save(args.keystore, identity, passphrase)
    record = identity.public_record()
    record["keystore"] = str(path)
    _print_record(record, args.json)
    return 0


def cmd_identity_show(args: argparse.Namespace) -> int:
    record = keystore.public_info(args.keystore)
    record["keystore"] = str(args.keystore)
    _print_record(record, args.json)
    return 0


def cmd_identity_rotate(args: argparse.Namespace) -> int:
    install_redaction()
    passphrase = _passphrase(args)
    old_public, new_identity = keystore.rotate(args.keystore, passphrase)
    _register(new_identity)
    record = {
        "retired": old_public,
        "announce_before_retiring": new_identity.public_record(),
        "note": ("announce the new protocol key to the chain (network-key binding) before the old "
                 "one is removed anywhere; keep the old key usable until the announcement is accepted"),
    }
    _print_record(record, args.json)
    return 0


def _chain_config(args: argparse.Namespace) -> chain.ChainConfig:
    return chain.ChainConfig(rpc_urls=[u.strip() for u in args.node.split(",") if u.strip()],
                             chain_id=args.chain_id, prismad=args.prismad)


def cmd_join(args: argparse.Namespace) -> int:
    install_redaction()
    identity = keystore.load(args.keystore, _passphrase(args))
    _register(identity)
    config = JoinConfig(chain=_chain_config(args), bond_amount_uprsm=args.bond,
                        state_path=pathlib.Path(args.state).expanduser(),
                        allow_unsupported_reason=args.allow_unsupported or "")
    try:
        state = run_join(config, identity)
    except (chain.ChainError, JoinError) as exc:
        print(f"prisma-worker: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(state, indent=1))
    else:
        print(f"joined chain {state['chain_id']} as {state['account_address']} "
              f"(node_id {state['node_id']}, bonded {state['bonded_uprsm']} uprsm, "
              f"status {state['capability']['status']})")
        print(f"state: {config.state_path}")
    return 0


def cmd_heartbeat(args: argparse.Namespace) -> int:
    install_redaction()
    identity = keystore.load(args.keystore, _passphrase(args))
    _register(identity)
    config = JoinConfig(chain=_chain_config(args), state_path=pathlib.Path(args.state).expanduser())
    try:
        state = run_heartbeat(config, identity, interval_seconds=args.interval,
                              iterations=args.iterations)
    except (chain.ChainError, JoinError) as exc:
        print(f"prisma-worker: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(state, indent=1))
    return 0


def _journal(args: argparse.Namespace) -> jobs.Journal:
    directory = pathlib.Path(args.journal).expanduser() if getattr(args, "journal", "") else (
        pathlib.Path(args.state).expanduser().parent / "journal")
    return jobs.Journal(directory)


def cmd_jobs_discover(args: argparse.Namespace) -> int:
    install_redaction()
    client = chain.ChainClient(_chain_config(args))
    try:
        client.validate_chain_id()
        found = jobs.discover(client, start_id=args.start_id, max_scan=args.max_scan)
    except chain.ChainError as exc:
        print(f"prisma-worker: {exc}", file=sys.stderr)
        return 1
    rows = []
    for task in found:
        ok, reason = jobs.compatibility(task, model_ids=tuple(args.model_id) if args.model_id
                                        else jobs.FROZEN_PROFILE_MODEL_IDS,
                                        modes=tuple(args.mode) if args.mode else jobs.FROZEN_MODES)
        rows.append({"task_id": task.task_id, "model_id": task.model_id, "mode": task.mode,
                     "spec_version": task.spec_version, "deadline": task.deadline,
                     "compatible": ok, "reason": reason})
    if args.json:
        print(json.dumps(rows, indent=1))
    else:
        if not rows:
            print("no open tasks found")
        for row in rows:
            print(f"task {row['task_id']}: model={row['model_id']} mode={row['mode']} "
                  f"deadline={row['deadline']} {'COMPATIBLE' if row['compatible'] else 'skip: ' + row['reason']}")
    return 0


def cmd_jobs_accept(args: argparse.Namespace) -> int:
    install_redaction()
    identity = keystore.load(args.keystore, _passphrase(args))
    _register(identity)
    client = chain.ChainClient(_chain_config(args))
    journal = _journal(args)
    # Idempotent first: a task this worker already accepted is reported as-is,
    # without touching the chain (duplicate/late acceptance cannot double-execute).
    existing = journal.read(args.task_id)
    if existing and existing.get("phase") in (jobs.PHASE_ACCEPTED, jobs.PHASE_EXECUTING,
                                              jobs.PHASE_SUBMITTED, jobs.PHASE_TERMINAL):
        print(json.dumps(existing, indent=1) if args.json else
              f"task {args.task_id}: already {existing['phase']} "
              f"(tx {existing.get('txhash', 'n/a')}); nothing submitted")
        return 0
    try:
        client.validate_chain_id()
        payload = client.task(args.task_id)
    except chain.ChainError as exc:
        print(f"prisma-worker: {exc}", file=sys.stderr)
        return 1
    if not payload:
        print(f"prisma-worker: task {args.task_id} does not exist", file=sys.stderr)
        return 1
    task = jobs.TaskSummary.from_chain(payload)
    model_ids = tuple(args.model_id) if args.model_id else jobs.FROZEN_PROFILE_MODEL_IDS
    ok, reason = jobs.compatibility(task, model_ids=model_ids,
                                    modes=tuple(args.mode) if args.mode else jobs.FROZEN_MODES)
    if not ok:
        print(f"prisma-worker: refusing task {task.task_id}: {reason}", file=sys.stderr)
        return 1
    height = int(client.status().get("sync_info", {}).get("latest_block_height", "0") or 0)
    try:
        entry = jobs.accept_task(client, journal, task,
                                 account_scalar_hex=identity.account_scalar.hex(),
                                 current_height=height, safety_margin=args.safety_margin)
    except (chain.ChainError, jobs.JobError) as exc:
        print(f"prisma-worker: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(entry, indent=1) if args.json else
          f"task {task.task_id}: {entry['phase']} (tx {entry.get('txhash', 'n/a')})")
    return 0


def cmd_model_build_manifest(args: argparse.Namespace) -> int:
    manifest = model.build_manifest(pathlib.Path(args.dir).expanduser(), model_id=args.model_id,
                                    spec_version=args.spec_version,
                                    digests={d.split("=", 1)[0]: d.split("=", 1)[1]
                                             for d in args.digest})
    print(json.dumps(manifest.to_dict(), indent=1) if args.json else
          f"manifest written for {manifest.model_id}/{manifest.spec_version} "
          f"({len(manifest.files)} files, graph_id_v2 {manifest.graph_id_v2[:12]}…)")
    return 0


def cmd_model_verify(args: argparse.Namespace) -> int:
    try:
        manifest = model.verify(pathlib.Path(args.dir).expanduser())
    except model.ProfileError as exc:
        print(f"prisma-worker: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest.to_dict(), indent=1) if args.json else
          f"profile {manifest.model_id}/{manifest.spec_version} verified "
          f"(graph_id_v2 {manifest.graph_id_v2[:12]}…, policy_id {manifest.policy_id[:12]}…)")
    return 0


def cmd_model_install(args: argparse.Namespace) -> int:
    try:
        report = model.install(args.source, pathlib.Path(args.models_dir).expanduser(),
                              model_id=args.model_id or None,
                              spec_version=args.spec_version or None)
    except model.ProfileError as exc:
        print(f"prisma-worker: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"model_id": report.model_id, "spec_version": report.spec_version,
                          "files": report.files, "cached": report.cached,
                          "resumed": report.resumed}, indent=1))
    else:
        print(f"installed {report.model_id}/{report.spec_version}: {len(report.files)} files "
              f"({len(report.cached)} from cache, {len(report.resumed)} resumed)")
    return 0


def cmd_model_check(args: argparse.Namespace) -> int:
    directory = pathlib.Path(args.dir).expanduser()
    try:
        manifest = model.verify(directory)
        client = chain.ChainClient(_chain_config(args))
        client.validate_chain_id()
        remote = client.model(manifest.model_id, manifest.spec_version)
    except (model.ProfileError, chain.ChainError) as exc:
        print(f"prisma-worker: {exc}", file=sys.stderr)
        return 1
    ok, reason = model.chain_check(manifest, remote)
    if args.json:
        print(json.dumps({"ok": ok, "reason": reason, "model_id": manifest.model_id,
                          "spec_version": manifest.spec_version}, indent=1))
    else:
        print(("OK   " if ok else "FAIL ") + reason)
    return 0 if ok else 1


def cmd_gpu_probe(args: argparse.Namespace) -> int:
    try:
        report = gpu.probe(nvidia_smi=args.nvidia_smi)
    except gpu.ProbeError as exc:
        print(f"prisma-worker: {exc}", file=sys.stderr)
        return 1
    print(gpu.format_report(report, args.json))
    # Exit codes: 0 supported, 2 CAPABILITY_UNSUPPORTED (the join flow branches
    # on this without parsing text), 1 probe error.
    return 0 if report.backend_supported else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prisma-worker", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="group", required=True)
    identity = sub.add_parser("identity", help="worker identity lifecycle")
    actions = identity.add_subparsers(dest="action", required=True)

    def common(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--keystore", default=str(keystore.default_path()),
                         help="keystore path (default: %(default)s)")
        cmd.add_argument("--json", action="store_true", help="machine-readable output")
        cmd.add_argument("--passphrase-stdin", action="store_true",
                         help="read the passphrase from stdin (first line)")

    init = actions.add_parser("init", help="generate a new identity and store it encrypted")
    common(init)
    init.set_defaults(func=cmd_identity_init)

    imp = actions.add_parser("import", help="import an existing operator key")
    common(imp)
    imp.add_argument("--protocol-key-hex", required=True, help="ed25519 protocol key, hex")
    imp.add_argument("--account-key-hex", default="", help="secp256k1 account key, hex")
    imp.add_argument("--account-key-file", default="", help="file containing the account key hex")
    imp.set_defaults(func=cmd_identity_import)

    show = actions.add_parser("show", help="print the public record (no passphrase needed)")
    common(show)
    show.set_defaults(func=cmd_identity_show)

    rotate = actions.add_parser("rotate", help="generate a new identity over the same keystore")
    common(rotate)
    rotate.set_defaults(func=cmd_identity_rotate)

    def chain_flags(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--chain-id", required=True, help="expected chain id (refused on mismatch)")
        cmd.add_argument("--node", required=True, help="RPC endpoint(s), comma-separated for fail-over")
        cmd.add_argument("--prismad", default="prismad", help="prismad binary for queries/transactions")
        cmd.add_argument("--state", default=str(pathlib.Path.home() / ".prisma-worker" / "state.json"),
                         help="worker state file (default: %(default)s)")

    join_cmd = sub.add_parser("join", help="join the network: validate, bond, register")
    common(join_cmd)
    chain_flags(join_cmd)
    join_cmd.add_argument("--bond", type=int, default=1_000_000,
                          help="bond target in uprsm (default: %(default)s = frozen MinBond)")
    join_cmd.add_argument("--allow-unsupported", default="",
                          help="join even if the GPU is not a proven target; REASON is recorded")
    join_cmd.set_defaults(func=cmd_join)

    hb = sub.add_parser("heartbeat", help="periodic liveness + capability refresh")
    common(hb)
    chain_flags(hb)
    hb.add_argument("--interval", type=float, default=30.0, help="seconds between heartbeats")
    hb.add_argument("--iterations", type=int, default=0, help="stop after N ticks (0 = forever)")
    hb.set_defaults(func=cmd_heartbeat)

    jobs_cmd = sub.add_parser("jobs", help="job discovery and acceptance")
    jobs_actions = jobs_cmd.add_subparsers(dest="action", required=True)

    def jobs_common(cmd: argparse.ArgumentParser) -> None:
        chain_flags(cmd)
        cmd.add_argument("--keystore", default=str(keystore.default_path()),
                         help="keystore path (default: %(default)s)")
        cmd.add_argument("--passphrase-stdin", action="store_true",
                         help="read the passphrase from stdin (first line)")
        cmd.add_argument("--journal", default="", help="journal directory (default: <state dir>/journal)")
        cmd.add_argument("--model-id", action="append", default=[],
                         help="accepted model id (repeatable; default: the frozen profile)")
        cmd.add_argument("--mode", action="append", default=[],
                         help="accepted task mode (repeatable; default: verifiable)")

    discover = jobs_actions.add_parser("discover", help="list open tasks and compatibility")
    jobs_common(discover)
    discover.add_argument("--start-id", type=int, default=1)
    discover.add_argument("--max-scan", type=int, default=512)
    discover.add_argument("--json", action="store_true")
    discover.set_defaults(func=cmd_jobs_discover)

    accept = jobs_actions.add_parser("accept", help="accept one compatible task (journaled)")
    jobs_common(accept)
    accept.add_argument("--task-id", type=int, required=True)
    accept.add_argument("--safety-margin", type=int, default=5,
                        help="refuse tasks whose deadline is within this many blocks")
    accept.add_argument("--json", action="store_true")
    accept.set_defaults(func=cmd_jobs_accept)

    model_cmd = sub.add_parser("model", help="frozen model/profile artifacts")
    model_actions = model_cmd.add_subparsers(dest="action", required=True)

    build = model_actions.add_parser("build-manifest", help="write a manifest for an artifact dir")
    build.add_argument("--dir", required=True)
    build.add_argument("--model-id", required=True)
    build.add_argument("--spec-version", required=True)
    build.add_argument("--digest", action="append", default=[],
                       help="name=value digest recorded in the manifest (repeatable)")
    build.add_argument("--json", action="store_true")
    build.set_defaults(func=cmd_model_build_manifest)

    verify = model_actions.add_parser("verify", help="verify a profile directory end to end")
    verify.add_argument("--dir", required=True)
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=cmd_model_verify)

    install = model_actions.add_parser("install", help="resumable install into the model cache")
    install.add_argument("--from", dest="source", required=True, help="directory path or URL base")
    install.add_argument("--models-dir", required=True)
    install.add_argument("--model-id", default="")
    install.add_argument("--spec-version", default="")
    install.add_argument("--json", action="store_true")
    install.set_defaults(func=cmd_model_install)

    check = model_actions.add_parser("check", help="verify locally and against the registered model")
    chain_flags(check)
    check.add_argument("--dir", required=True)
    check.add_argument("--json", action="store_true")
    check.set_defaults(func=cmd_model_check)

    gpu_cmd = sub.add_parser("gpu", help="GPU capability probe")
    gpu_actions = gpu_cmd.add_subparsers(dest="action", required=True)
    probe = gpu_actions.add_parser("probe", help="report GPU model/CC/VRAM/driver and backend support")
    probe.add_argument("--nvidia-smi", default=None, help="path to nvidia-smi (default: PATH lookup)")
    probe.add_argument("--json", action="store_true", help="machine-readable output")
    probe.set_defaults(func=cmd_gpu_probe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Only the identity/join/heartbeat subcommands carry a keystore path.
    if hasattr(args, "keystore"):
        args.keystore = pathlib.Path(args.keystore).expanduser()
    if hasattr(args, "iterations") and not args.iterations:
        args.iterations = None
    try:
        return args.func(args)
    except (keystore.KeystoreError, CliError, ValueError) as exc:
        print(f"prisma-worker: {exc}", file=sys.stderr)
        return 1
