"""``prisma-da`` command line: init / register / run / status.

    prisma-da init     [--config PATH] [--data-dir DIR] [--listen HOST:PORT] --chain-id ID --node URL
    prisma-da register [--config PATH]
    prisma-da run      [--config PATH]
    prisma-da status   [--config PATH] [--json]

The provider key follows the same rules as the worker (B2-01): an encrypted
keystore, 0600, passphrase via ``--passphrase-stdin``/``PRISMA_DA_PASSPHRASE``
or an interactive prompt, and no key material in any log line.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import pathlib
import sys

from prisma_worker import keystore as worker_keystore
from prisma_worker.identity import WorkerIdentity
from prisma_worker.redact import install as install_redaction
from prisma_worker.redact import register_secret

from .chain_tx import ChainTxError, provider_keyring, run_tx
from .config import ChainConfig, DaemonConfig
from .daemon import Daemon
from .chain_ops import PrismadChainOps
from .policy import RetentionPolicy, gc as run_gc, integrity_scan
from .responder import ChallengeResponder
from .storage import ArtifactIndex


class CliError(Exception):
    """User-facing command failure: printed, exit code 1."""


def _passphrase(args: argparse.Namespace, confirm: bool = False) -> str:
    if getattr(args, "passphrase_stdin", False):
        passphrase = sys.stdin.readline().rstrip("\n")
    elif os.environ.get("PRISMA_DA_PASSPHRASE"):
        passphrase = os.environ["PRISMA_DA_PASSPHRASE"]
    else:
        passphrase = getpass.getpass("provider key passphrase: ")
        if confirm:
            again = getpass.getpass("confirm passphrase: ")
            if passphrase != again:
                raise CliError("passphrases do not match")
    if not passphrase:
        raise CliError("an empty passphrase is refused")
    return passphrase


def _load_config(args: argparse.Namespace) -> DaemonConfig:
    path = pathlib.Path(args.config).expanduser()
    if not path.exists():
        raise CliError(f"no config at {path}; run `prisma-da init` first")
    return DaemonConfig.load(path)


def cmd_init(args: argparse.Namespace) -> int:
    install_redaction()
    path = pathlib.Path(args.config).expanduser()
    if path.exists() and not args.force:
        raise CliError(f"{path} already exists; pass --force to replace it")
    host, _, port = args.listen.partition(":")
    identity = WorkerIdentity.generate()
    for secret in identity.secret_material():
        register_secret(secret)
    passphrase = _passphrase(args, confirm=True)
    config = DaemonConfig(
        account=identity.account_address,
        listen_host=host or "127.0.0.1",
        listen_port=int(port or 8401),
        data_dir=args.data_dir or str(path.parent / "data"),
        keystore=str(path.parent / "provider-key.json"),
        chain=ChainConfig(rpc_urls=[u.strip() for u in args.node.split(",") if u.strip()],
                          chain_id=args.chain_id, prismad=args.prismad))
    worker_keystore.save(pathlib.Path(config.keystore), identity, passphrase)
    config.save(path)
    record = identity.public_record()
    record.update({"config": str(path), "keystore": config.keystore,
                   "listen": f"{config.listen_host}:{config.listen_port}",
                   "data_dir": config.data_dir})
    print(json.dumps(record, indent=1) if args.json else
          f"prisma-da initialised\n  provider account : {record['account_address']}\n"
          f"  node id          : {record['node_id']}\n"
          f"  keystore         : {record['keystore']}\n"
          f"  config           : {record['config']}")
    print("next: prisma-da register (needs the account funded), then prisma-da run",
          file=sys.stderr if args.json else sys.stdout)
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    config = _load_config(args)
    identity = worker_keystore.load(pathlib.Path(config.keystore), _passphrase(args))
    for secret in identity.secret_material():
        register_secret(secret)
    if not config.chain.rpc_urls or not config.chain.chain_id:
        raise CliError("config has no chain endpoints; re-run init with --node/--chain-id")
    rpc = config.chain.rpc_urls[0]
    with provider_keyring(identity.account_scalar.hex(), prismad=config.chain.prismad,
                          chain_id=config.chain.chain_id, rpc_url=rpc) as keyring:
        txhash = run_tx(keyring, ["compute", "register-da-provider"],
                        prismad=config.chain.prismad, chain_id=config.chain.chain_id, rpc_url=rpc)
    print(json.dumps({"registered": True, "provider": config.account, "txhash": txhash}, indent=1)
          if args.json else f"provider {config.account} registered (tx {txhash})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args)
    identity = worker_keystore.load(pathlib.Path(config.keystore), _passphrase(args))
    for secret in identity.secret_material():
        register_secret(secret)
    daemon = Daemon(config, identity)
    daemon.serve(run_forever=not args.once)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = _load_config(args)
    index = ArtifactIndex(config.resolved_data_dir())
    report = index.rescan()
    payload = {"provider": config.account, "data_dir": str(index.root),
               "listen": f"{config.listen_host}:{config.listen_port}",
               "index": index.metrics(), "rescan": report,
               "quota_bytes": config.quota_bytes, "ttl_seconds": config.ttl_seconds}
    print(json.dumps(payload, indent=1) if args.json else
          f"provider {config.account or '(unset)'}\n"
          f"  data dir : {index.root}\n  artifacts: {payload['index']['artifacts']} "
          f"({payload['index']['bytes']} bytes)\n  serving  : {report['serving']} "
          f"scanned, {len(report['bad'])} quarantined at last scan")
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    config = _load_config(args)
    index = ArtifactIndex(config.resolved_data_dir())
    index.rescan()
    policy = RetentionPolicy(quota_bytes=config.quota_bytes,
                             ttl_seconds=args.ttl_seconds if args.ttl_seconds is not None
                             else config.ttl_seconds)
    report = run_gc(index, policy, live_challenges=args.live_challenge, dry_run=args.dry_run)
    print(json.dumps(report, indent=1) if args.json else
          f"gc: deleted {len(report['deleted'])} artifact(s), freed {report['freed_bytes']} bytes, "
          f"kept {len(report['kept'])}{' (dry run)' if report['dry_run'] else ''}")
    return 0


def cmd_responder(args: argparse.Namespace) -> int:
    """Answer on-chain chunk challenges inside their deadlines (B4-03)."""
    config = _load_config(args)
    identity = worker_keystore.load(pathlib.Path(config.keystore), _passphrase(args))
    for secret in identity.secret_material():
        register_secret(secret)
    index = ArtifactIndex(config.resolved_data_dir())
    index.rescan()
    ops = PrismadChainOps(account=config.account, account_scalar_hex=identity.account_scalar.hex(),
                          prismad=config.chain.prismad, chain_id=config.chain.chain_id,
                          rpc_urls=config.chain.rpc_urls, storage=index, max_scan=args.max_scan)
    responder = ChallengeResponder(ops, index, config.account,
                                   safety_margin_blocks=args.safety_margin)
    metrics = responder.run(interval_seconds=args.interval,
                            max_iterations=1 if args.once else None)
    if args.json:
        print(json.dumps({"metrics": metrics.to_dict(), "losses": responder.losses}, indent=1))
    else:
        report = metrics.to_dict()
        print(f"responder: discovered={report['discovered']} answered={report['answered']} "
              f"too_late={report['too_late']} lost={report['lost']} failed={report['failed']}")
        for loss in responder.losses:
            print(f"  after-loss: task {loss['task_id']}: {loss['reason']}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    config = _load_config(args)
    index = ArtifactIndex(config.resolved_data_dir())
    index.rescan()
    report = integrity_scan(index)
    print(json.dumps(report, indent=1) if args.json else
          f"integrity scan: {report['healthy']}/{report['checked']} healthy, "
          f"{len(report['bad'])} quarantined")
    return 1 if report["bad"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prisma-da", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--config", default=str(DaemonConfig.default_path()))
        cmd.add_argument("--json", action="store_true")
        cmd.add_argument("--passphrase-stdin", action="store_true")

    init = sub.add_parser("init", help="create the provider key and config")
    common(init)
    init.add_argument("--data-dir", default="", help="artifact root (default: <home>/data)")
    init.add_argument("--listen", default="127.0.0.1:8401", help="host:port for the DA surface")
    init.add_argument("--chain-id", required=True)
    init.add_argument("--node", default="http://127.0.0.1:26657",
                      help="chain RPC endpoint(s), comma-separated")
    init.add_argument("--prismad", default="prismad")
    init.add_argument("--force", action="store_true", help="replace an existing config")
    init.set_defaults(func=cmd_init)

    register = sub.add_parser("register", help="register the provider against the chain")
    common(register)
    register.set_defaults(func=cmd_register)

    run = sub.add_parser("run", help="serve the DA surface")
    common(run)
    run.add_argument("--once", action="store_true",
                     help="start, run restart recovery, and exit (tests/CI)")
    run.set_defaults(func=cmd_run)

    gc_cmd = sub.add_parser("gc", help="delete artifacts past TTL with no live challenge")
    common(gc_cmd)
    gc_cmd.add_argument("--ttl-seconds", type=int, default=None,
                        help="override the configured TTL for this run")
    gc_cmd.add_argument("--live-challenge", type=int, action="append", default=[],
                        help="task id with a live challenge (never deleted); repeatable")
    gc_cmd.add_argument("--dry-run", action="store_true")
    gc_cmd.set_defaults(func=cmd_gc)

    responder = sub.add_parser("responder", help="answer on-chain chunk challenges in time")
    common(responder)
    responder.add_argument("--interval", type=float, default=10.0,
                           help="seconds between challenge polls")
    responder.add_argument("--once", action="store_true", help="one poll and exit (tests/CI)")
    responder.add_argument("--safety-margin", type=int, default=5,
                           help="refuse to start a response within this many blocks of the deadline")
    responder.add_argument("--max-scan", type=int, default=512, help="task scan bound")
    responder.set_defaults(func=cmd_responder)

    scan = sub.add_parser("scan", help="full integrity scan (quarantine mismatches)")
    common(scan)
    scan.set_defaults(func=cmd_scan)

    status = sub.add_parser("status", help="local index and configuration summary")
    common(status)
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (CliError, ChainTxError, worker_keystore.KeystoreError, OSError) as exc:
        print(f"prisma-da: {exc}", file=sys.stderr)
        return 1
