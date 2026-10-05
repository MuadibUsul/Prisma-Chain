"""``prisma-watcher`` command line: init / run / status / verify.

    prisma-watcher init   --chain-id ID --node URL [--provider URL]... [--config PATH]
    prisma-watcher run    [--config PATH] [--once] [--interval N]
    prisma-watcher status [--config PATH] [--json]
    prisma-watcher verify --bundle PATH --task-id N [--json]

The challenger key follows the worker's identity rules (B2-01). The daemon
verifies with the frozen WatcherV2 and journals every task, so a restart
resumes instead of re-verifying.
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

from .chain_scan import PrismadTaskScan
from .config import ChainConfig, DaemonConfig
from .daemon import WatcherDaemon
from .frozen import FrozenLibraryMissing
from .sources import FrozenVerifier, HttpBundleSource


class CliError(Exception):
    """User-facing command failure: printed, exit code 1."""


def _load_config(path_arg: str) -> DaemonConfig:
    path = pathlib.Path(path_arg).expanduser()
    if not path.exists():
        raise CliError(f"no config at {path}; run `prisma-watcher init` first")
    return DaemonConfig.load(path)


def _passphrase(args: argparse.Namespace) -> str:
    if getattr(args, "passphrase_stdin", False):
        passphrase = sys.stdin.readline().rstrip()
    elif os.environ.get("PRISMA_WATCHER_PASSPHRASE"):
        passphrase = os.environ["PRISMA_WATCHER_PASSPHRASE"]
    else:
        passphrase = getpass.getpass("challenger key passphrase: ")
    if not passphrase:
        raise CliError("an empty passphrase is refused")
    return passphrase


def cmd_init(args: argparse.Namespace) -> int:
    install_redaction()
    path = pathlib.Path(args.config).expanduser()
    if path.exists() and not args.force:
        raise CliError(f"{path} already exists; pass --force to replace it")
    identity = WorkerIdentity.generate()
    for secret in identity.secret_material():
        register_secret(secret)
    passphrase = _passphrase(args)
    config = DaemonConfig(account=identity.account_address,
                          keystore=str(path.parent / "challenger-key.json"),
                          journal_dir=args.journal_dir or str(path.parent / "journal"),
                          providers=list(args.provider),
                          chain=ChainConfig(rpc_urls=[u.strip() for u in args.node.split(",")
                                                      if u.strip()],
                                            chain_id=args.chain_id, prismad=args.prismad),
                          interval_seconds=args.interval)
    worker_keystore.save(pathlib.Path(config.keystore), identity, passphrase)
    config.save(path)
    record = identity.public_record()
    record.update({"config": str(path), "keystore": config.keystore,
                   "providers": config.providers})
    print(json.dumps(record, indent=1) if args.json else
          f"prisma-watcher initialised\n  challenger account : {record['account_address']}\n"
          f"  journal            : {config.journal_dir}\n  providers          : "
          f"{', '.join(config.providers) or '(none configured)'}")
    print("next: prisma-watcher run", file=sys.stderr if args.json else sys.stdout)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    identity = worker_keystore.load(pathlib.Path(config.keystore), _passphrase(args))
    for secret in identity.secret_material():
        register_secret(secret)
    daemon = WatcherDaemon(PrismadTaskScan(prismad=config.chain.prismad,
                                           chain_id=config.chain.chain_id,
                                           rpc_urls=config.chain.rpc_urls,
                                           max_scan=config.max_scan),
                           HttpBundleSource(config.providers), FrozenVerifier(),
                           config.resolved_journal_dir(), max_scan=config.max_scan)
    daemon.run(interval_seconds=args.interval or config.interval_seconds,
               max_iterations=1 if args.once else None)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    journal = config.resolved_journal_dir()
    entries = []
    for path in sorted(journal.glob("task-*.json")):
        entries.append(json.loads(path.read_text(encoding="utf-8")))
    by_phase: dict = {}
    verdicts: dict = {}
    for entry in entries:
        phase = entry.get("phase", "?")
        by_phase[phase] = by_phase.get(phase, 0) + 1
        if entry.get("verdict"):
            verdicts[entry["verdict"]] = verdicts.get(entry["verdict"], 0) + 1
    payload = {"account": config.account, "journal": str(journal), "tasks": len(entries),
               "by_phase": by_phase, "verdicts": verdicts, "providers": config.providers}
    print(json.dumps(payload, indent=1) if args.json else
          f"watcher {config.account or '(unset)'}\n  journal : {journal}\n"
          f"  tasks   : {len(entries)} {by_phase}\n  verdicts: {verdicts}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    task = {"id": args.task_id, "task_id": args.task_id, "status": "result_submitted"}
    try:
        result = FrozenVerifier().verify(pathlib.Path(args.bundle), task)
    except (FrozenLibraryMissing, OSError) as exc:
        print(f"prisma-watcher: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1) if args.json else
          f"task {args.task_id}: verdict={result['verdict']}")
    return 0 if result["verdict"] == "clean" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prisma-watcher", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--config", default=str(DaemonConfig.default_path()))
        cmd.add_argument("--json", action="store_true")
        cmd.add_argument("--passphrase-stdin", action="store_true")

    init = sub.add_parser("init", help="create the challenger key and config")
    common(init)
    init.add_argument("--chain-id", required=True)
    init.add_argument("--node", default="http://127.0.0.1:26657")
    init.add_argument("--prismad", default="prismad")
    init.add_argument("--provider", action="append", default=[], help="DA endpoint (repeatable)")
    init.add_argument("--journal-dir", default="")
    init.add_argument("--interval", type=float, default=15.0)
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    run = sub.add_parser("run", help="discover, retrieve, verify, journal")
    common(run)
    run.add_argument("--once", action="store_true", help="one pass and exit")
    run.add_argument("--interval", type=float, default=0.0)
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="journal summary")
    common(status)
    status.set_defaults(func=cmd_status)

    verify = sub.add_parser("verify", help="verify one bundle (one-off)")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--task-id", type=int, required=True)
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (CliError, worker_keystore.KeystoreError, OSError) as exc:
        print(f"prisma-watcher: {exc}", file=sys.stderr)
        return 1
