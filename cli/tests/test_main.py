"""B5-01: the prisma CLI surface — config, honest ownership markers, chain
delegation and daemon launchers (with a fake exec target)."""

from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
for extra in ("cli", "worker", "da", "watcher", "network"):
    sys.path.insert(0, str(REPO / extra))

from prisma_cli.common import CliError, GlobalConfig  # noqa: E402
from prisma_cli.main import build_parser, main  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMA_CLI_CONFIG", str(tmp_path / "cli.json"))
    monkeypatch.setenv("PRISMA_CLI_PASSPHRASE", "test passphrase")
    return tmp_path


def test_config_set_show_and_validation():
    assert main(["config", "set", "--chain-id", "prisma-testnet-1",
                 "--node", "http://127.0.0.1:26657", "--faucet", "http://127.0.0.1:9000"]) == 0
    config = GlobalConfig.load()
    assert config.chain_id == "prisma-testnet-1" and config.faucet.endswith(":9000")
    # an invalid chain id is refused before saving
    assert main(["config", "set", "--chain-id", "not-a-prisma-id"]) == 1
    assert GlobalConfig.load().chain_id == "prisma-testnet-1", "the invalid set must not save"
    assert main(["config", "show"]) == 0


def test_wallet_init_and_show(tmp_path):
    keystore = tmp_path / "k.json"
    assert main(["wallet", "init", "--keystore", str(keystore), "--json"]) == 0
    assert json.loads(keystore.read_text())["public"]["account_address"].startswith("prsm1")
    assert main(["wallet", "show", "--keystore", str(keystore), "--json"]) == 0


def test_wallet_show_requires_a_keystore(capsys):
    assert main(["wallet", "show", "--json"]) == 1
    assert "no operator keystore" in capsys.readouterr().err


def test_version_delegates_to_prismad(monkeypatch, capsys):
    import prisma_cli.main as main_module

    def fake_prismad(config, *args, **kw):
        assert args[0] == "version"
        return {"software_version": "test-1", "protocol": {"freeze_tag": "x"}}

    monkeypatch.setattr(main_module, "prismad", fake_prismad)
    assert main(["version", "--json"]) == 0
    assert "test-1" in capsys.readouterr().out


def test_version_reports_an_unreachable_node(capsys):
    assert main(["--prismad", "definitely-not-a-binary", "version", "--json"]) == 1


def test_api_groups_refuse_without_configuration(capsys):
    assert main(["job", "post", "--profile", "x"]) == 1
    assert "no Job API configured" in capsys.readouterr().err
    assert main(["faucet", "--address", "prsm1abc"]) == 1
    assert "no faucet endpoint" in capsys.readouterr().err


def test_job_post_uses_the_api_not_a_chain_tx(monkeypatch):
    assert main(["config", "set", "--api", "http://127.0.0.1:8080"]) == 0
    seen: dict = {}

    def fake_http(method, url, **kw):
        seen["method"], seen["url"], seen["body"] = method, url, kw.get("body")
        return 200, {"job_id": "j-1"}

    from prisma_cli import main as main_module

    monkeypatch.setattr(main_module, "http_json", fake_http)
    assert main(["job", "post", "--profile", "qwen3-0.6b-layer0-v2",
                 "--input", '{"x": 1}', "--idempotency-key", "abc", "--json"]) == 0
    assert seen["method"] == "POST" and seen["url"].endswith("/v1/jobs")
    assert seen["body"]["idempotency_key"] == "abc"


def test_faucet_uses_the_configured_endpoint(monkeypatch):
    assert main(["config", "set", "--faucet", "http://127.0.0.1:9000"]) == 0
    seen: dict = {}

    def fake_http(method, url, **kw):
        seen["method"], seen["url"] = method, url
        return 200, {"funded": True}

    from prisma_cli import main as main_module

    monkeypatch.setattr(main_module, "http_json", fake_http)
    assert main(["faucet", "--address", "prsm1abc", "--json"]) == 0
    assert seen["url"] == "http://127.0.0.1:9000/fund"


def test_chain_queries_delegate_to_prismad(monkeypatch):
    import prisma_cli.main as main_module

    def fake_prismad(config, *args, **kw):
        if args[:2] == ("query", "compute") and "task" in args:
            return {"task_json": __import__("base64").b64encode(
                json.dumps({"id": 5, "status": "result_submitted"}).encode()).decode()}
        return {"ok": True}

    monkeypatch.setattr(main_module, "prismad", fake_prismad)
    assert main(["task", "--task-id", "5", "--json"]) == 0
    assert main(["da", "--task-id", "5", "--json"]) == 0
    assert main(["dispute", "--task-id", "5", "--json"]) == 0


def test_all_expected_commands_exist():
    parser = build_parser()
    group_action = next(a for a in parser._actions if a.dest == "group")
    groups = set(group_action.choices)
    for expected in ("wallet", "balance", "bond", "worker-register", "faucet", "job",
                     "task", "receipt", "da", "dispute", "run-worker", "run-da",
                     "run-watcher", "version", "config"):
        assert expected in groups, f"missing command group: {expected}"
