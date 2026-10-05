"""B4-03: challenge answering inside deadlines, objective after-loss reporting,
and the GC link (live challenges are never deleted)."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
for extra in ("da", "worker", "network", "compute/canonical/python", "compute/gemmv1/python"):
    sys.path.insert(0, str(REPO / extra))

from prisma_da.chain_ops import PrismadChainOps  # noqa: E402
from prisma_da.frozen import frozen  # noqa: E402
from prisma_da.policy import RetentionPolicy, gc  # noqa: E402
from prisma_da.responder import ChallengeResponder  # noqa: E402
from prisma_da.storage import ArtifactIndex  # noqa: E402

M, N = 2, 2
TASK_ID = 77
TASK_ID32 = "55" * 32
ASSIGNMENT = "66" * 32
VALUES = [11, 12, 13, 14]
PROVIDER = "prsm1provider"


def blob_and_root():
    blob = b"".join(int(v).to_bytes(4, "big", signed=True) for v in VALUES)
    tiles = frozen.tensors.output_tiles(list(VALUES), M, N)
    cols_c = (N + 7) // 8
    leaves = [frozen.protocol.leaf_output_tile(bytes.fromhex(TASK_ID32), bytes.fromhex(ASSIGNMENT),
                                               i // cols_c, i % cols_c,
                                               frozen.tensors.int32s_to_canonical(tile))
              for i, tile in enumerate(tiles)]
    return blob, frozen.protocol.merkle_root(leaves).hex()


def stored_index(tmp_path) -> ArtifactIndex:
    index = ArtifactIndex(tmp_path / "data")
    blob, root = blob_and_root()
    index.store(task_id=TASK_ID, c_hex=blob.hex(), output_root_hex=root, m=M, n=N,
                task_id32_hex=TASK_ID32, assignment_id_hex=ASSIGNMENT)
    return index


class FakeChain:
    def __init__(self, challenges, *, height=100, fail_discovery=False, fail_tx=False):
        self.challenges = challenges
        self._height = height
        self.fail_discovery = fail_discovery
        self.fail_tx = fail_tx
        self.responses: list[dict] = []

    def height(self):
        return self._height

    def open_challenges(self, provider):
        if self.fail_discovery:
            raise RuntimeError("chain unreachable")
        return [c for c in self.challenges if c.get("provider", provider) == provider]

    def respond(self, challenge, tile):
        if self.fail_tx:
            raise RuntimeError("mempool rejected the response")
        self.responses.append({"challenge": challenge, "tile": tile})
        return "RESPOND_TX"


def challenge(**kw) -> dict:
    base = {"task_id": TASK_ID, "challenge_id": 3, "provider": PROVIDER,
            "deadline_height": 500, "tile_i": 0, "tile_j": 0}
    base.update(kw)
    return base


def test_answers_a_challenge_in_time(tmp_path):
    chain = FakeChain([challenge()])
    responder = ChallengeResponder(chain, stored_index(tmp_path), PROVIDER, log=lambda _m: None)
    results = responder.poll_once()
    assert results[0]["action"] == "answered" and chain.responses
    tile = chain.responses[0]["tile"]
    proof = tile["proof"]
    assert frozen.merkle.verify_inclusion(bytes.fromhex(tile["output_root"]),
                                          bytes.fromhex(tile["leaf"]), proof["index"],
                                          proof["count"],
                                          [bytes.fromhex(s) for s in proof["siblings"]])
    assert responder.metrics.answered == 1 and responder.metrics.discovered == 1
    assert responder.metrics.to_dict()["response_latency_ms"]["last"] >= 0


def test_deadline_inside_the_safety_margin_is_not_answered(tmp_path):
    chain = FakeChain([challenge(deadline_height=103)], height=100)
    responder = ChallengeResponder(chain, stored_index(tmp_path), PROVIDER, safety_margin_blocks=5,
                                   log=lambda _m: None)
    result = responder.poll_once()[0]
    assert result["action"] == "too_late" and chain.responses == []
    assert responder.metrics.too_late == 1


def test_missing_artifact_is_reported_objectively(tmp_path):
    chain = FakeChain([challenge(task_id=1234)])
    responder = ChallengeResponder(chain, stored_index(tmp_path), PROVIDER, log=lambda _m: None)
    result = responder.poll_once()[0]
    assert result["action"] == "lost" and "not stored" in result["reason"]
    assert chain.responses == [] and responder.losses[0]["task_id"] == 1234
    assert responder.metrics.lost == 1


def test_corrupted_artifact_is_reported_as_a_loss(tmp_path):
    index = stored_index(tmp_path)
    (index.task_dir(TASK_ID) / "output.bin").write_bytes(b"\x07" * 16)
    chain = FakeChain([challenge()])
    responder = ChallengeResponder(chain, index, PROVIDER, log=lambda _m: None)
    result = responder.poll_once()[0]
    assert result["action"] == "lost" and "content hash" in result["reason"]
    assert chain.responses == [] and responder.metrics.lost == 1


def test_discovery_and_tx_failures_are_counted_not_fatal(tmp_path):
    responder = ChallengeResponder(FakeChain([], fail_discovery=True), stored_index(tmp_path),
                                   PROVIDER, log=lambda _m: None)
    assert responder.poll_once() == [] and responder.metrics.failed == 1
    failing_tx = ChallengeResponder(FakeChain([challenge()], fail_tx=True), stored_index(tmp_path),
                                    PROVIDER, log=lambda _m: None)
    result = failing_tx.poll_once()[0]
    assert result["action"] == "failed" and failing_tx.metrics.failed == 1


def test_scheduler_runs_bounded_iterations(tmp_path):
    chain = FakeChain([challenge()])
    responder = ChallengeResponder(chain, stored_index(tmp_path), PROVIDER, log=lambda _m: None)
    sleeps: list[float] = []
    responder.run(interval_seconds=0.01, max_iterations=2, sleep=sleeps.append)
    assert responder.metrics.discovered == 2 and len(chain.responses) == 2 and sleeps == [0.01]


def test_live_challenges_protect_artifacts_from_gc(tmp_path):
    index = stored_index(tmp_path)
    chain = FakeChain([challenge()])
    responder = ChallengeResponder(chain, index, PROVIDER, log=lambda _m: None)
    live = responder.live_challenges()
    assert live == {TASK_ID}
    original = index.entries[TASK_ID].stored_at_ms
    index.entries[TASK_ID].stored_at_ms = original - 10_000_000
    report = gc(index, RetentionPolicy(ttl_seconds=1), live_challenges=live, now_ms=original)
    assert report["deleted"] == [] and "live challenge" in report["kept"][0]["reason"]


# --- the prismad adapter -----------------------------------------------------

def test_challenge_extraction_reads_the_task_document():
    document = {"id": 5, "status": "result_submitted",
                "da": {"status": "open", "id": 9, "provider": PROVIDER, "deadline_height": 321,
                       "chunk_index": 2}}
    extracted = PrismadChainOps._extract_challenge(document)
    assert extracted["task_id"] == 5 and extracted["challenge_id"] == 9
    assert extracted["deadline_height"] == 321 and extracted["chunk_index"] == 2
    assert PrismadChainOps._extract_challenge({"id": 5, "da": {"status": "answered", "deadline_height": 9}}) is None
    assert PrismadChainOps._extract_challenge({"id": 5, "da": {"status": "open"}}) is None
    assert PrismadChainOps._extract_challenge({"id": 5}) is None


def test_respond_command_shape(tmp_path):
    commands: list[list[str]] = []
    tx_output = json.dumps({"code": 0, "txhash": "ABCD"})

    def runner(command, **_kw):
        commands.append(command)
        if "keys" in command:
            return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")
        if command[1] == "tx":
            return subprocess.CompletedProcess(command, 0, stdout=tx_output, stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    ops = PrismadChainOps(account=PROVIDER, account_scalar_hex="11" * 32, prismad="prismad",
                          chain_id="prisma-test-1", rpc_urls=["http://127.0.0.1:1"],
                          storage=stored_index(tmp_path), runner=runner,
                          http_get=lambda url: {"result": {"tx_result": {"code": 0}}})
    tile = {"tile": "aa" * 64, "proof": {"index": 2, "count": 4, "siblings": ["bb" * 32, "cc" * 32]},
            "leaf": "dd" * 32, "output_root": "ee" * 32}
    txhash = ops.respond({"challenge_id": 9}, tile)
    assert txhash == "ABCD"
    tx = [c for c in commands if "respond-graph-da-challenge" in c][0]
    assert tx[tx.index("--challenge-id") + 1] == "9"
    assert tx[tx.index("--chunk") + 1] == "aa" * 64
    assert tx[tx.index("--chunk-index") + 1] == "2"
    assert tx[tx.index("--chunk-count") + 1] == "4"
    assert tx.count("--chunk-proof") == 2
