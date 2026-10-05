"""B2-05: manifest, verification, quarantine, resumable install, chain check."""

from __future__ import annotations

import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from prisma_worker import model

REPO = pathlib.Path(__file__).resolve().parents[2]
FROZEN_GRAPH = REPO / "testdata" / "f5c_qwen3_block_v2.json"
FROZEN_GRAPH_ID_V2 = "8fb86087697e277602bb4b2088ba48eb4172c21d9c4f83914582f2cbf00d4def"


def make_profile(tmp_path: pathlib.Path, *, extra: dict[str, bytes] | None = None) -> pathlib.Path:
    """A small profile directory whose graph.json carries the frozen identity."""
    directory = tmp_path / "profile-src"
    directory.mkdir()
    (directory / "graph.json").write_bytes(FROZEN_GRAPH.read_bytes())
    (directory / "tensor_shapes.json").write_text('{"shapes": [1, 2]}', encoding="utf-8")
    for name, blob in (extra or {}).items():
        (directory / name).write_bytes(blob)
    model.build_manifest(directory, model_id="qwen3-0.6b-layer0-v2", spec_version="v1",
                         digests={"weights": "a" * 64, "tokenizer": "c" * 64})
    return directory


def test_build_and_verify_a_frozen_profile(tmp_path):
    directory = make_profile(tmp_path)
    manifest = model.verify(directory)
    assert manifest.graph_id_v2 == FROZEN_GRAPH_ID_V2
    assert manifest.model_id == "qwen3-0.6b-layer0-v2"
    assert {f["path"] for f in manifest.files} == {"graph.json", "tensor_shapes.json"}


def test_tampered_graph_is_refused_and_quarantined(tmp_path):
    directory = make_profile(tmp_path)
    graph = directory / "graph.json"
    document = json.loads(graph.read_text(encoding="utf-8"))
    document["nodes"][0]["op"] = "TAMPERED"
    graph.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(model.ProfileVerificationError, match="checksum mismatch"):
        model.verify(directory)
    assert list(directory.glob("graph.json.quarantined-*")), "the bad artifact must be moved aside"
    assert list(directory.glob("graph.json.quarantined-*.reason"))


def test_any_file_mismatch_is_refused(tmp_path):
    directory = make_profile(tmp_path)
    (directory / "tensor_shapes.json").write_text('{"shapes": [9, 9]}', encoding="utf-8")
    with pytest.raises(model.ProfileVerificationError, match="tensor_shapes.json"):
        model.verify(directory)


def test_non_frozen_identity_is_refused(tmp_path):
    directory = make_profile(tmp_path)
    manifest = model.load_manifest(directory)
    document = manifest.to_dict()
    document["graph_id_v2"] = "0" * 64
    (directory / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(model.ProfileVerificationError, match="GraphIDV2 mismatch"):
        model.verify(directory)


def test_missing_file_is_refused(tmp_path):
    directory = make_profile(tmp_path)
    (directory / "tensor_shapes.json").unlink()
    with pytest.raises(model.ProfileVerificationError, match="missing"):
        model.verify(directory)


def test_install_from_directory_then_cache_hit(tmp_path):
    source = make_profile(tmp_path)
    models = tmp_path / "models"
    report = model.install(str(source), models, log=lambda _m: None)
    assert sorted(report.files) == ["graph.json", "tensor_shapes.json"]
    profile = model.profile_dir(models, "qwen3-0.6b-layer0-v2", "v1")
    assert model.verify(profile).graph_id_v2 == FROZEN_GRAPH_ID_V2

    again = model.install(str(source), models, log=lambda _m: None)
    assert sorted(again.cached) == ["graph.json", "tensor_shapes.json"], "second install hits the cache"


def test_install_refuses_a_tampered_source_file(tmp_path):
    source = make_profile(tmp_path)
    (source / "tensor_shapes.json").write_text('{"shapes": "evil"}', encoding="utf-8")
    with pytest.raises(model.ProfileVerificationError, match="checksum mismatch"):
        model.install(str(source), tmp_path / "models", log=lambda _m: None)
    cache = tmp_path / "models" / ".cache"
    assert list(cache.glob("tensor_shapes.json.part.quarantined-*"))


class _RangeServer(BaseHTTPRequestHandler):
    """Minimal static server with Range support (SimpleHTTPRequestHandler has none)."""

    directory: pathlib.Path

    def do_GET(self):  # noqa: N802
        name = self.path.lstrip("/").split("?")[0]
        path = self.directory / name
        if not path.exists():
            self.send_error(404)
            return
        blob = path.read_bytes()
        start = 0
        if self.headers.get("Range", "").startswith("bytes="):
            start = int(self.headers["Range"][6:].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(blob) - 1}/{len(blob)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(blob) - start))
        self.end_headers()
        self.wfile.write(blob[start:])

    def log_message(self, *args):  # silence
        return


@pytest.fixture()
def range_server(tmp_path):
    source = make_profile(tmp_path)
    handler = type("Handler", (_RangeServer,), {"directory": source})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", source
    server.shutdown()


def test_resumable_install_over_http(tmp_path, range_server):
    url, source = range_server
    models = tmp_path / "models"
    model.install(url, models, log=lambda _m: None)

    # Simulate an interrupted transfer: drop the published copy and the cached
    # blob, and leave a partial download behind.
    graph = source / "graph.json"
    blob = models / ".cache" / model.sha256_file(graph)
    partial = models / ".cache" / "graph.json.part"
    blob.unlink()
    partial.write_bytes(graph.read_bytes()[: 4096])

    report = model.install(url, models, log=lambda _m: None)
    assert "graph.json" in report.resumed, report.resumed
    assert (models / ".cache" / model.sha256_file(graph)).exists()
    assert model.verify(model.profile_dir(models, "qwen3-0.6b-layer0-v2", "v1"))


def test_chain_check_matching_and_mismatching(tmp_path):
    manifest = model.load_manifest(make_profile(tmp_path))
    matching = {"image_digest": "b" * 64, "tokenizer_digest": "c" * 64,
                "weights_digest": "a" * 64, "program_digest": ""}
    ok, reason = model.chain_check(manifest, matching)
    assert ok and "matches" in reason
    ok, reason = model.chain_check(manifest, {**matching, "weights_digest": "d" * 64})
    assert not ok and "weights digest" in reason
    ok, reason = model.chain_check(manifest, {})
    assert not ok and "no registered model" in reason
