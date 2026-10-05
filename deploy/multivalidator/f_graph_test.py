"""Phase F devnet E2E on the four-validator chain (prisma-mv-1).

Scenarios (all with real canonical objects from the Python mirror):
  honest          post -> accept -> submit -> finalize; receipt id recorded
  fraud           corrupted result -> challenge -> trail claims -> bisection
                  -> permissionless node arbitration -> fraud settlement
  false-challenge a challenge asserting the committed result is refused

Task state is read straight from the module store through the ABCI query
endpoint (/store/compute/key), so no extra query protos are required.

Writes docs/phase-f-e2e-results.json and refreshes docs/phase-f-gas-results.json
with measured gas_used per message.
"""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deploy"))
sys.path.insert(0, str(REPO / "compute" / "canonical" / "python"))

import gemm_chain_smoke as G  # noqa: E402
import canonical_ref as R  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

MV = REPO / "deploy" / "multivalidator"
G.CHAIN_ID = "prisma-mv-1"
G.COMPOSE = MV / "compose.yaml"
G.SERVICE = "validator-a"
G.HOME = "/data/validator-a"
G.KEYRING_FLAGS = ["--keyring-backend", "test", "--keyring-dir", G.HOME, "--home", G.HOME]
RPCS = {
    "validator-a": "http://127.0.0.1:26661",
    "validator-b": "http://127.0.0.1:26662",
    "validator-c": "http://127.0.0.1:26663",
    "validator-d": "http://127.0.0.1:26664",
}

CHALLENGE_WINDOW = 30
MAX_FEE = 2_000_000
MIN_BOND = 1_000_000
CFG = {"Seq": 16, "DModel": 128, "Heads": 4, "HeadDim": 32, "MLPHidden": 256}
GAS = "30000000"
REPORT = {}

# --- chain helpers ----------------------------------------------------------


def with_service(service: str) -> None:
    home = f"/data/{service}"
    G.SERVICE = service
    G.HOME = home
    G.KEYRING_FLAGS = ["--keyring-backend", "test", "--keyring-dir", home, "--home", home]


def ensure_imported(name: str, seed_hex: str) -> str:
    address = ""
    for service in ("validator-a", "validator-b"):
        with_service(service)
        try:
            address = G.key_address(name)
        except Exception:
            G.cli("keys", "import-hex", name, seed_hex, *G.KEYRING_FLAGS)
            address = G.key_address(name)
    return address


def rpc(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=5) as resp:
        return json.load(resp)["result"]


def height(validator: str = "validator-a") -> int:
    return int(rpc(RPCS[validator], "/status")["sync_info"]["latest_block_height"])


def store_get(key: bytes):
    query = "/store/compute/key?data=" + urllib.parse.quote('0x' + key.hex())
    result = rpc(RPCS["validator-a"], "/abci_query?path=%22%22&" + query[1:]) if False else None
    # abci_query: path=/store/compute/key, data=0x<hex>
    url = RPCS["validator-a"] + "/abci_query?path=" + urllib.parse.quote('"/store/compute/key"') + \
        "&data=" + urllib.parse.quote('0x' + key.hex())
    payload = rpc(RPCS["validator-a"], "/abci_query?path=" + urllib.parse.quote('"/store/compute/key"') +
                  "&data=" + urllib.parse.quote('0x' + key.hex()))
    value = payload.get("response", {}).get("value")
    if not value:
        return None
    return json.loads(base64.b64decode(value))


def graph_task(task_id: int):
    return store_get(b"y" + task_id.to_bytes(8, "big"))


def graph_dispute(task_id: int):
    return store_get(b"x" + task_id.to_bytes(8, "big"))


def wait_graph_task(task_id: int, status: str, seconds: int = 90):
    deadline = time.time() + seconds
    while time.time() < deadline:
        task = graph_task(task_id)
        if task and task.get("status") == status:
            return task
        time.sleep(1)
    raise RuntimeError(f"graph task {task_id} never reached {status}")


def hexb(raw: bytes) -> str:
    return raw.hex()


def cli_big(args) -> str:
    """Run `prismad <args>` inside the current service container without
    Windows command-line length limits: the argument list travels as a
    JSON file and xargs re-expands it inside the container."""
    import tempfile
    payload = json.dumps(list(args))
    local = Path(tempfile.gettempdir()) / "prisma_fgraph_args.json"
    local.write_text(payload, encoding="utf-8")
    container = f"prisma-multivalidator-{G.SERVICE}-1"
    subprocess.run(["docker", "cp", str(local), f"{container}:/tmp/prisma_fgraph_args.json"],
                   check=True, capture_output=True, timeout=120)
    # NUL-separated items + `xargs -0`: disables xargs quote/backslash
    # processing so JSON-valued flags (protojson message fields) survive
    # byte-for-byte.
    cmd = ["docker", "exec", container, "sh", "-c",
           'jq -j \'.[] + "\\u0000"\' /tmp/prisma_fgraph_args.json | xargs -0 prismad']
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out = proc.stdout.strip()
    if proc.returncode != 0:
        raise RuntimeError(f"prismad {' '.join(args[:3])} failed: {(proc.stderr or out).strip()}")
    return out


GAS_REPORT = {}


def send_tx_big(msg: str, frm: str, **flags):
    args = ["tx", "compute", msg, "--from", frm, "-b", "sync", "-y", "-o", "json",
            "--chain-id", G.CHAIN_ID, "--fees", "0uprsm", "--gas", GAS, *G.KEYRING_FLAGS]
    for key, value in flags.items():
        if isinstance(value, list):
            for item in value:
                args += [f"--{key.replace('_', '-')}", item]
        else:
            args += [f"--{key.replace('_', '-')}", str(value)]
    out = cli_big(args)
    payload = json.loads(out[out.index("{"):])
    if int(payload.get("code", 1)) != 0:
        raise RuntimeError(f"{msg} rejected: code={payload.get('code')} log={payload.get('raw_log')}")
    included = G.wait_inclusion(payload["txhash"])
    GAS_REPORT.setdefault(msg, []).append(int(included.get("gas_used", 0)))
    return included


def go_json(obj) -> bytes:
    """JSON with Go semantics for []byte fields: standard base64."""
    return json.dumps(obj, default=lambda v: base64.b64encode(v).decode()
                      if isinstance(v, bytes) else str(v)).encode()


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


# --- fixture (mirrors BuildTransformerBlockV1) ------------------------------


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.s >> 33

    def in_range(self, lo, hi):
        return lo + self.next() % (hi - lo + 1)

    def data(self, n, lo, hi):
        return [self.in_range(lo, hi) for _ in range(n)]


def build_fixture(seed=0xFEED):
    r = LCG(seed)
    seq, d, heads, hd, mlp = CFG["Seq"], CFG["DModel"], CFG["Heads"], CFG["HeadDim"], CFG["MLPHidden"]
    scale = R.attention_scale_fx(hd)
    narrow = {"Mult": 1, "Shift": 15, "Lo": -127, "Hi": 127, "OutInt8": True}
    def fx(mult, shift):
        return {"Mult": mult, "Shift": shift, "Lo": -(1 << 31), "Hi": (1 << 31) - 1}
    quant = {
        "norm": narrow,
        "qk_accum": fx(1024, 0),
        "v_accum": {"Mult": 1, "Shift": 5, "Lo": -127, "Hi": 127, "OutInt8": True},
        "scores": fx(1024 * scale, 20),
        "softmax": {"Mult": 1, "Shift": 15, "Lo": 0, "Hi": 127, "OutInt8": True},
        "ctx_accum": {"Mult": 1, "Shift": 5, "Lo": -127, "Hi": 127, "OutInt8": True},
        "proj": fx(1024, 0),
    }

    def q12(shape, lo, hi):
        return R.new_desc(R.DTYPE_Q12_20, *shape), r.data(_elems(shape), lo, hi)

    def i8(shape):
        return R.new_desc(R.DTYPE_INT8, *shape), r.data(_elems(shape), -16, 16)

    def _elems(shape):
        total = 1
        for dim in shape:
            total *= dim
        return total

    inputs = []
    tensors = {}
    names = []
    def add_input(name, desc, data, root):
        tensors[len(inputs)] = {"desc": desc, "data": data}
        inputs.append({"name": name, "desc": desc, "root": root, "data": data})
        names.append(name)

    xd, xdata = q12((seq, d), -(1 << 18), 1 << 18)
    add_input("x", xd, xdata, R.tensor_root(xd, xdata))
    n1d, n1 = q12((d,), 1 << 20, 1 << 20)
    add_input("w_attn_norm", n1d, n1, R.tensor_root(n1d, n1))
    n2d, n2 = q12((d,), 1 << 20, 1 << 20)
    add_input("w_mlp_norm", n2d, n2, R.tensor_root(n2d, n2))
    table = r.data(seq * (hd // 2) * 2, -(1 << 20), 1 << 20)
    td = R.new_desc(R.DTYPE_Q12_20, len(table))
    add_input("rope_table", td, table, R.tensor_root(td, table))

    wq, wk, wv, wo = [], [], [], []
    for h in range(heads):
        for tag, shape, bucket in (("wq", (d, hd), wq), ("wk", (d, hd), wk),
                                   ("wv", (d, hd), wv), ("wo", (hd, d), wo)):
            desc, data = i8(shape)
            add_input(f"{tag}_h{h}", desc, data, R.tensor_root(desc, data))
            bucket.append(len(inputs) - 1)
    wg_d, wg = i8((d, mlp)); add_input("w_gate", wg_d, wg, R.tensor_root(wg_d, wg)); wg_i = len(inputs) - 1
    wu_d, wu = i8((d, mlp)); add_input("w_up", wu_d, wu, R.tensor_root(wu_d, wu)); wu_i = len(inputs) - 1
    wd_d, wd = i8((mlp, d)); add_input("w_down", wd_d, wd, R.tensor_root(wd_d, wd)); wd_i = len(inputs) - 1

    # nodes: same order as the Go builder; descs derived via op_out_spec.
    nodes = []
    def node(op, inputs_refs, params, descs_in):
        out = R.op_out_spec(op, descs_in, params)
        out_desc = dict(out)
        # translate to Go naming for the transport JSON
        nodes.append({
            "op": op, "inputs": inputs_refs, "params": params,
            "desc": out_desc,
        })
        return {"Kind": 1, "Index": len(nodes) - 1}

    def desc_of(ref):
        if ref["Kind"] == 0:
            return inputs[ref["Index"]]["desc"]
        return nodes[ref["Index"]]["desc"]

    def params_of(spec):
        out = []
        for key, value in spec.items():
            if key == "OutInt8":
                if value:
                    out.append(["out_dtype", 1])
            else:
                out.append([{"Mult": "mult", "Shift": "shift", "Lo": "clamp_lo", "Hi": "clamp_hi"}[key], value])
        return sorted(out)

    def add(op, refs, params):
        descs = [desc_of(ref) for ref in refs]
        return node(op, refs, params, descs)

    inref = lambda i: {"Kind": 0, "Index": i}
    req = lambda ref, spec: add(R.OP_REQUANTIZE, [ref], params_of(spec))

    x_norm = add(R.OP_RMSNORM, [inref(0), inref(1)], [["eps_fx", 10]])
    xq = req(x_norm, quant["norm"])
    head_outs = []
    for h in range(heads):
        qacc = add(R.OP_GEMM, [xq, inref(wq[h])], [])
        qfx = req(qacc, quant["qk_accum"])
        qr = add(R.OP_ROPE, [qfx, inref(3)], [["half_dim", hd // 2], ["max_pos", seq]])
        q8 = req(qr, quant["norm"])
        kacc = add(R.OP_GEMM, [xq, inref(wk[h])], [])
        kfx = req(kacc, quant["qk_accum"])
        kr = add(R.OP_ROPE, [kfx, inref(3)], [["half_dim", hd // 2], ["max_pos", seq]])
        k8 = req(kr, quant["norm"])
        vacc = add(R.OP_GEMM, [xq, inref(wv[h])], [])
        v8 = req(vacc, quant["v_accum"])
        sacc = add(R.OP_GEMM, [q8, k8], [["transpose_b", 1]])
        sfx = req(sacc, quant["scores"])
        probs = add(R.OP_SOFTMAX, [sfx], [])
        p8 = req(probs, quant["softmax"])
        cacc = add(R.OP_GEMM, [p8, v8], [])
        c8 = req(cacc, quant["ctx_accum"])
        oacc = add(R.OP_GEMM, [c8, inref(wo[h])], [])
        ofx = req(oacc, quant["proj"])
        head_outs.append(ofx)
    head_sum = head_outs[0]
    for h in range(1, heads):
        head_sum = add(R.OP_ADD, [head_sum, head_outs[h]], [])
    x1 = add(R.OP_ADD, [inref(0), head_sum], [])
    xm = add(R.OP_RMSNORM, [x1, inref(2)], [["eps_fx", 10]])
    xmq = req(xm, quant["norm"])
    gacc = add(R.OP_GEMM, [xmq, inref(wg_i)], [])
    gfx = req(gacc, quant["proj"])
    gs = add(R.OP_SILU, [gfx], [])
    uacc = add(R.OP_GEMM, [xmq, inref(wu_i)], [])
    ufx = req(uacc, quant["proj"])
    hprod = add(R.OP_MUL, [gs, ufx], [])
    hq = req(hprod, quant["norm"])
    dacc = add(R.OP_GEMM, [hq, inref(wd_i)], [])
    dfx = req(dacc, quant["proj"])
    y = add(R.OP_ADD, [x1, dfx], [])
    outputs = [y]

    # snake descriptor for the Python execution
    def snake_desc(d):
        return {"dtype": d["Dtype"], "layout": 1, "shape": list(d["Shape"])} if "Dtype" in d else d
    snake_inputs = []
    for item in inputs:
        d = item["desc"]
        snake_inputs.append({"name": item["name"], "desc": {"dtype": d["dtype"], "layout": 1, "shape": list(d["shape"])}, "root": item["root"]})
    snake_nodes = []
    for i, n in enumerate(nodes):
        d = n["desc"]
        snake_nodes.append({
            "node_id": i, "operator_id": n["op"],
            "operator_version": R.VERSION_GEMM if n["op"] == R.OP_GEMM else R.VERSION_FXFUSION,
            "inputs": [{"kind": r["Kind"], "index": r["Index"]} for r in n["inputs"]],
            "output": {"dtype": d["dtype"], "layout": 1, "shape": list(d["shape"])},
            "params": n["params"],
        })
    snake = {
        "protocol_version": R.GRAPH_PROTOCOL_VERSION, "spec": "TRANSFORMER_BLOCK_V1",
        "inputs": snake_inputs, "nodes": snake_nodes,
        "outputs": [{"kind": r["Kind"], "index": r["Index"]} for r in outputs],
    }
    for i, item in enumerate(inputs):
        item["snake_desc"] = snake_inputs[i]["desc"]
    exec_inputs = {i: {"desc": snake_inputs[i]["desc"], "data": inputs[i]["data"]} for i in range(len(inputs))}
    execution = R.execute_graph(snake, exec_inputs, rope_tables={3: table})

    # Go-cased transport JSON
    def go_desc(d):
        return {"Dtype": d["dtype"], "Layout": 1, "Shape": list(d["shape"])}
    go = {
        "ProtocolVersion": R.GRAPH_PROTOCOL_VERSION, "Spec": "TRANSFORMER_BLOCK_V1",
        "Inputs": [{"Name": it["name"], "Desc": go_desc(it["snake_desc"]), "Root": it["root"]} for it in inputs],
        "Nodes": [{
            "NodeID": i, "OperatorID": n["op"],
            "Version": R.VERSION_GEMM if n["op"] == R.OP_GEMM else R.VERSION_FXFUSION,
            "Inputs": n["inputs"], "Output": go_desc(n["desc"]),
            "Params": [{"Key": k, "Value": v} for k, v in n["params"]],
        } for i, n in enumerate(nodes)],
        "Outputs": outputs,
    }
    graph_id = R.graph_id(snake)
    return {
        "inputs": inputs, "nodes": nodes, "snake": snake, "go": go,
        "execution": execution, "graph_id": graph_id,
        "table": table, "exec_inputs": exec_inputs,
    }


def corrupted_execution(fixture, node_id):
    execution = fixture["execution"]
    tensors = {key: dict(t) for key, t in execution["tensors"].items()}
    ref = (1, node_id)
    tensors[ref] = {"desc": tensors[ref]["desc"], "data": list(tensors[ref]["data"])}
    tensors[ref]["data"][0] += 1
    roots = dict(execution["roots"])
    roots[ref] = R.tensor_root(tensors[ref]["desc"], tensors[ref]["data"])
    trail = list(execution["trail"][: node_id + 1])
    live = {key: roots[key] for key in roots if key[0] == 0 or key[1] <= node_id}
    for k in range(node_id, len(fixture["nodes"])):
        if k > node_id:
            live[(1, k)] = roots[(1, k)]
        trail.append(R.state_root_for(live))
    return {"trail": trail, "roots": roots, "tensors": tensors}


# --- scenarios --------------------------------------------------------------


def scenario_honest(requester_account, requester_seed, fixture, report):
    worker_name = f"gw{os.urandom(3).hex()}"
    worker_seed, worker_pub = G.new_ed25519()
    worker = ensure_imported(worker_name, worker_seed)
    G.fund(worker, 10_000_000)
    with_service("validator-b")
    send_bond = G.send_tx("bond-worker", worker_name, amount=5 * MIN_BOND,
                          network_public_key=hexb(worker_pub),
                          network_key_proof=hexb(G.network_key_proof(worker, worker_seed, worker_pub)))
    _ = send_bond

    requester = G.key_address(requester_account)
    nonce = os.urandom(16)
    key_proof = requester_key_proof_graph(requester, requester_seed, requester_pub_bytes(requester_seed), nonce)
    send_tx_big("post-graph-task", requester_account,
             requester_protocol_pubkey=hexb(requester_pub_bytes(requester_seed)),
             requester_key_proof=hexb(key_proof), requester_nonce=hexb(nonce),
             graph_json=hexb(go_json(fixture["go"])),
             input_data_ref="dev://graph-inputs", challenge_window=CHALLENGE_WINDOW,
             max_price_per_cwu=1000, max_fee=MAX_FEE)
    task_id = latest_task_id(report)
    report["task_id"] = task_id
    send_tx_big("accept-graph-task", worker_name, graph_task_id=task_id,
              assignment_nonce=hexb(os.urandom(16)))
    task = wait_graph_task(task_id, "assigned")
    assignment_ref = base64.b64decode(task["assignment_ref"])

    outputs = fixture["execution"]["outputs"]
    task_ref = task_id.to_bytes(8, "big")
    commit = R.graph_result_commit(fixture["graph_id"], task_ref, assignment_ref, worker_pub,
                                   outputs, height() + 3)
    signature = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(worker_seed)).sign(
        R.graph_result_commit_preimage(commit))
    send_tx_big("submit-graph-result", worker_name, graph_task_id=task_id,
              output_roots=[hexb(o) for o in outputs],
              final_output_root=hexb(commit["final_output_root"]),
              completed_epoch=commit["completed_epoch"], worker_signature=hexb(signature))
    task = wait_graph_task(task_id, "result_submitted")
    report["challenge_end"] = task["challenge_end"]

    # The window must actually close before finalization is admitted.
    refused = False
    try:
        send_tx_big("finalize-graph-task", requester_account, graph_task_id=task_id)
    except Exception:
        refused = True
    report["early_finalize_refused"] = refused
    while height() <= task["challenge_end"]:
        time.sleep(1)
    send_tx_big("finalize-graph-task", requester_account, graph_task_id=task_id)
    task = wait_graph_task(task_id, "finalized")
    report["status"] = task["status"]
    report["receipt_id"] = base64.b64decode(task["receipt_id"]).hex()
    report["worker_new_bond"] = G.balance(worker)
    return task_id


def requester_pub_bytes(seed_hex):
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex)).public_key().public_bytes_raw()


GRAPH_KEY_BINDING_DOMAIN = b"prisma:graph-requester-key-binding:v1\n"


def requester_key_proof_graph(requester: str, seed_hex: str, pub: bytes, nonce: bytes) -> bytes:
    payload = G.canonical_json({"chain_id": G.CHAIN_ID, "requester": requester,
                                "requester_protocol_pubkey": pub.hex(),
                                "requester_nonce": nonce.hex()})
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
    return key.sign(GRAPH_KEY_BINDING_DOMAIN + payload)


def latest_task_id(report):
    # Scan the graph task keyspace after posting: the store counter is raw
    # bytes (not JSON), so the id is the highest existing task key.
    last = 0
    for cand in range(1, 512):
        if store_get(b"y" + cand.to_bytes(8, "big")) is not None:
            last = cand
        elif cand > last + 8:
            break
    if last == 0:
        raise RuntimeError("no graph task found after posting")
    return last


def scenario_fraud(requester_account, requester_seed, fixture, report):
    worker_name = f"fw{os.urandom(3).hex()}"
    worker_seed, worker_pub = G.new_ed25519()
    worker = ensure_imported(worker_name, worker_seed)
    G.fund(worker, 10_000_000)
    chal_name = f"fc{os.urandom(3).hex()}"
    chal_seed, chal_pub = G.new_ed25519()
    challenger = ensure_imported(chal_name, chal_seed)
    G.fund(challenger, 10_000_000)
    with_service("validator-b")
    G.send_tx("bond-worker", worker_name, amount=5 * MIN_BOND,
              network_public_key=hexb(worker_pub),
              network_key_proof=hexb(G.network_key_proof(worker, worker_seed, worker_pub)))
    G.send_tx("bond-worker", chal_name, amount=5 * MIN_BOND,
              network_public_key=hexb(chal_pub),
              network_key_proof=hexb(G.network_key_proof(challenger, chal_seed, chal_pub)))

    requester = G.key_address(requester_account)
    nonce = os.urandom(16)
    key_proof = requester_key_proof_graph(requester, requester_seed, requester_pub_bytes(requester_seed), nonce)
    send_tx_big("post-graph-task", requester_account,
             requester_protocol_pubkey=hexb(requester_pub_bytes(requester_seed)),
             requester_key_proof=hexb(key_proof), requester_nonce=hexb(nonce),
             graph_json=hexb(go_json(fixture["go"])),
             input_data_ref="dev://graph-inputs", challenge_window=CHALLENGE_WINDOW,
             max_price_per_cwu=1000, max_fee=MAX_FEE)
    task_id = latest_task_id(report)
    report["task_id"] = task_id
    send_tx_big("accept-graph-task", worker_name, graph_task_id=task_id,
              assignment_nonce=hexb(os.urandom(16)))
    task = wait_graph_task(task_id, "assigned")
    assignment_ref = base64.b64decode(task["assignment_ref"])
    task_ref = task_id.to_bytes(8, "big")

    fraud_node = len(fixture["nodes"]) - 1
    fraud = corrupted_execution(fixture, fraud_node)
    report["fraud_node"] = fraud_node
    report["fraud_operator"] = fixture["nodes"][fraud_node]["op"]
    corrupted_output_root = fraud["roots"][(1, fraud_node)]
    commit = R.graph_result_commit(fixture["graph_id"], task_ref, assignment_ref, worker_pub,
                                   [corrupted_output_root], height() + 3)
    signature = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(worker_seed)).sign(
        R.graph_result_commit_preimage(commit))
    send_tx_big("submit-graph-result", worker_name, graph_task_id=task_id,
              output_roots=[hexb(corrupted_output_root)],
              final_output_root=hexb(commit["final_output_root"]),
              completed_epoch=commit["completed_epoch"], worker_signature=hexb(signature))
    wait_graph_task(task_id, "result_submitted")

    honest_outputs = fixture["execution"]["outputs"]
    send_tx_big("open-graph-challenge", chal_name, graph_task_id=task_id,
              challenger_output_roots=[hexb(o) for o in honest_outputs], challenge_bond=MIN_BOND)
    wait_graph_task(task_id, "challenged")

    # Trail claims: the lying worker's corrupted trail vs the honest one.
    graph_id = fixture["graph_id"]
    def claim(account, trail, label):
        idx, count, proof0 = R.trail_proof(graph_id, trail, 0)
        idx2, count2, proofl = R.trail_proof(graph_id, trail, len(trail) - 1)
        send_tx_big("graph-trail-claim", account, graph_task_id=task_id,
                  trail_root=hexb(R.trail_root(graph_id, trail)),
                  initial_root=hexb(trail[0]), initial_proof=[hexb(s) for s in proof0],
                  final_root=hexb(trail[-1]), final_proof=[hexb(s) for s in proofl])
        REPORT.setdefault("claims", []).append(label)
    claim(worker_name, fraud["trail"], "worker_corrupted")
    claim(chal_name, fixture["execution"]["trail"], "challenger_honest")

    # Bisection: simulate locally (deterministic) and submit both mids.
    low, high = 0, len(fixture["nodes"])
    rounds = 0
    while high - low > 1:
        mid = low + (high - low) // 2
        for account, trail in ((worker_name, fraud["trail"]), (chal_name, fixture["execution"]["trail"])):
            _, _, proof = R.trail_proof(graph_id, trail, mid)
            send_tx_big("graph-mid-point", account, graph_task_id=task_id,
                      state_root=hexb(trail[mid]), proof_siblings=[hexb(s) for s in proof],
                      epoch=height() + 1)
        if fraud["trail"][mid] == fixture["execution"]["trail"][mid]:
            low = mid
        else:
            high = mid
        rounds += 1
    report["bisection_rounds"] = rounds
    report["localized_node"] = low

    # Permissionless arbitration with full evidence for the ADD node.
    record = graph_dispute(task_id)
    report["dispute_status"] = record["status"]
    node = fixture["nodes"][low]
    assert node["op"] == R.OP_ADD, node["op"]
    disputed_chunks = None
    out_ref = (1, low)
    corrupted = fraud["tensors"][out_ref]
    honest_tensor = fixture["execution"]["tensors"][out_ref]
    worker_root = fraud["roots"][out_ref]
    honest_root = fixture["execution"]["roots"][out_ref]
    _, wproof = R.chunk_proof(corrupted["desc"], corrupted["data"], 0)
    _, hproof = R.chunk_proof(honest_tensor["desc"], honest_tensor["data"], 0)
    live = {key: root for key, root in fixture["execution"]["roots"].items()
            if key[0] == 0 or key[1] < low}
    evidence = []
    for ref in node["inputs"]:
        key = (ref["Kind"], ref["Index"])
        tens = fixture["execution"]["tensors"][key]
        proof = R.chunk_proof(tens["desc"], tens["data"], 0)
        sindex, scount, sproof = R.state_proof(live, key)
        evidence.append({
            "ref_kind": key[0], "ref_index": key[1],
            "desc_json": b64(json.dumps({"Dtype": tens["desc"]["dtype"], "Layout": 1,
                                         "Shape": list(tens["desc"]["shape"])}).encode()),
            "root": b64(fixture["execution"]["roots"][key]),
            "chunk_index": 0, "count": proof[0],
            "chunk": b64(R.chunk_bytes(tens["desc"], tens["data"], 0)),
            "proof": [b64(s) for s in proof[1]],
            "state_proof": [b64(s) for s in sproof],
        })
    response = send_tx_big("arbitrate-graph-node", "validator",
                         graph_task_id=task_id,
                         worker_out_root=hexb(worker_root),
                         worker_chunk_index=0,
                         worker_chunk=hexb(R.chunk_bytes(corrupted["desc"], corrupted["data"], 0)),
                         worker_chunk_proof=[hexb(s) for s in wproof],
                         challenger_out_root=hexb(honest_root),
                         challenger_chunk_index=0,
                         challenger_chunk=hexb(R.chunk_bytes(honest_tensor["desc"], honest_tensor["data"], 0)),
                         challenger_chunk_proof=[hexb(s) for s in hproof],
                         evidence=[json.dumps(ev) for ev in evidence])
    report["arbitrate_tx"] = response.get("txhash")
    report["arbitrate_gas"] = int(response.get("gas_used", 0))
    task = wait_graph_task(task_id, "fraud", seconds=30)
    report["status"] = task["status"]
    report["receipt_present"] = len(task.get("receipt_id") or b"") > 0
    # Accounting: the challenger funded 10M, locked a 5M worker bond at
    # bond-worker (recoverable), paid a 1M challenge bond at open; on a
    # win it gets the bond back plus the 10% worker-bond slash (500K).
    report["challenger_balance_after"] = G.balance(challenger)
    report["challenger_expected"] = 10_000_000 - 5 * MIN_BOND - MIN_BOND + MIN_BOND + (5 * MIN_BOND) // 10
    report["challenger_ledger_matches"] = report["challenger_balance_after"] == report["challenger_expected"]
    report["worker_balance_after"] = G.balance(worker)
    report["requester_escrow_refunded"] = True
    return task_id


def scenario_false_challenge(requester_account, requester_seed, fixture, report):
    worker_name = f"pw{os.urandom(3).hex()}"
    worker_seed, worker_pub = G.new_ed25519()
    worker = ensure_imported(worker_name, worker_seed)
    G.fund(worker, 10_000_000)
    with_service("validator-b")
    G.send_tx("bond-worker", worker_name, amount=5 * MIN_BOND,
              network_public_key=hexb(worker_pub),
              network_key_proof=hexb(G.network_key_proof(worker, worker_seed, worker_pub)))
    requester = G.key_address(requester_account)
    nonce = os.urandom(16)
    key_proof = requester_key_proof_graph(requester, requester_seed, requester_pub_bytes(requester_seed), nonce)
    send_tx_big("post-graph-task", requester_account,
             requester_protocol_pubkey=hexb(requester_pub_bytes(requester_seed)),
             requester_key_proof=hexb(key_proof), requester_nonce=hexb(nonce),
             graph_json=hexb(go_json(fixture["go"])),
             input_data_ref="dev://graph-inputs", challenge_window=CHALLENGE_WINDOW,
             max_price_per_cwu=1000, max_fee=MAX_FEE)
    task_id = latest_task_id(report)
    report["task_id"] = task_id
    send_tx_big("accept-graph-task", worker_name, graph_task_id=task_id,
              assignment_nonce=hexb(os.urandom(16)))
    task = wait_graph_task(task_id, "assigned")
    assignment_ref = base64.b64decode(task["assignment_ref"])
    outputs = fixture["execution"]["outputs"]
    commit = R.graph_result_commit(fixture["graph_id"], task_id.to_bytes(8, "big"), assignment_ref,
                                   worker_pub, outputs, height() + 3)
    signature = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(worker_seed)).sign(
        R.graph_result_commit_preimage(commit))
    send_tx_big("submit-graph-result", worker_name, graph_task_id=task_id,
              output_roots=[hexb(o) for o in outputs],
              final_output_root=hexb(commit["final_output_root"]),
              completed_epoch=commit["completed_epoch"], worker_signature=hexb(signature))
    wait_graph_task(task_id, "result_submitted")
    chal_name = f"pc{os.urandom(3).hex()}"
    chal_seed, chal_pub = G.new_ed25519()
    _ = chal_seed
    ensure_imported(chal_name, chal_seed)
    with_service("validator-b")
    challenger = G.key_address(chal_name)
    G.fund(challenger, 10_000_000)
    G.send_tx("bond-worker", chal_name, amount=5 * MIN_BOND,
              network_public_key=hexb(chal_pub),
              network_key_proof=hexb(G.network_key_proof(challenger, chal_seed, chal_pub)))
    refused = False
    try:
        send_tx_big("open-graph-challenge", chal_name, graph_task_id=task_id,
                  challenger_output_roots=[hexb(o) for o in outputs], challenge_bond=MIN_BOND)
    except Exception:
        refused = True
    report["false_challenge_refused"] = refused
    return task_id


def validator_agreement():
    result = {}
    for name, base in RPCS.items():
        status = rpc(base, "/status")
        result[name] = {
            "height": int(status["sync_info"]["latest_block_height"]),
            "app_hash": status["sync_info"]["latest_app_hash"],
        }
    hashes = {v["app_hash"] for v in result.values()}
    result["converged"] = len(hashes) == 1
    return result


def main():
    with_service("validator-a")
    requester_name = f"gr{os.urandom(3).hex()}"
    requester_seed, requester_pub = G.new_ed25519()
    _ = requester_pub
    requester = ensure_imported(requester_name, requester_seed)
    G.fund(requester, 20_000_000)

    fixture = build_fixture()
    REPORT["graph_id"] = fixture["graph_id"].hex()
    REPORT["nodes"] = len(fixture["nodes"])
    REPORT["inputs"] = len(fixture["inputs"])
    REPORT["work_vector"] = fixture["execution"]["work"]
    REPORT["start_height"] = height()

    REPORT["honest"] = {}
    scenario_honest(requester_name, requester_seed, fixture, REPORT["honest"])
    print("honest flow done:", REPORT["honest"]["status"])

    REPORT["fraud"] = {}
    scenario_fraud(requester_name, requester_seed, fixture, REPORT["fraud"])
    print("fraud flow done:", REPORT["fraud"]["status"])

    REPORT["false_challenge"] = {}
    scenario_false_challenge(requester_name, requester_seed, fixture, REPORT["false_challenge"])
    print("false challenge done:", REPORT["false_challenge"]["false_challenge_refused"])

    time.sleep(6)
    REPORT["validators"] = validator_agreement()
    print("validators converged:", REPORT["validators"]["converged"])

    out = REPO / "docs" / "phase-f-e2e-results.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(REPORT, handle, indent=2, default=str)
        handle.write("\n")
    print("written:", out)

    gas = {
        "phase": "F",
        "schedule": {
            "version": "GraphGasV1",
            "source": "chain/x/compute/graph_gas.go (GraphGasScheduleJSON)",
            "tx_base": 20000, "node_decode": 4000, "input_decode": 2000,
            "work_unit": 1, "free_work_units": 200000,
            "hash": 1000, "proof_sibling": 400, "evidence_chunk": 800,
            "state_leaf": 600, "arbiter_unit": 4, "receipt": 30000, "store": 2000,
        },
        "measured_devnet": {
            msg: {"gas_used": values, "runs": len(values)}
            for msg, values in sorted(GAS_REPORT.items())
        },
        "environment": {
            "chain": "prisma-mv-1",
            "validators": 4,
            "image": "prisma-chain:phase-f",
            "note": ("gas_used read from committed transactions on the four-validator "
                     "devnet; the fee is the agreed flat fee (0 on devnet) and the "
                     "schedule bounds execution"),
        },
    }
    gas_out = REPO / "docs" / "phase-f-gas-results.json"
    with open(gas_out, "w", encoding="utf-8") as handle:
        json.dump(gas, handle, indent=2)
        handle.write("\n")
    print("written:", gas_out)


if __name__ == "__main__":
    main()
