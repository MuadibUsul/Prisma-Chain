"""Independent float64 reference of one Qwen3 layer (Reference B) plus the
row permutation that maps HF's rotate-half RoPE onto canonical adjacent
pair rotation (Reference A comparison lives in f1_accuracy.py).

Reference B is deliberately hand-written from the pinned config and the
raw safetensors weights: no transformers code is involved, so an
architecture mistake cannot hide behind the official implementation.

Layout note (critical): HF rotates the pair (x[i], x[i + head_dim/2]).
Canonical ROPE_FIXED_V1 rotates the adjacent pair (x[2j], x[2j+1]) with
frequency index j. The permutation

    perm[2j] = j,  perm[2j+1] = head_dim/2 + j

applied to the Q and K head coordinates (weights and norms included)
turns canonical adjacent rotation into exactly HF's rotate-half, and it
leaves the attention scores unchanged because both operands are permuted
identically.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen3-0.6b-base"


def load_qwen3_block_config():
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    return {
        "hidden_size": cfg["hidden_size"],
        "intermediate_size": cfg["intermediate_size"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "head_dim": cfg["head_dim"],
        "rms_norm_eps": cfg["rms_norm_eps"],
        "rope_theta": cfg["rope_theta"],
        "attention_bias": cfg["attention_bias"],
        "max_position_embeddings": cfg["max_position_embeddings"],
    }


_TORCH_TENSORS = None


def _load_all():
    """Torch tensors of the whole checkpoint (bfloat16 -> float32 exact)."""
    global _TORCH_TENSORS
    if _TORCH_TENSORS is None:
        import torch
        from safetensors.torch import load_file
        _TORCH_TENSORS = load_file(str(MODEL_DIR / "model.safetensors"))
    return _TORCH_TENSORS


def load_layer_weights(layer: int = 0):
    """Load float32 numpy weights of one layer from the raw safetensors."""
    import torch

    tensors = {k: v.to(torch.float32).numpy() for k, v in _load_all().items()}
    prefix = f"model.layers.{layer}."
    wanted = {}
    for key, value in tensors.items():
        if key.startswith(prefix):
            short = key[len(prefix):]
            wanted[short] = value.astype(np.float32)
    required = [
        "input_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    ]
    for name in required:
        if name not in wanted:
            raise KeyError(f"missing weight {name}; have {sorted(wanted)}")
    biases = [n for n in wanted if n.endswith(".bias")]
    if biases:
        raise ValueError(f"unexpected biases in layer {layer}: {biases}")
    return wanted


def qwen_rope_permutation(head_dim: int) -> np.ndarray:
    """perm[new] = old index; new(2j)=j, new(2j+1)=d/2+j."""
    half = head_dim // 2
    perm = np.zeros(head_dim, dtype=np.int64)
    perm[0::2] = np.arange(half)
    perm[1::2] = np.arange(half) + half
    return perm


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    var = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float64)
    return (x / np.sqrt(var + eps)) * weight


def rope_rotate_half(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """HF rotate-half RoPE: rotate (x[:half], x[half:]) per position."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    first = x1 * cos - x2 * sin
    second = x1 * sin + x2 * cos
    return np.concatenate([first, second], axis=-1)


def rope_adjacent(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Canonical ROPE_FIXED_V1: rotate adjacent pairs (2j, 2j+1)."""
    even = x[..., 0::2]
    odd = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


def rope_table(seq: int, head_dim: int, theta: float, dtype=np.float64):
    """cos/sin tables of shape [seq, head_dim/2] with inv_freq = theta^(-2j/d)."""
    half = head_dim // 2
    inv_freq = theta ** (-2.0 * np.arange(half, dtype=np.float64) / head_dim)
    positions = np.arange(seq, dtype=np.float64)
    angles = np.outer(positions, inv_freq)
    return np.cos(angles).astype(dtype), np.sin(angles).astype(dtype)


def block_forward(hidden: np.ndarray, weights, cfg, *, dtype=np.float64,
                  rotate: str = "half", permute_qk: bool = False):
    """One pre-norm Qwen3 decoder block in float64 (Reference B).

    hidden: [seq, hidden_size]; returns [seq, hidden_size].
    rotate: "half" (HF) or "adjacent" (canonical).
    permute_qk: apply the canonical coordinate permutation to Q and K
    (weights and norms) before the rotation — used to prove the
    permutation maps one rotation convention onto the other.
    """
    seq = hidden.shape[0]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    x = hidden.astype(dtype)

    h = rms_norm(x, weights["input_layernorm.weight"].astype(dtype), eps)
    wq = weights["self_attn.q_proj.weight"].astype(dtype)
    wk = weights["self_attn.k_proj.weight"].astype(dtype)
    wv = weights["self_attn.v_proj.weight"].astype(dtype)
    wo = weights["self_attn.o_proj.weight"].astype(dtype)
    q = h @ wq.T
    k = h @ wk.T
    v = h @ wv.T

    q = q.reshape(seq, heads, head_dim)
    k = k.reshape(seq, kv_heads, head_dim)
    v = v.reshape(seq, kv_heads, head_dim)

    qn = weights["self_attn.q_norm.weight"].astype(dtype)
    kn = weights["self_attn.k_norm.weight"].astype(dtype)
    if permute_qk:
        perm = qwen_rope_permutation(head_dim)
        # Permute head coordinates WITHIN each head block of rows.
        q_idx = np.concatenate([h * head_dim + perm for h in range(heads)])
        k_idx = np.concatenate([h * head_dim + perm for h in range(kv_heads)])
        q = (h @ wq[q_idx, :].T).reshape(seq, heads, head_dim)
        k = (h @ wk[k_idx, :].T).reshape(seq, kv_heads, head_dim)
        qn = qn[perm]
        kn = kn[perm]

    q = rms_norm(q, qn, eps)
    k = rms_norm(k, kn, eps)

    cos, sin = rope_table(seq, head_dim, cfg["rope_theta"], dtype)
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    if rotate == "half":
        q = rope_rotate_half(q, cos, sin)
        k = rope_rotate_half(k, cos, sin)
    else:
        q = rope_adjacent(q, cos, sin)
        k = rope_adjacent(k, cos, sin)

    # GQA: KV head h serves Q heads [h * (heads // kv_heads), ...)
    group = heads // kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    context = np.empty_like(q)
    causal = np.triu(np.full((seq, seq), -np.inf, dtype=dtype), k=1)
    for qh in range(heads):
        kvh = qh // group
        scores = (q[:, qh, :] @ k[:, kvh, :].T) * scale
        scores = scores + causal
        mx = scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores - mx)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        context[:, qh, :] = probs @ v[:, kvh, :]

    attn = context.reshape(seq, heads * head_dim) @ wo.T
    x1 = x + attn

    h2 = rms_norm(x1, weights["post_attention_layernorm.weight"].astype(dtype), eps)
    gate = h2 @ weights["mlp.gate_proj.weight"].astype(dtype).T
    up = h2 @ weights["mlp.up_proj.weight"].astype(dtype).T
    silu = gate / (1.0 + np.exp(-gate))
    mlp = (silu * up) @ weights["mlp.down_proj.weight"].astype(dtype).T
    return x1 + mlp


def official_block_output(hidden_states: np.ndarray, layer: int = 0):
    """Reference A: the official transformers Qwen3 implementation."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32, attn_implementation="eager",
        local_files_only=True
    )
    model.eval()
    batch = torch.from_numpy(np.asarray(hidden_states, dtype=np.float32))[None, :, :]
    with torch.no_grad():
        out = model(inputs_embeds=batch, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer + 1][0].numpy().astype(np.float64)
