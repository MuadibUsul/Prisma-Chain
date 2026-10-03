"""GEMM v1 protocol core in Python.

Mirrors compute/gemmv1 (Go): canonical CBOR, domain-separated hashing,
canonical tensor layout and the reference INT8 GEMM. The cross-language
vector file at compute/gemmv1/testdata/cross_language_vectors.json enforces
bit-exact agreement between the two implementations.
"""

from .canonical_cbor import encode_canonical
from .protocol import (
    ProtocolVersion,
    Operator,
    ArithmeticSpec,
    DomainTask,
    DomainAssignment,
    DomainInputTile,
    DomainOutputTile,
    DomainTraceState,
    DomainVWR,
    task_id,
    assignment_id,
    leaf_input_tile,
    leaf_output_tile,
    leaf_trace_state,
    build_matrix_roots,
    merkle_root,
    receipt_id,
)
from .tensors import (
    int32s_to_canonical,
    canonical_to_int32s,
    extract_a_tile,
    extract_b_tile,
    output_tiles,
    reference_gemm,
)
from .testgen import gen_test_matrix

__all__ = [
    "encode_canonical",
    "ProtocolVersion",
    "Operator",
    "ArithmeticSpec",
    "DomainTask",
    "DomainAssignment",
    "DomainInputTile",
    "DomainOutputTile",
    "DomainTraceState",
    "DomainVWR",
    "task_id",
    "assignment_id",
    "leaf_input_tile",
    "leaf_output_tile",
    "leaf_trace_state",
    "build_matrix_roots",
    "merkle_root",
    "receipt_id",
    "int32s_to_canonical",
    "canonical_to_int32s",
    "extract_a_tile",
    "extract_b_tile",
    "output_tiles",
    "reference_gemm",
    "gen_test_matrix",
]
