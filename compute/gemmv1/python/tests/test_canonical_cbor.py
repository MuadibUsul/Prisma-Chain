"""Canonical CBOR self-checks mirroring encoding_test.go in Go."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemmv1.canonical_cbor import CanonicalCBORSError, encode_canonical  # noqa: E402


class TestCanonicalCBOR(unittest.TestCase):
    def test_rfc8949_appendix_a_subset(self):
        cases = [
            (0, "00"),
            (23, "17"),
            (24, "1818"),
            (100, "1864"),
            (1000, "1903e8"),
            (1000000, "1a000f4240"),
            (1000000000000, "1b000000e8d4a51000"),
            (18446744073709551615, "1bffffffffffffffff"),
            (-1, "20"),
            (-10, "29"),
            (-100, "3863"),
            (-1000, "3903e7"),
            (b"", "40"),
            (b"\x01\x02\x03\x04", "4401020304"),
            ("", "60"),
            ("a", "6161"),
            ("IETF", "6449455446"),
            ([], "80"),
            ([1, 2, 3], "83010203"),
        ]
        for value, want_hex in cases:
            self.assertEqual(encode_canonical(value).hex(), want_hex, repr(value))

    def test_map_key_ordering_is_bytewise_on_encodings(self):
        got = encode_canonical({"zeta": 1, "alpha": "IETF", "data": b"\x01"}).hex()
        want = (
            "a3"
            + "646461746141" + "01"  # data
            + "647a657461" + "01"    # zeta
            + "65616c706861" + "6449455446"  # alpha
        )
        self.assertEqual(got, want)

    def test_rejections(self):
        # Note: dicts are the canonical object form in Python (mirroring Go
        # structs), so plain string-keyed dicts are valid here.
        for bad in (True, 3.14, None, [None], {1: b""}):
            with self.assertRaises(CanonicalCBORSError):
                encode_canonical(bad)


if __name__ == "__main__":
    unittest.main()
