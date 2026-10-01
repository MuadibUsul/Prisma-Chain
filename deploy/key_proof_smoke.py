"""Reject network-key squatting on the live local chain."""

import base64
import sys
import uuid
from pathlib import Path

from chain_smoke import CHAIN_ID, height, wait_for
from compute_smoke import account, query_worker, submit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "network"))
from prisma_network.core import Identity


def main() -> None:
    wait_for(lambda: height() >= 1, "first block")
    suffix = uuid.uuid4().hex[:12]
    victim_name, attacker_name = "key-owner-" + suffix, "key-squatter-" + suffix
    victim, attacker = account(victim_name), account(attacker_name)
    identity = Identity.generate()
    for address in (victim, attacker):
        submit("bank", "send", "validator", address, "3000000uprsm", signer="validator")
    key = identity.public_key.hex()
    victim_proof = identity.network_key_proof_hex(CHAIN_ID, victim)
    other_chain_proof = identity.network_key_proof_hex("another-chain", victim)
    base = ("compute", "bond-worker", "--amount", "1000000", "--network-public-key", key)
    submit(*base, signer=attacker_name, expected_code=1)
    submit(*base, "--network-key-proof", victim_proof, signer=attacker_name, expected_code=1)
    submit(*base, "--network-key-proof", other_chain_proof, signer=victim_name, expected_code=1)
    if query_worker(attacker).get("network_public_key"):
        raise RuntimeError("attacker reserved a network key without its account-bound proof")
    submit(*base, "--network-key-proof", victim_proof, signer=victim_name)
    bonded = query_worker(victim)
    if (int(bonded["bonded_uprsm"]) != 1_000_000
            or base64.b64decode(bonded.get("network_public_key", "")) != identity.public_key):
        raise RuntimeError("valid network key proof did not register the bonded key")
    submit("compute", "bond-worker", "--amount", "1", signer=victim_name)
    print("PASS: local chain rejected missing, stolen-account and wrong-chain key proofs; "
          "accepted the rightful bonded key and a proof-free top-up")


if __name__ == "__main__":
    main()
