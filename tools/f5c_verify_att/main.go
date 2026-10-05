// f5c-verify-att: verifies a GraphDAAttestationV2 JSON produced by the
// Python provider simulator against the Go canonical verifier (cross-
// language signing preimage check).
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"prismachain/compute/canonical"
)

func main() {
	raw, err := os.ReadFile(os.Args[1])
	if err != nil {
		panic(err)
	}
	var att canonical.GraphDAAttestationV2
	if err := json.Unmarshal(raw, &att); err != nil {
		panic(err)
	}
	if err := canonical.ValidateGraphDAAttestationV2(&att); err != nil {
		fmt.Println("VALIDATE FAIL:", err)
		os.Exit(1)
	}
	if !canonical.VerifyGraphDAAttestationV2Signature(&att) {
		unsigned := att
		unsigned.Signature = nil
		enc, _ := canonical.EncodeCanonical(&unsigned)
		n := 80
		if len(enc) < n {
			n = len(enc)
		}
		fmt.Printf("SIGNATURE FAIL | preimage len %d head %x\n", len(enc), enc[:n])
		os.Exit(1)
	}
	fmt.Println("ATTESTATION OK")
}
