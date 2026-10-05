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
	var rc canonical.GraphResultCommitV3
	if err := json.Unmarshal(raw, &rc); err != nil {
		panic(err)
	}
	if !canonical.VerifyGraphResultCommitV3Signature(&rc) {
		unsigned := rc
		unsigned.Signature = nil
		enc, _ := canonical.EncodeCanonical(&unsigned)
		if err := os.WriteFile("testdata/f5c_tmp/go_preimage.bin", enc, 0o644); err != nil {
			panic(err)
		}
		fmt.Printf("SIGNATURE FAIL | preimage len %d (dumped)\n", len(enc))
		os.Exit(1)
	}
	fmt.Println("COMMIT OK")
}
