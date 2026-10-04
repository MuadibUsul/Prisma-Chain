package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"

	"prismachain/compute/canonical"
)

func main() {
	raw, err := os.ReadFile("testdata/f5c_qwen3_block_v2.json")
	if err != nil { panic(err) }
	var g canonical.GraphDescriptorV2
	if err := json.Unmarshal(raw, &g); err != nil { panic(err) }
	enc, err := canonical.EncodeCanonical(&g)
	if err != nil { panic(err) }
	sum := sha256.Sum256(enc)
	fmt.Println("go hash:", hex.EncodeToString(sum[:]))
	n := 220
	if len(enc) < n { n = len(enc) }
	fmt.Println("go head:", hex.EncodeToString(enc[:n]))
	fmt.Println("go len:", len(enc))
}
