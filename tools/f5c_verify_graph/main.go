package main

import (
	"encoding/hex"
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
	var g canonical.GraphDescriptorV2
	if err := json.Unmarshal(raw, &g); err != nil {
		panic(err)
	}
	if err := g.ValidateV2(); err != nil {
		fmt.Println("VALIDATE:", err)
	}
	id, err := g.GraphIDV2()
	if err != nil {
		panic(err)
	}
	fmt.Println("go graph_id:", hex.EncodeToString(id[:]))
	enc, err := canonical.EncodeCanonical(&g)
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile("testdata/f5c_tmp/go_desc.bin", enc, 0o644); err != nil {
		panic(err)
	}
}
