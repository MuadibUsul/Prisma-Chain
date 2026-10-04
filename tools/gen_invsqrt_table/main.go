// Command gen_invsqrt_table prints the pinned CanonicalMathV1 invsqrt
// table generated with exact big.Int square roots. Run from the repo root
// after changing the format spec:
//
//	go run ./tools/gen_invsqrt_table
package main

import (
	"fmt"

	"prismachain/compute/canonical"
)

func main() {
	table := canonical.BuildInvSqrtTable()
	fmt.Print("var invSqrtTable = [INV_SQRT_TABLE_POINTS]int32{")
	for i, v := range table {
		if i%8 == 0 {
			fmt.Print("\n\t")
		}
		fmt.Printf("%d,", v)
	}
	fmt.Println("\n}")
}
