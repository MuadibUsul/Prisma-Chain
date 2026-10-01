package main

import (
	"fmt"
	"os"

	svrcmd "github.com/cosmos/cosmos-sdk/server/cmd"
	"prismachain/chain/cmd/prismad/cmd"
)

func main() {
	root := cmd.NewRootCmd()
	if err := svrcmd.Execute(root, "PRISMA", cmd.DefaultNodeHome()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
