// Command netconfig generates, verifies and provisions Prisma testnet
// network configuration bundles (roadmap B1-03).
//
// All frozen protocol parameters are imported from the chain code itself
// (prismachain/chain/x/compute), never retyped, so the generated bundle
// cannot drift from the frozen constants silently.
//
// Subcommands:
//
//	netconfig newseeds --count N
//	netconfig generate --spec spec.json --out DIR [--keep-homes DIR]
//	netconfig verify   --dir DIR [--prismad path]
//	netconfig provision --spec spec.json --bundle DIR --moniker M --home DIR
package main

import (
	"fmt"
	"os"
)

func usage() {
	fmt.Fprint(os.Stderr, `netconfig — Prisma testnet network configuration tool (B1-03)

usage:
  netconfig newseeds --count N
      print N fresh base64 validator seeds (keep them secret; only public
      keys ever enter the generated artifacts)

  netconfig generate --spec spec.json --out DIR [--keep-homes DIR]
      run the full genesis ceremony and write the public bundle to DIR
      (chain-id, genesis, params, validators, README, checksum manifest)

  netconfig verify --dir DIR [--prismad path]
      verify a bundle: manifest checksums (hard error on mismatch),
      chain-id rule, denom policy, devnet-leak checks, and the SDK-level
      genesis validation via prismad

  netconfig provision --spec spec.json --bundle DIR --moniker M --home DIR
      materialize a runnable node home for one validator (private keys are
      written only inside DIR)
`)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "newseeds":
		err = cmdNewSeeds(os.Args[2:])
	case "generate":
		err = cmdGenerate(os.Args[2:])
	case "verify":
		err = cmdVerify(os.Args[2:])
	case "provision":
		err = cmdProvision(os.Args[2:])
	case "-h", "--help", "help":
		usage()
		return
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "netconfig: %v\n", err)
		os.Exit(1)
	}
}
