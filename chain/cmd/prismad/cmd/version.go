package cmd

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/spf13/cobra"

	"prismachain/chain/version"
)

// NewVersionCmd prints the build identity and the frozen protocol identity
// embedded in the binary (B1-01b). It is intentionally self-contained: it
// must not read node config or create any files, so it works on a clean
// machine and inside minimal containers.
func NewVersionCmd() *cobra.Command {
	var asJSON bool
	cmd := &cobra.Command{
		Use:   "version",
		Short: "Print software and frozen-protocol version identity",
		Args:  cobra.NoArgs,
		// The root's PersistentPreRunE intercepts node config and may touch
		// the home directory; a version query must have no side effects.
		PersistentPreRunE: func(*cobra.Command, []string) error { return nil },
		RunE: func(cmd *cobra.Command, _ []string) error {
			info := version.Get()
			out := cmd.OutOrStdout()
			if asJSON {
				enc := json.NewEncoder(out)
				enc.SetIndent("", "  ")
				return enc.Encode(info)
			}
			fmt.Fprintf(out, "prismad (Prisma Chain node)\n")
			fmt.Fprintf(out, "  software version : %s\n", info.SoftwareVersion)
			fmt.Fprintf(out, "  git commit       : %s\n", info.GitCommit)
			fmt.Fprintf(out, "  build target     : %s\n", info.BuildTarget)
			fmt.Fprintf(out, "  protocol phase   : %s\n", info.Protocol.Phase)
			fmt.Fprintf(out, "  freeze tag       : %s\n", info.Protocol.FreezeTag)
			fmt.Fprintf(out, "  freeze commit    : %s\n", info.Protocol.FreezeCommit)
			fmt.Fprintf(out, "  graph protocol   : %s\n", info.Protocol.Components["graph_v2"])
			fmt.Fprintf(out, "  graph id v2      : %s\n", info.Protocol.GraphIDV2)
			fmt.Fprintf(out, "  policy id        : %s\n", info.Protocol.PolicyID)
			fmt.Fprintf(out, "  protocol components:\n")
			keys := make([]string, 0, len(info.Protocol.Components))
			for k := range info.Protocol.Components {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				fmt.Fprintf(out, "    %-20s %s\n", k, info.Protocol.Components[k])
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&asJSON, "json", false, "machine-readable JSON output")
	return cmd
}
