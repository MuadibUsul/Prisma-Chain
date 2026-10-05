package cmd

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

// NewHealthCmd checks that the node's RPC answers within a bounded timeout.
// It is written for container HEALTHCHECK usage: no config reads, no files,
// exit 0 when healthy, exit 1 with a reason on stderr otherwise.
//
// "Healthy" means: /status answered HTTP 200 with a parseable payload that
// carries a network id. A node still catching up is healthy (liveness), so
// catching_up is reported but not enforced.
func NewHealthCmd() *cobra.Command {
	var rpc string
	var timeout time.Duration
	cmd := &cobra.Command{
		Use:               "health",
		Short:             "Check the node RPC health (bounded timeout; for container healthchecks)",
		Args:              cobra.NoArgs,
		PersistentPreRunE: func(*cobra.Command, []string) error { return nil },
		RunE: func(cmd *cobra.Command, _ []string) error {
			client := &http.Client{Timeout: timeout}
			url := strings.TrimSuffix(rpc, "/") + "/status"
			resp, err := client.Get(url)
			if err != nil {
				return fmt.Errorf("unhealthy: %v", err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				return fmt.Errorf("unhealthy: HTTP %d from %s", resp.StatusCode, url)
			}
			var payload struct {
				Result struct {
					NodeInfo struct {
						Network string `json:"network"`
					} `json:"node_info"`
					SyncInfo struct {
						LatestBlockHeight string `json:"latest_block_height"`
						CatchingUp        bool   `json:"catching_up"`
					} `json:"sync_info"`
				} `json:"result"`
			}
			if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&payload); err != nil {
				return fmt.Errorf("unhealthy: bad /status payload: %v", err)
			}
			if payload.Result.NodeInfo.Network == "" {
				return errors.New("unhealthy: /status has no network id")
			}
			fmt.Fprintf(cmd.OutOrStdout(), "ok chain_id=%s height=%s catching_up=%v\n",
				payload.Result.NodeInfo.Network,
				payload.Result.SyncInfo.LatestBlockHeight,
				payload.Result.SyncInfo.CatchingUp)
			return nil
		},
	}
	cmd.Flags().StringVar(&rpc, "rpc", "http://127.0.0.1:26657", "node RPC endpoint to check")
	cmd.Flags().DurationVar(&timeout, "timeout", 3*time.Second, "bounded request timeout")
	return cmd
}
