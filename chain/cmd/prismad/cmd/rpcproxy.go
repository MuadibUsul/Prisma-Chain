package cmd

import (
	"context"
	"fmt"
	"net/http"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"prismachain/chain/rpcproxy"
)

// NewRPCProxyCmd runs the public RPC front door (B1-02): a policy-enforcing
// reverse proxy in front of a node's CometBFT RPC. It holds no keys, signs
// nothing and reads no node config, so it can face the public internet
// while the node itself keeps its RPC bound to localhost.
func NewRPCProxyCmd() *cobra.Command {
	var (
		listen       string
		upstream     string
		rate         float64
		burst        int
		maxBodyBytes int64
		timeout      time.Duration
		allowMethods string
		allowPaths   string
	)
	cmd := &cobra.Command{
		Use:   "rpc-proxy",
		Short: "Public RPC front door: allow-lists, per-IP rate limits, size caps, bounded timeouts, metrics, health",
		Args:  cobra.NoArgs,
		// No node config reads: this process must start on a clean machine
		// and never touch key material.
		PersistentPreRunE: func(*cobra.Command, []string) error { return nil },
		RunE: func(cmd *cobra.Command, _ []string) error {
			policy := rpcproxy.ReadOnlyPolicy()
			policy.RatePerSecond = rate
			policy.Burst = burst
			policy.MaxBodyBytes = maxBodyBytes
			policy.UpstreamTimeout = timeout
			if allowMethods != "" {
				policy.AllowMethods = splitCSV(allowMethods)
			}
			if allowPaths != "" {
				policy.AllowPaths = splitCSV(allowPaths)
			}
			proxy, err := rpcproxy.New(policy, upstream)
			if err != nil {
				return err
			}
			srv := &http.Server{
				Addr:              listen,
				Handler:           proxy,
				ReadHeaderTimeout: 10 * time.Second,
				ReadTimeout:       30 * time.Second,
				WriteTimeout:      60 * time.Second,
				IdleTimeout:       120 * time.Second,
			}
			fmt.Fprintf(cmd.OutOrStdout(),
				"prismad rpc-proxy: listening on %s -> %s (rate %.0f/s burst %d, body cap %d bytes, upstream timeout %s, %d methods, %d paths allowed)\n",
				listen, upstream, policy.RatePerSecond, policy.Burst, policy.MaxBodyBytes,
				policy.UpstreamTimeout, len(policy.AllowMethods), len(policy.AllowPaths))
			ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
			defer stop()
			errCh := make(chan error, 1)
			go func() { errCh <- srv.ListenAndServe() }()
			select {
			case err := <-errCh:
				return err
			case <-ctx.Done():
				shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				return srv.Shutdown(shutdownCtx)
			}
		},
	}
	cmd.Flags().StringVar(&listen, "listen", "0.0.0.0:8545", "public listen address")
	cmd.Flags().StringVar(&upstream, "upstream", "http://127.0.0.1:26657", "node RPC to protect (keep it on localhost)")
	cmd.Flags().Float64Var(&rate, "rate", 20, "per-IP requests per second")
	cmd.Flags().IntVar(&burst, "burst", 40, "per-IP burst allowance")
	cmd.Flags().Int64Var(&maxBodyBytes, "max-body-bytes", 1<<20, "hard request body cap in bytes")
	cmd.Flags().DurationVar(&timeout, "upstream-timeout", 15*time.Second, "bounded upstream timeout")
	cmd.Flags().StringVar(&allowMethods, "allow-methods", "", "comma-separated JSON-RPC method allow-list (default: read-only preset)")
	cmd.Flags().StringVar(&allowPaths, "allow-paths", "", "comma-separated GET path allow-list (default: read-only preset)")
	return cmd
}

func splitCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
