// Package rpcproxy is the public RPC front door for testnet nodes
// (roadmap B1-02): a policy-enforcing reverse proxy in front of a node's
// CometBFT RPC.
//
// Enforced policy, all with explicit machine-readable errors:
//
//	request size    hard cap (413) before anything is parsed
//	method allow-list  JSON-RPC methods (POST) and path prefixes (GET) (403)
//	per-IP rate     token bucket, 429 + Retry-After
//	upstream timeout bounded (504)
//	websockets      refused (501) - the proxy is HTTP-only by design
//
// Plus a health endpoint that does not expose the RPC surface (/healthz)
// and Prometheus-style counters (/metrics), and no key material anywhere:
// the proxy holds no keys and signs nothing.
package rpcproxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"path"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Policy is the enforced configuration of the public front door.
type Policy struct {
	// AllowMethods are the JSON-RPC method names permitted over POST.
	AllowMethods []string
	// AllowPaths are the path prefixes permitted over GET ("" entries are
	// ignored); matching is on the first path segment.
	AllowPaths []string
	// MaxBodyBytes caps the request body (413 above it).
	MaxBodyBytes int64
	// RatePerSecond / Burst configure the per-IP token bucket.
	RatePerSecond float64
	Burst         int
	// UpstreamTimeout bounds the whole upstream exchange.
	UpstreamTimeout time.Duration
	// HealthTimeout bounds the /healthz upstream check.
	HealthTimeout time.Duration
}

// ReadOnlyPolicy is the default public profile: cometbft's read-only
// surfaces, no writes, no mempool, no unsafe/control methods, no
// websockets. A public testnet RPC should start here and widen only with
// a recorded reason.
func ReadOnlyPolicy() Policy {
	methods := []string{
		"status", "health", "net_info", "blockchain", "header", "header_by_hash",
		"block", "block_by_hash", "block_results", "commit", "validators",
		"consensus_state", "consensus_params", "abci_info", "abci_query",
		"tx", "tx_search", "genesis",
	}
	paths := []string{
		"status", "health", "net_info", "blockchain", "header", "block",
		"block_results", "commit", "validators", "consensus_state",
		"consensus_params", "abci_info", "abci_query", "tx", "tx_search",
		"genesis", "unconfirmed_txs",
	}
	return Policy{
		AllowMethods:    methods,
		AllowPaths:      paths,
		MaxBodyBytes:    1 << 20, // 1 MiB, comet's own max_body_bytes default range
		RatePerSecond:   20,
		Burst:           40,
		UpstreamTimeout: 15 * time.Second,
		HealthTimeout:   3 * time.Second,
	}
}

type bucket struct {
	tokens float64
	last   time.Time
}

type metrics struct {
	total        atomic.Int64
	ok           atomic.Int64
	status2xx    atomic.Int64
	status4xx    atomic.Int64
	status5xx    atomic.Int64
	rateLimited  atomic.Int64
	methodDenied atomic.Int64
	sizeDenied   atomic.Int64
	websocket    atomic.Int64
	timeouts     atomic.Int64
	upstreamErrs atomic.Int64
	upstreamNs   atomic.Int64
}

// Proxy is the http.Handler implementing the front door.
type Proxy struct {
	policy   Policy
	upstream *url.URL
	client   *http.Client

	mu      sync.Mutex
	buckets map[string]*bucket

	m         metrics
	now       func() time.Time
	maxKeys   int
	closeOnce sync.Once
}

// New builds the proxy for the given policy and upstream (e.g.
// http://127.0.0.1:26657, the node's local RPC).
func New(policy Policy, upstream string) (*Proxy, error) {
	u, err := url.Parse(upstream)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("invalid upstream %q", upstream)
	}
	if policy.MaxBodyBytes <= 0 {
		return nil, fmt.Errorf("MaxBodyBytes must be positive")
	}
	if policy.UpstreamTimeout <= 0 {
		return nil, fmt.Errorf("UpstreamTimeout must be positive")
	}
	if policy.HealthTimeout <= 0 {
		policy.HealthTimeout = 3 * time.Second
	}
	if policy.Burst <= 0 {
		policy.Burst = 1
	}
	if policy.RatePerSecond <= 0 {
		return nil, fmt.Errorf("RatePerSecond must be positive")
	}
	return &Proxy{
		policy:   policy,
		upstream: u,
		client: &http.Client{
			Timeout: policy.UpstreamTimeout,
			// No redirects: an upstream redirect is a policy bypass vector.
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		buckets: map[string]*bucket{},
		now:     time.Now,
		maxKeys: 100_000,
	}, nil
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	p.m.total.Add(1)
	switch r.URL.Path {
	case "/healthz":
		p.handleHealth(w, r)
		return
	case "/metrics":
		p.handleMetrics(w)
		return
	}
	if isUpgrade(r) {
		p.m.websocket.Add(1)
		writeJSONError(w, http.StatusNotImplemented,
			"websockets are not served by the public RPC proxy; use a dedicated websocket endpoint with its own limits")
		p.m.status5xx.Add(1)
		return
	}
	// Rate limit first: floods cost nothing downstream.
	if !p.allow(clientIP(r)) {
		w.Header().Set("Retry-After", "1")
		writeJSONError(w, http.StatusTooManyRequests, "rate limit exceeded for this client address")
		p.m.rateLimited.Add(1)
		p.m.status4xx.Add(1)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, p.policy.MaxBodyBytes)

	switch r.Method {
	case http.MethodGet, http.MethodHead:
		if !p.pathAllowed(r.URL.Path) {
			writeJSONError(w, http.StatusForbidden, fmt.Sprintf("path %q is not in the public RPC allow-list", r.URL.Path))
			p.m.methodDenied.Add(1)
			p.m.status4xx.Add(1)
			return
		}
	case http.MethodPost:
		body, err := io.ReadAll(r.Body)
		if err != nil {
			// http.MaxBytesReader surfaces the cap as a read error.
			writeJSONError(w, http.StatusRequestEntityTooLarge,
				fmt.Sprintf("request body exceeds the %d byte cap", p.policy.MaxBodyBytes))
			p.m.sizeDenied.Add(1)
			p.m.status4xx.Add(1)
			return
		}
		denied, why := p.postAllowed(body)
		if denied {
			writeJSONError(w, http.StatusForbidden, why)
			p.m.methodDenied.Add(1)
			p.m.status4xx.Add(1)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		r.ContentLength = int64(len(body))
	default:
		writeJSONError(w, http.StatusMethodNotAllowed, "only GET and POST are served")
		p.m.status4xx.Add(1)
		return
	}

	started := p.now()
	resp, err := p.forward(r)
	latency := p.now().Sub(started)
	if err != nil {
		if isTimeout(err) {
			p.m.timeouts.Add(1)
			writeJSONError(w, http.StatusGatewayTimeout,
				fmt.Sprintf("upstream did not answer within %s", p.policy.UpstreamTimeout))
		} else {
			p.m.upstreamErrs.Add(1)
			writeJSONError(w, http.StatusBadGateway, "upstream request failed")
		}
		p.m.status5xx.Add(1)
		return
	}
	defer resp.Body.Close()
	p.m.upstreamNs.Add(int64(latency))
	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	n, _ := io.Copy(w, resp.Body)
	if resp.StatusCode < 300 {
		p.m.status2xx.Add(1)
		p.m.ok.Add(n)
	} else if resp.StatusCode < 500 {
		p.m.status4xx.Add(1)
	} else {
		p.m.status5xx.Add(1)
	}
}

func (p *Proxy) forward(r *http.Request) (*http.Response, error) {
	target := *p.upstream
	target.Path = r.URL.Path
	target.RawQuery = r.URL.RawQuery
	req, err := http.NewRequestWithContext(r.Context(), r.Method, target.String(), r.Body)
	if err != nil {
		return nil, err
	}
	req.ContentLength = r.ContentLength
	if ct := r.Header.Get("Content-Type"); ct != "" {
		req.Header.Set("Content-Type", ct)
	}
	return p.client.Do(req)
}

// postAllowed inspects a JSON-RPC body (single or batch) and enforces the
// method allow-list. An unparseable body is refused: a public front door
// only forwards requests it understands.
func (p *Proxy) postAllowed(body []byte) (bool, string) {
	var raw json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil || len(raw) == 0 {
		return true, "request body is not valid JSON-RPC"
	}
	type rpcRequest struct {
		Method string `json:"method"`
	}
	allowed := func(method string) (bool, string) {
		if method == "" {
			return true, "request has no JSON-RPC method"
		}
		for _, m := range p.policy.AllowMethods {
			if m == method {
				return false, ""
			}
		}
		return true, fmt.Sprintf("method %q is not in the public RPC allow-list", method)
	}
	if raw[0] == '[' {
		var batch []rpcRequest
		if err := json.Unmarshal(raw, &batch); err != nil {
			return true, "batch request is not valid JSON-RPC"
		}
		for _, req := range batch {
			if denied, why := allowed(req.Method); denied {
				return denied, why
			}
		}
		return false, ""
	}
	var single rpcRequest
	if err := json.Unmarshal(raw, &single); err != nil {
		return true, "request is not valid JSON-RPC"
	}
	return allowed(single.Method)
}

func (p *Proxy) pathAllowed(urlPath string) bool {
	seg := path.Clean(urlPath)
	seg = strings.TrimPrefix(seg, "/")
	if seg == "" {
		return false
	}
	if i := strings.IndexByte(seg, '/'); i >= 0 {
		seg = seg[:i]
	}
	for _, allowed := range p.policy.AllowPaths {
		if allowed != "" && allowed == seg {
			return true
		}
	}
	return false
}

// allow is the per-IP token bucket.
func (p *Proxy) allow(key string) bool {
	now := p.now()
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.buckets) > p.maxKeys {
		for k := range p.buckets {
			delete(p.buckets, k)
		}
	}
	b, ok := p.buckets[key]
	if !ok {
		b = &bucket{tokens: float64(p.policy.Burst), last: now}
		p.buckets[key] = b
	}
	b.tokens += now.Sub(b.last).Seconds() * p.policy.RatePerSecond
	if b.tokens > float64(p.policy.Burst) {
		b.tokens = float64(p.policy.Burst)
	}
	b.last = now
	if b.tokens < 1 {
		return false
	}
	b.tokens--
	return true
}

func (p *Proxy) handleHealth(w http.ResponseWriter, _ *http.Request) {
	client := &http.Client{Timeout: p.policy.HealthTimeout}
	target := *p.upstream
	target.Path = "/status"
	started := time.Now()
	resp, err := client.Get(target.String())
	if err != nil || resp.StatusCode != http.StatusOK {
		if resp != nil {
			resp.Body.Close()
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprintf(w, `{"status":"unhealthy","upstream":%q}`+"\n", p.upstream.String())
		return
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<16))
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"status":"ok","upstream":%q,"latency_ms":%d}`+"\n",
		p.upstream.String(), time.Since(started).Milliseconds())
}

func (p *Proxy) handleMetrics(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "# TYPE prisma_rpc_proxy_requests_total counter\nprisma_rpc_proxy_requests_total %d\n", p.m.total.Load())
	fmt.Fprintf(w, "# TYPE prisma_rpc_proxy_responses_total counter\n")
	fmt.Fprintf(w, "prisma_rpc_proxy_responses_total{class=\"2xx\"} %d\n", p.m.status2xx.Load())
	fmt.Fprintf(w, "prisma_rpc_proxy_responses_total{class=\"4xx\"} %d\n", p.m.status4xx.Load())
	fmt.Fprintf(w, "prisma_rpc_proxy_responses_total{class=\"5xx\"} %d\n", p.m.status5xx.Load())
	fmt.Fprintf(w, "# TYPE prisma_rpc_proxy_rate_limited_total counter\nprisma_rpc_proxy_rate_limited_total %d\n", p.m.rateLimited.Load())
	fmt.Fprintf(w, "# TYPE prisma_rpc_proxy_method_denied_total counter\nprisma_rpc_proxy_method_denied_total %d\n", p.m.methodDenied.Load())
	fmt.Fprintf(w, "# TYPE prisma_rpc_proxy_size_denied_total counter\nprisma_rpc_proxy_size_denied_total %d\n", p.m.sizeDenied.Load())
	fmt.Fprintf(w, "# TYPE prisma_rpc_proxy_websocket_refused_total counter\nprisma_rpc_proxy_websocket_refused_total %d\n", p.m.websocket.Load())
	fmt.Fprintf(w, "# TYPE prisma_rpc_proxy_upstream_timeouts_total counter\nprisma_rpc_proxy_upstream_timeouts_total %d\n", p.m.timeouts.Load())
	fmt.Fprintf(w, "# TYPE prisma_rpc_proxy_upstream_errors_total counter\nprisma_rpc_proxy_upstream_errors_total %d\n", p.m.upstreamErrs.Load())
	fmt.Fprintf(w, "# TYPE prisma_rpc_proxy_upstream_latency_ns_total counter\nprisma_rpc_proxy_upstream_latency_ns_total %d\n", p.m.upstreamNs.Load())
}

func writeJSONError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	body, _ := json.Marshal(map[string]any{"error": msg, "code": code})
	w.Write(append(body, '\n'))
}

func copyHeaders(dst, src http.Header) {
	for k, vv := range src {
		if k == "Content-Length" {
			continue // the body is re-framed
		}
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

func clientIP(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

func isUpgrade(r *http.Request) bool {
	return strings.EqualFold(r.Header.Get("Upgrade"), "websocket") ||
		strings.Contains(strings.ToLower(r.Header.Get("Connection")), "upgrade")
}

func isTimeout(err error) bool {
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	return strings.Contains(err.Error(), "context deadline exceeded") ||
		strings.Contains(err.Error(), "Client.Timeout")
}
