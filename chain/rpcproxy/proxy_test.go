package rpcproxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"

	"testing"
	"time"
)

func upstream(t *testing.T) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/status", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"result": map[string]string{"node": "ok"}})
	})
	mux.HandleFunc("/block", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"result":{"block":"1"}}`)
	})
	// Deliberately reachable upstream surfaces that the front door must block.
	mux.HandleFunc("/dump_consensus_state", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"result":{"heap":"everything"}}`)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			body, _ := io.ReadAll(r.Body)
			var req struct {
				ID     any    `json:"id"`
				Method string `json:"method"`
			}
			if err := json.Unmarshal(body, &req); err == nil {
				if req.Method == "broadcast_tx_commit" {
					io.WriteString(w, `{"result":{"code":0}}`)
					return
				}
				if req.Method == "slow" {
					time.Sleep(500 * time.Millisecond)
					io.WriteString(w, `{"result":{}}`)
					return
				}
			}
			json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": 1, "result": map[string]string{"ok": "true"}})
			return
		}
		io.WriteString(w, `{"result":{}}`)
	})
	s := httptest.NewServer(mux)
	t.Cleanup(s.Close)
	return s
}

func newProxy(t *testing.T, upstreamURL string, mutate func(*Policy)) *Proxy {
	t.Helper()
	policy := ReadOnlyPolicy()
	policy.RatePerSecond = 1e6
	policy.Burst = 1e6
	if mutate != nil {
		mutate(&policy)
	}
	p, err := New(policy, upstreamURL)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func do(t *testing.T, p *Proxy, req *http.Request) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	p.ServeHTTP(rec, req)
	return rec
}

func get(t *testing.T, p *Proxy, path string) *httptest.ResponseRecorder {
	t.Helper()
	return do(t, p, httptest.NewRequest(http.MethodGet, path, nil))
}

func post(t *testing.T, p *Proxy, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	return do(t, p, req)
}

func TestAllowsReadOnlySurface(t *testing.T) {
	up := upstream(t)
	p := newProxy(t, up.URL, nil)
	if rec := get(t, p, "/status"); rec.Code != http.StatusOK {
		t.Fatalf("GET /status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
	if rec := get(t, p, "/block?height=1"); rec.Code != http.StatusOK {
		t.Fatalf("GET /block = %d, want 200", rec.Code)
	}
	if rec := post(t, p, `{"jsonrpc":"2.0","id":1,"method":"status"}`); rec.Code != http.StatusOK {
		t.Fatalf("POST status = %d, want 200", rec.Code)
	}
}

func TestDeniesWritesAndUnsafe(t *testing.T) {
	up := upstream(t)
	p := newProxy(t, up.URL, nil)
	rec := post(t, p, `{"jsonrpc":"2.0","id":1,"method":"broadcast_tx_commit","params":{}}`)
	if rec.Code != http.StatusForbidden || !strings.Contains(rec.Body.String(), "broadcast_tx_commit") {
		t.Fatalf("write method: code=%d body=%s", rec.Code, rec.Body.String())
	}
	if rec := get(t, p, "/dump_consensus_state"); rec.Code != http.StatusForbidden {
		t.Fatalf("GET /dump_consensus_state = %d, want 403", rec.Code)
	}
	rec = post(t, p, `[{"jsonrpc":"2.0","id":1,"method":"status"},{"jsonrpc":"2.0","id":2,"method":"unsafe_flush_mempool"}]`)
	if rec.Code != http.StatusForbidden || !strings.Contains(rec.Body.String(), "unsafe_flush_mempool") {
		t.Fatalf("batch with one denied method: code=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestDeniesUnparseableBody(t *testing.T) {
	up := upstream(t)
	p := newProxy(t, up.URL, nil)
	if rec := post(t, p, `not json at all`); rec.Code != http.StatusForbidden {
		t.Fatalf("invalid JSON = %d, want 403", rec.Code)
	}
}

func TestSizeCap(t *testing.T) {
	up := upstream(t)
	p := newProxy(t, up.URL, func(policy *Policy) { policy.MaxBodyBytes = 256 })
	big := `{"jsonrpc":"2.0","id":1,"method":"status","params":{"pad":"` + strings.Repeat("x", 512) + `"}}`
	rec := post(t, p, big)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversize body = %d, want 413 (%s)", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "cap") {
		t.Fatalf("413 body lacks an explicit reason: %s", rec.Body.String())
	}
}

func TestRateLimitPerIP(t *testing.T) {
	up := upstream(t)
	p := newProxy(t, up.URL, func(policy *Policy) {
		policy.RatePerSecond = 1
		policy.Burst = 2
	})
	frozen := time.Unix(0, 0)
	p.now = func() time.Time { return frozen }

	for i := 0; i < 2; i++ {
		if rec := get(t, p, "/status"); rec.Code != http.StatusOK {
			t.Fatalf("request %d = %d, want 200", i, rec.Code)
		}
	}
	rec := get(t, p, "/status")
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("third request = %d, want 429", rec.Code)
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Fatalf("429 without Retry-After")
	}
	frozen = frozen.Add(time.Second)
	if rec := get(t, p, "/status"); rec.Code != http.StatusOK {
		t.Fatalf("after refill = %d, want 200", rec.Code)
	}
}

func TestUpstreamTimeoutIsBounded(t *testing.T) {
	up := upstream(t)
	p := newProxy(t, up.URL, func(policy *Policy) {
		policy.UpstreamTimeout = 100 * time.Millisecond
		// Allow "slow" explicitly: the allow-list itself is covered above,
		// this test is about the bounded upstream wait.
		policy.AllowMethods = append(policy.AllowMethods, "slow")
	})
	rec := post(t, p, `{"jsonrpc":"2.0","id":1,"method":"slow"}`)
	if rec.Code != http.StatusGatewayTimeout {
		t.Fatalf("slow upstream = %d, want 504 (%s)", rec.Code, rec.Body.String())
	}
}

func TestHealthEndpoint(t *testing.T) {
	up := upstream(t)
	p := newProxy(t, up.URL, nil)
	rec := get(t, p, "/healthz")
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"status":"ok"`) {
		t.Fatalf("/healthz = %d %s", rec.Code, rec.Body.String())
	}
	up.Close()
	rec = get(t, p, "/healthz")
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("/healthz with dead upstream = %d, want 503", rec.Code)
	}
}

func TestWebsocketRefused(t *testing.T) {
	up := upstream(t)
	p := newProxy(t, up.URL, nil)
	req := httptest.NewRequest(http.MethodGet, "/websocket", nil)
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Upgrade", "websocket")
	if rec := do(t, p, req); rec.Code != http.StatusNotImplemented {
		t.Fatalf("websocket = %d, want 501", rec.Code)
	}
}

func TestMetricsCounters(t *testing.T) {
	up := upstream(t)
	p := newProxy(t, up.URL, func(policy *Policy) {
		policy.RatePerSecond = 1
		policy.Burst = 2
	})
	frozen := time.Unix(0, 0)
	p.now = func() time.Time { return frozen }
	get(t, p, "/status")                                                  // ok, 1 token left
	post(t, p, `{"jsonrpc":"2.0","id":1,"method":"broadcast_tx_commit"}`) // denied, 0 tokens left
	get(t, p, "/status")                                                  // rate limited
	rec := get(t, p, "/metrics")
	body := rec.Body.String()
	for _, want := range []string{
		"prisma_rpc_proxy_requests_total",
		"prisma_rpc_proxy_method_denied_total 1",
		"prisma_rpc_proxy_rate_limited_total 1",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("metrics missing %q:\n%s", want, body)
		}
	}
}
