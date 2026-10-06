package proxy

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	"synapse-gateway/internal/cache"
	"synapse-gateway/internal/client"
)

type ChatCompletionRequest struct {
	Prompt string `json:"prompt"`
}

type ChatCompletionResponse struct {
	Response  string `json:"response"`
	HitType   string `json:"hit_type"`
	LatencyMs int64  `json:"latency_ms"`
}

type ErrorResponse struct {
	Error string `json:"error"`
}

type GatewayHandler struct {
	cache     *cache.SemanticCache
	client    *client.GeminiClient
	threshold float32
}

func NewGatewayHandler(c *cache.SemanticCache, gc *client.GeminiClient, threshold float32) *GatewayHandler {
	return &GatewayHandler{
		cache:     c,
		client:    gc,
		threshold: threshold,
	}
}

func (h *GatewayHandler) HandlePrompt(prompt string) (string, string, time.Duration, error) {
	startTime := time.Now()

	// 1. Exact Hit (L1)
	if resp, hitType, ok := h.cache.Get(prompt, nil, h.threshold); ok {
		return resp, hitType, time.Since(startTime), nil
	}

	// 2. Semantic Hit (L2)
	vec, err := h.client.GetEmbedding(prompt)
	if err == nil {
		if resp, hitType, ok := h.cache.Get(prompt, vec, h.threshold); ok {
			return resp, hitType, time.Since(startTime), nil
		}
	} else {
		log.Printf("Warning: failed to compute embedding: %v", err)
	}

	// 3. Cache Miss: Upstream Call
	answer, err := h.client.GenerateCompletionWithRetry(prompt, 3)
	if err != nil {
		return "", "ERROR", time.Since(startTime), err
	}

	// 4. Save to Cache
	if vec != nil {
		h.cache.Set(prompt, vec, answer)
	}

	return answer, "CACHE_MISS (Fetched Upstream)", time.Since(startTime), nil
}

func (h *GatewayHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "only POST method is accepted"})
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Prompt == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{Error: "missing or invalid 'prompt' field in request body"})
		return
	}

	log.Printf("[Incoming Request] Prompt: %q", req.Prompt)

	answer, hitType, duration, err := h.HandlePrompt(req.Prompt)
	if err != nil {
		w.WriteHeader(http.StatusBadGateway)
		json.NewEncoder(w).Encode(ErrorResponse{Error: err.Error()})
		return
	}

	log.Printf("[Completed] Source: %s | Latency: %dms", hitType, duration.Milliseconds())
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(ChatCompletionResponse{
		Response:  answer,
		HitType:   hitType,
		LatencyMs: duration.Milliseconds(),
	})
}
