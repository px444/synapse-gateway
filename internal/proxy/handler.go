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
	gemini    *client.GeminiClient
	groq      *client.GroqClient
	threshold float32
}

func NewGatewayHandler(c *cache.SemanticCache, gc *client.GeminiClient, groq *client.GroqClient, threshold float32) *GatewayHandler {
	return &GatewayHandler{
		cache:     c,
		gemini:    gc,
		groq:      groq,
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
	vec, err := h.gemini.GetEmbedding(prompt)
	if err == nil {
		if resp, hitType, ok := h.cache.Get(prompt, vec, h.threshold); ok {
			return resp, hitType, time.Since(startTime), nil
		}
	} else {
		log.Printf("Warning: failed to compute embedding: %v", err)
	}

	// 3. Primary Provider (Gemini with Retry)
	answer, err := h.gemini.GenerateCompletionWithRetry(prompt, 3)
	hitSource := "CACHE_MISS (Gemini Primary)"

	// 4. Failover to Secondary Provider (Groq) if Gemini fails
	if err != nil {
		log.Printf("⚠️ Primary provider (Gemini) failed: %v. Initiating failover to Groq...", err)
		if h.groq != nil {
			fallbackAnswer, fallbackErr := h.groq.GenerateCompletion(prompt)
			if fallbackErr == nil {
				answer = fallbackAnswer
				hitSource = "CACHE_MISS (Groq Failover)"
				err = nil
			} else {
				log.Printf("❌ Secondary provider (Groq) also failed: %v", fallbackErr)
			}
		}
	}

	if err != nil {
		return "", "ERROR", time.Since(startTime), err
	}

	// 5. Save to Cache so future requests hit RAM
	if vec != nil {
		h.cache.Set(prompt, vec, answer)
	}

	return answer, hitSource, time.Since(startTime), nil
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
