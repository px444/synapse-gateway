package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"
)

// ==========================================
// 1. Upstream Gemini API Structs
// ==========================================

type EmbedRequest struct {
	Content struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"content"`
}

type EmbedResponse struct {
	Embedding struct {
		Values []float32 `json:"values"`
	} `json:"embedding"`
}

type ChatRequest struct {
	Contents []struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"contents"`
}

type ChatResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

// ==========================================
// 2. Gateway HTTP Request / Response DTOs
// ==========================================

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

// ==========================================
// 3. In-Memory Cache Structures
// ==========================================

type CacheEntry struct {
	Prompt    string
	Embedding []float32
	Response  string
}

type SemanticCache struct {
	mu      sync.RWMutex
	exact   map[string]string
	entries []CacheEntry
}

func NewSemanticCache() *SemanticCache {
	return &SemanticCache{
		exact:   make(map[string]string),
		entries: make([]CacheEntry, 0),
	}
}

func hashPrompt(prompt string) string {
	h := sha256.Sum256([]byte(prompt))
	return hex.EncodeToString(h[:])
}

func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}
	var dot float32
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
	}
	return dot
}

func (c *SemanticCache) Get(prompt string, vec []float32, threshold float32) (string, string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Tier 1: Exact Hash Match
	hash := hashPrompt(prompt)
	if resp, exists := c.exact[hash]; exists {
		return resp, "EXACT_HIT", true
	}

	// Tier 2: Semantic Vector Scan
	if vec == nil {
		return "", "CACHE_MISS", false
	}

	var bestScore float32 = -1.0
	var bestResp string

	for _, entry := range c.entries {
		score := CosineSimilarity(vec, entry.Embedding)
		if score > bestScore {
			bestScore = score
			bestResp = entry.Response
		}
	}

	if bestScore >= threshold {
		return bestResp, fmt.Sprintf("SEMANTIC_HIT (score: %.3f)", bestScore), true
	}

	return "", "CACHE_MISS", false
}

func (c *SemanticCache) Set(prompt string, vec []float32, response string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	hash := hashPrompt(prompt)
	c.exact[hash] = response
	c.entries = append(c.entries, CacheEntry{
		Prompt:    prompt,
		Embedding: vec,
		Response:  response,
	})
}

// ==========================================
// 4. Gemini API Client
// ==========================================

func getEmbedding(apiKey, text string) ([]float32, error) {
	reqPayload := EmbedRequest{}
	reqPayload.Content.Parts = []struct {
		Text string `json:"text"`
	}{{Text: text}}

	jsonBody, err := json.Marshal(reqPayload)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=%s", apiKey)
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding API HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Embedding.Values, nil
}

func generateCompletion(apiKey, prompt string) (string, error) {
	reqPayload := ChatRequest{}
	reqPayload.Contents = []struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	}{
		{
			Parts: []struct {
				Text string `json:"text"`
			}{{Text: prompt}},
		},
	}

	jsonBody, err := json.Marshal(reqPayload)
	if err != nil {
		return "", err
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key=%s", apiKey)
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("chat API HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("empty answer received from Gemini")
	}

	return result.Candidates[0].Content.Parts[0].Text, nil
}

func generateCompletionWithRetry(apiKey, prompt string, maxRetries int) (string, error) {
	backoff := 500 * time.Millisecond

	for attempt := 0; attempt < maxRetries; attempt++ {
		ans, err := generateCompletion(apiKey, prompt)
		if err == nil {
			return ans, nil
		}

		log.Printf("[Attempt %d/%d] Upstream error: %v. Retrying in %v...", attempt+1, maxRetries, err, backoff)
		time.Sleep(backoff)
		backoff *= 2
	}

	return "", fmt.Errorf("exceeded max retries calling upstream model")
}

// ==========================================
// 5. Core Gateway Pipeline
// ==========================================

func HandlePrompt(cache *SemanticCache, apiKey, prompt string, threshold float32) (string, string, time.Duration, error) {
	startTime := time.Now()

	// 1. Exact Hit (L1)
	if resp, hitType, ok := cache.Get(prompt, nil, threshold); ok {
		return resp, hitType, time.Since(startTime), nil
	}

	// 2. Semantic Hit (L2)
	vec, err := getEmbedding(apiKey, prompt)
	if err == nil {
		if resp, hitType, ok := cache.Get(prompt, vec, threshold); ok {
			return resp, hitType, time.Since(startTime), nil
		}
	} else {
		log.Printf("Warning: failed to compute embedding: %v", err)
	}

	// 3. Cache Miss: Upstream Call
	answer, err := generateCompletionWithRetry(apiKey, prompt, 3)
	if err != nil {
		return "", "ERROR", time.Since(startTime), err
	}

	// 4. Save to Cache
	if vec != nil {
		cache.Set(prompt, vec, answer)
	}

	return answer, "CACHE_MISS (Fetched Upstream)", time.Since(startTime), nil
}

// ==========================================
// 6. HTTP Server Handler
// ==========================================

func handleChatCompletions(cache *SemanticCache, apiKey string, threshold float32) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
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

		answer, hitType, duration, err := HandlePrompt(cache, apiKey, req.Prompt, threshold)
		if err != nil {
			w.WriteHeader(http.StatusBadGateway)
			json.NewEncoder(w).Encode(ErrorResponse{Error: err.Error()})
			return
		}

		resp := ChatCompletionResponse{
			Response:  answer,
			HitType:   hitType,
			LatencyMs: duration.Milliseconds(),
		}

		log.Printf("[Completed] Source: %s | Latency: %dms", hitType, duration.Milliseconds())
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(resp)
	}
}

// ==========================================
// 7. Entry Point
// ==========================================

func main() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY environment variable is not set")
	}

	cache := NewSemanticCache()
	threshold := float32(0.88)
	port := ":8080"

	http.HandleFunc("/v1/chat/completions", handleChatCompletions(cache, apiKey, threshold))

	// Health check endpoint
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	})

	log.Printf("🚀 SynapseGateway listening on http://localhost%s", port)
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
