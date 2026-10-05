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
// 1. Google Gemini API Structs
// ==========================================

// For getting embedding vectors (768 numbers)
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

// For getting generated answers from Gemini
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
// 2. In-Memory Cache Structures
// ==========================================

type CacheEntry struct {
	Prompt    string
	Embedding []float32
	Response  string
}

type SemanticCache struct {
	mu      sync.RWMutex
	exact   map[string]string // SHA-256 hash -> Response
	entries []CacheEntry      // Vector list
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

	// 1. Exact match (L1)
	hash := hashPrompt(prompt)
	if resp, exists := c.exact[hash]; exists {
		return resp, "EXACT_HIT", true
	}

	// 2. Semantic match (L2)
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
// 3. Gemini API Calls
// ==========================================

// Get 768 coordinate numbers representing the sentence
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

// Ask Gemini for an answer when cache misses
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

	// Updated to gemini-3.8-flash as required by the API
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

		// If it's a 503 (high demand), sleep briefly and try again
		log.Printf("[Attempt %d/%d] Model busy (503). Retrying in %v...", attempt+1, maxRetries, backoff)
		time.Sleep(backoff)
		backoff *= 2 // Exponential backoff: 500ms -> 1s -> 2s
	}

	return "", fmt.Errorf("exceeded max retries due to high demand")
}

// ==========================================
// 4. Gateway Dispatcher
// ==========================================

func HandlePrompt(cache *SemanticCache, apiKey, prompt string, threshold float32) (string, string, time.Duration) {
	startTime := time.Now()

	// 1. Exact Hit (L1)
	if resp, hitType, ok := cache.Get(prompt, nil, threshold); ok {
		return resp, hitType, time.Since(startTime)
	}

	// 2. Semantic Hit (L2)
	vec, err := getEmbedding(apiKey, prompt)
	if err == nil {
		if resp, hitType, ok := cache.Get(prompt, vec, threshold); ok {
			return resp, hitType, time.Since(startTime)
		}
	}

	// 3. Cache Miss: Ask Gemini
	answer, err := generateCompletionWithRetry(apiKey, prompt, 3)
	if err != nil {
		return fmt.Sprintf("Error: %v", err), "ERROR", time.Since(startTime)
	}

	// 4. Save to cache for next time
	if vec != nil {
		cache.Set(prompt, vec, answer)
	}

	return answer, "CACHE_MISS (Fetched from Gemini)", time.Since(startTime)
}

// ==========================================
// 5. Main Execution
// ==========================================

func main() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY environment variable is not set")
	}

	cache := NewSemanticCache()
	threshold := float32(0.88)

	// Test 1: Cold Start (Cache miss -> calls Gemini)
	q1 := "What is the capital of France?"
	fmt.Printf("[Test 1: Cold Start] Prompt: %q\n", q1)
	ans1, status1, dur1 := HandlePrompt(cache, apiKey, q1, threshold)
	fmt.Printf("Status: %s\nLatency: %v\nAnswer: %s\n\n", status1, dur1, ans1)

	// Test 2: Exact Match (L1 hit in microseconds -> 0 API calls)
	q2 := "What is the capital of France?"
	fmt.Printf("[Test 2: Exact Match] Prompt: %q\n", q2)
	ans2, status2, dur2 := HandlePrompt(cache, apiKey, q2, threshold)
	fmt.Printf("Status: %s\nLatency: %v\nAnswer: %s\n\n", status2, dur2, ans2)

	// Test 3: Paraphrase (L2 Semantic hit -> gets cached answer without asking LLM)
	q3 := "Can you tell me the capital city of France?"
	fmt.Printf("[Test 3: Paraphrase] Prompt: %q\n", q3)
	ans3, status3, dur3 := HandlePrompt(cache, apiKey, q3, threshold)
	fmt.Printf("Status: %s\nLatency: %v\nAnswer: %s\n", status3, dur3, ans3)
}
