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
)

// --- 1. Structs for Google Gemini Embedding API ---

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

// --- 2. In-Memory Cache Structures ---

// CacheEntry holds the prompt, its 768-dim embedding, and the cached answer.
type CacheEntry struct {
	Prompt    string
	Embedding []float32
	Response  string
}

// SemanticCache manages exact string lookups and vector searches.
type SemanticCache struct {
	mu      sync.RWMutex
	exact   map[string]string // SHA-256 hash -> Response
	entries []CacheEntry      // Stored vectors for semantic scan
}

func NewSemanticCache() *SemanticCache {
	return &SemanticCache{
		exact:   make(map[string]string),
		entries: make([]CacheEntry, 0),
	}
}

// hashPrompt turns a prompt into a SHA-256 string for O(1) exact lookups.
func hashPrompt(prompt string) string {
	h := sha256.Sum256([]byte(prompt))
	return hex.EncodeToString(h[:])
}

// CosineSimilarity computes the dot product of two unit-normalized vectors.
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

// Get checks exact cache first, then scans vectors for similarity.
func (c *SemanticCache) Get(prompt string, vec []float32, threshold float32) (string, string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Tier 1: Exact Hash Match (O(1))
	hash := hashPrompt(prompt)
	if resp, exists := c.exact[hash]; exists {
		return resp, "EXACT_HIT", true
	}

	// Tier 2: Vector Semantic Search (Linear Scan)
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

// Set stores the prompt, its embedding, and the response.
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

// --- 3. Gemini Helper ---

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
		return nil, fmt.Errorf("API error HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.Embedding.Values, nil
}

// --- 4. Main Demonstration ---

func main() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY environment variable is not set")
	}

	cache := NewSemanticCache()
	similarityThreshold := float32(0.88)

	// Step A: Seed the cache with an initial answered question
	seedPrompt := "How do I cancel my subscription?"
	seedAnswer := "Go to Account Settings -> Billing -> Click 'Cancel Subscription'."

	fmt.Println("1. Embedding and seeding original Q&A into cache...")
	seedVec, err := getEmbedding(apiKey, seedPrompt)
	if err != nil {
		log.Fatalf("Failed to embed seed prompt: %v", err)
	}
	cache.Set(seedPrompt, seedVec, seedAnswer)
	fmt.Println("   Seed prompt stored successfully!")

	// Step B: Query with an exact match
	test1 := "How do I cancel my subscription?"
	fmt.Printf("2. Querying exact match: %q\n", test1)
	resp, hitType, hit := cache.Get(test1, nil, similarityThreshold)
	fmt.Printf("   Result: %s | Hit: %t\n   Answer: %s\n\n", hitType, hit, resp)

	// Step C: Query with a paraphrase (semantic match)
	test2 := "Where can I terminate my plan?"
	fmt.Printf("3. Querying paraphrase: %q\n", test2)
	test2Vec, err := getEmbedding(apiKey, test2)
	if err != nil {
		log.Fatalf("Failed to embed test2 prompt: %v", err)
	}
	resp, hitType, hit = cache.Get(test2, test2Vec, similarityThreshold)
	fmt.Printf("   Result: %s | Hit: %t\n   Answer: %s\n\n", hitType, hit, resp)

	// Step D: Query with an unrelated question
	test3 := "How do I bake chocolate chip cookies?"
	fmt.Printf("4. Querying unrelated question: %q\n", test3)
	test3Vec, err := getEmbedding(apiKey, test3)
	if err != nil {
		log.Fatalf("Failed to embed test3 prompt: %v", err)
	}
	resp, hitType, hit = cache.Get(test3, test3Vec, similarityThreshold)
	fmt.Printf("   Result: %s | Hit: %t\n   Answer: (none, forwarded to LLM)\n", hitType, hit)
}
