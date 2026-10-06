package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
)

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

func HashPrompt(prompt string) string {
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
	hash := HashPrompt(prompt)
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

	hash := HashPrompt(prompt)
	c.exact[hash] = response
	c.entries = append(c.entries, CacheEntry{
		Prompt:    prompt,
		Embedding: vec,
		Response:  response,
	})
}
