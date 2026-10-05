package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
)

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

// getEmbedding sends a sentence to Google Gemini and returns its 768-number vector.
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

// CosineSimilarity calculates the dot product between two unit-normalized vectors.
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

func main() {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY environment variable is not set")
	}

	s1 := "How do I cancel my plan?"
	s2 := "Where can I terminate my plan?"
	s3 := "How do I bake chocolate chip cookies?" // Unrelated comparison

	fmt.Println("Fetching embeddings from Gemini...")

	vec1, err := getEmbedding(apiKey, s1)
	if err != nil {
		log.Fatalf("Error embedding s1: %v", err)
	}

	vec2, err := getEmbedding(apiKey, s2)
	if err != nil {
		log.Fatalf("Error embedding s2: %v", err)
	}

	vec3, err := getEmbedding(apiKey, s3)
	if err != nil {
		log.Fatalf("Error embedding s3: %v", err)
	}

	// Calculate similarities
	scoreSimilar := CosineSimilarity(vec1, vec2)
	scoreUnrelated := CosineSimilarity(vec1, vec3)

	fmt.Printf("\nSentence 1: %q\n", s1)
	fmt.Printf("Sentence 2: %q\n", s2)
	fmt.Printf("Similarity Score: %.4f\n", scoreSimilar)

	fmt.Println("\n--------------------------------------------------")
	fmt.Printf("Sentence 1: %q\n", s1)
	fmt.Printf("Sentence 3: %q\n", s3)
	fmt.Printf("Similarity Score: %.4f\n", scoreUnrelated)
}
