package main

import (
	"log"
	"net/http"
	"os"

	"synapse-gateway/internal/cache"
	"synapse-gateway/internal/client"
	"synapse-gateway/internal/proxy"
)

func main() {
	geminiKey := os.Getenv("GEMINI_API_KEY")
	if geminiKey == "" {
		log.Fatal("GEMINI_API_KEY environment variable is not set")
	}

	groqKey := os.Getenv("GROQ_API_KEY") // Optional fallback key

	// Initialize dependencies
	semanticCache := cache.NewSemanticCache()
	geminiClient := client.NewGeminiClient(geminiKey)
	groqClient := client.NewGroqClient(groqKey)

	// Notice all 4 arguments passed here:
	gatewayHandler := proxy.NewGatewayHandler(semanticCache, geminiClient, groqClient, 0.88)

	// Routes
	http.Handle("/v1/chat/completions", gatewayHandler)
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	})

	port := ":8080"
	log.Printf("🚀 SynapseGateway listening on http://localhost%s", port)
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
