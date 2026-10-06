# ⚡ SynapseGateway

A high-performance, resilient AI gateway and reverse proxy built in Go. **SynapseGateway** sits between client applications and Large Language Model (LLM) providers to eliminate redundant upstream calls, cut API costs, and lower response latency using a thread-safe two-tier cache and automated multi-provider failover.

---

## 🌟 Architecture & Features

```text
Client (curl / SDK / App)
          │
          ▼  POST /v1/chat/completions
┌─────────────────────────────────────────────────────────────┐
│                       SynapseGateway                        │
│                                                             │
│  [ Tier 1: L1 Exact Match Cache ]                           │
│  - SHA-256 string hash lookup                               │
│  - O(1) in-memory lookup via sync.RWMutex                   │
│  - Latency: < 1ms                                           │
│          │ (Miss)                                           │
│          ▼                                                  │
│  [ Tier 2: L2 Semantic Cache ]                              │
│  - Vector embeddings (gemini-embedding-001)                 │
│  - Cosine Similarity comparison (threshold: 0.88)           │
│  - Latency: ~100-200ms                                      │
│          │ (Miss)                                           │
│          ▼                                                  │
│  [ Upstream LLM Execution & Failover ]                      │
│  - Primary: Google Gemini 3.8 Flash (with backoff retry)    │
│  - Fallback: Groq (openai/gpt-oss-20b)                      │
│  - Network Safety: 8s HTTP timeout to prevent hangs         │
│  - Latency: ~1500-3000ms                                    │
└─────────────────────────────────────────────────────────────┘
```

- **Two-Tier Caching Strategy**:
  - **L1 Exact Cache**: Instant $O(1)$ response via SHA-256 key hashing without external network calls.
  - **L2 Semantic Cache**: Evaluates query semantics using 768-dimensional vector embeddings and Cosine Similarity. Answers paraphrased questions directly from RAM.
- **Automated Provider Failover**: If Gemini hits rate limits, timeouts, or HTTP 503 spikes, requests automatically divert to Groq without dropping the client connection.
- **Thread-Safe Concurrency**: Uses Go's `sync.RWMutex` to guarantee safe concurrent reads and exclusive writes across parallel goroutines.
- **Clean Architecture**: Modular structure strictly separated into `cmd/server`, `internal/cache`, `internal/client`, and `internal/proxy`.

---

## 📂 Project Structure

```text
synapse-gateway/
├── cmd/
│   └── server/
│       └── main.go          # Application entrypoint & dependency injection
├── internal/
│   ├── cache/
│   │   └── cache.go         # L1 hash map, L2 vector store & Cosine Similarity math
│   ├── client/
│   │   ├── gemini.go        # Gemini Embeddings & Chat Completion (retry + timeout)
│   │   └── groq.go          # Groq OpenAI-compatible fallback client
│   └── proxy/
│       └── handler.go       # HTTP request/response DTOs & gateway pipeline handler
├── go.mod
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- [Go 1.22+](https://golang.org/dl/)
- Google Gemini API Key
- Groq API Key (for failover)

### 1. Clone the Repository

```bash
git clone https://github.com/px444/synapse-gateway.git
cd synapse-gateway
```

### 2. Configure Environment Variables

**PowerShell (Windows):**
```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
$env:GROQ_API_KEY="your_groq_api_key_here"
```

**Bash (Linux / macOS):**
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export GROQ_API_KEY="your_groq_api_key_here"
```

### 3. Run the Server

```bash
go run ./cmd/server
```

---

## 🧪 Testing the Endpoints

Send a request using PowerShell:

```powershell
Invoke-RestMethod -Uri "http://localhost:8080/v1/chat/completions" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"prompt": "What is the capital of France?"}'
```

### Performance & Routing in Action

| Request Type | Query | Source / Status | Observed Latency |
| :--- | :--- | :--- | :--- |
| **Cold Start** | `"What is the capital of France?"` | `CACHE_MISS (Gemini Primary)` | ~2400 ms |
| **Exact Match** | `"What is the capital of France?"` | `EXACT_HIT` | **0 ms (< 1ms)** |
| **Paraphrased** | `"Can you tell me the capital city of France?"` | `SEMANTIC_HIT (score: 0.912)` | ~180 ms |
| **Primary Down** | *(Any uncached prompt)* | `CACHE_MISS (Groq Failover)` | Seamless fallback |

---

## 📋 Completed Architecture

- [x] L1 SHA-256 Exact Hash Cache
- [x] L2 768-dim Vector Cosine Similarity Semantic Cache
- [x] Concurrency safety with `sync.RWMutex`
- [x] Upstream API integration with exponential backoff and timeouts
- [x] Multi-provider failover fallback (Gemini → Groq)
- [x] Standard `net/http` reverse proxy server
- [x] Modular Go package refactor (`cmd/`, `internal/`)