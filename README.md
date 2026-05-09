# LLM Hallucination Detector

Claim-level hallucination detection pipeline using **LangChain + Bayesian inference**.

Built from real evaluation work at scale — the core insight is that **response-level scoring hides partial hallucinations**. A response that is 90% accurate with one fabricated statistic will score well on semantic similarity, but the fabricated claim will pass undetected. This system solves that by decomposing responses into atomic claims and scoring each independently.

---

## Architecture

```
Response Text
     │
     ▼
┌─────────────────────┐
│   Claim Extractor   │  ← LangChain + GPT
│  (atomic claims)    │
└────────┬────────────┘
         │ claims[]
         ▼
┌─────────────────────────────────────────────┐
│              Per-Claim Signals              │
│                                             │
│  ┌──────────────────┐  weight: 2.0          │
│  │ Retrieval Ground │  ← LangChain + FAISS  │
│  └──────────────────┘                       │
│  ┌──────────────────┐  weight: 1.0          │
│  │ Semantic Simil.  │  ← OpenAI Embeddings  │
│  └──────────────────┘                       │
│  ┌──────────────────┐  weight: 0.5          │
│  │ Uncertainty Mrk. │  ← Rule-based         │
│  └──────────────────┘                       │
└────────┬────────────────────────────────────┘
         │ signals[]
         ▼
┌─────────────────────┐
│   Bayesian Scorer   │  ← Beta distribution
│  (posterior update) │     independent signals
└────────┬────────────┘
         │
         ▼
   Per-claim score + Response summary + Escalation flag
```

**Why Bayesian instead of averaging?**

Simple averaging hides the problem. If semantic similarity is 0.88 but retrieval grounding is 0.05, an average gives 0.46 — looks borderline. Bayesian updating treats each signal as independent evidence: the low retrieval score raises the posterior hallucination probability on its own, even when other signals look good.

---

## Quickstart

### Run with Docker (recommended)

```bash
# Clone the repo
git clone https://github.com/Rdev-ops-lab/llm-hallucination-detector
cd llm-hallucination-detector

# Copy env file
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Start the API
docker-compose up hallucination-detector

# Or run in mock mode (no API key needed)
docker-compose --profile mock up hallucination-detector-mock
```

### Run locally

```bash
pip install -r requirements.txt

# Full mode
OPENAI_API_KEY=your_key uvicorn api.main:app --reload

# Mock mode (no API key)
USE_MOCK=true uvicorn api.main:app --reload
```

---

## API Usage

### Detect hallucinations in a response

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "response_text": "Studies show 94% of Fortune 500 companies use Python as their primary language.",
    "reference_context": "Python is widely used in industry. Exact adoption statistics vary.",
    "task_id": "eval-001"
  }'
```

**Response:**

```json
{
  "task_id": "eval-001",
  "response_flagged": true,
  "risk_level": "HIGH",
  "flagged_claim_count": 1,
  "total_claim_count": 1,
  "max_hallucination_probability": 0.7812,
  "mean_hallucination_probability": 0.7812,
  "escalate_to_human": true,
  "claim_scores": [
    {
      "claim": "94% of Fortune 500 companies use Python as their primary language",
      "hallucination_probability": 0.7812,
      "flagged": true,
      "confidence_level": "HIGH",
      "explanation": "Flagged as likely hallucination (probability: 78.1%). Weak support from: retrieval_grounding, semantic_similarity.",
      "signals": [
        {"name": "retrieval_grounding", "support_score": 0.04, "weight": 2.0},
        {"name": "semantic_similarity",  "support_score": 0.12, "weight": 1.0},
        {"name": "uncertainty_markers",  "support_score": 0.45, "weight": 0.5}
      ]
    }
  ],
  "processing_time_ms": 1842.3
}
```

### Batch evaluation

```bash
curl -X POST http://localhost:8000/detect/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"response_text": "...", "task_id": "001"},
    {"response_text": "...", "task_id": "002"}
  ]'
```

### Health check

```bash
curl http://localhost:8000/health
```

---

## Run Tests

```bash
# All tests in mock mode (no API key needed)
USE_MOCK=true pytest tests/ -v

# With coverage
USE_MOCK=true pytest tests/ --cov=src --cov-report=term-missing
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (required for full mode) |
| `USE_MOCK` | `false` | Run without API calls (testing) |
| `FLAG_THRESHOLD` | `0.45` | Posterior probability above which claims are flagged |
| `PORT` | `8000` | API server port |

---

## Key Design Decisions

**Claim-level over response-level** — Response-level scoring hides partial hallucinations. One fabricated claim inside a correct response gets diluted. Claim decomposition surfaces the specific sentence that was hallucinated, which also makes reviewer escalation faster — they know exactly where to look.

**Bayesian aggregation over averaging** — Each signal updates the posterior independently. High semantic similarity cannot cancel a low retrieval grounding score. The worst signal drives suspicion, not the average.

**escalate_to_human flag** — Automation is a filtering layer, not a final judge. Ambiguous or high-probability cases are routed to human review automatically.

**Mock mode** — Full test coverage without API keys. CI/CD runs entirely in mock mode. Real integration tests run separately with a dedicated API key.

---

## Project Structure

```
llm-hallucination-detector/
├── src/
│   ├── detector.py           # Main pipeline
│   ├── claim_extractor.py    # LangChain claim decomposition
│   ├── retrieval_grounder.py # LangChain + FAISS retrieval
│   └── bayesian_scorer.py    # Beta distribution scoring
├── api/
│   └── main.py               # FastAPI endpoints
├── tests/
│   ├── test_bayesian_scorer.py
│   └── test_detector.py
├── examples/
│   └── sample_usage.py
├── .github/workflows/ci.yml  # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Related Projects

- [Automated QA Pipeline](https://github.com/Rdev-ops-lab/automated-qa-pipeline) — Docker + GitHub Actions pipeline for large-scale LLM evaluation
- [PhD-Level STEM Reasoning Framework](https://github.com/Rdev-ops-lab/stem-reasoning-framework) — Symbolic deduction engine for frontier model evaluation

---

## Author

**Rishi Pal Singh** — AI Evaluation Specialist  
[LinkedIn](https://linkedin.com/in/rishi-singh-1413b3384) · [GitHub](https://github.com/Rdev-ops-lab)
