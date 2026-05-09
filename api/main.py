"""
FastAPI REST endpoint for hallucination detection.

Design principles:
- Thin API layer — evaluation logic lives in src/, not here
- Pydantic validation on all inputs
- Structured response with explainability fields
- Async processing for scale
- Reviewer escalation flag in every response
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import os
import time

from src.detector import HallucinationDetector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LLM Hallucination Detector",
    description=(
        "Claim-level hallucination detection using LangChain + "
        "Bayesian inference. Each atomic claim is independently "
        "scored — partial hallucinations that hide inside mostly-"
        "correct responses are surfaced instead of being diluted."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Detector initialization
# ---------------------------------------------------------------------------
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"
FLAG_THRESHOLD = float(os.getenv("FLAG_THRESHOLD", "0.45"))

detector = HallucinationDetector(
    use_mock=USE_MOCK,
    flag_threshold=FLAG_THRESHOLD,
)
logger.info(f"Detector initialized — mock={USE_MOCK}, threshold={FLAG_THRESHOLD}")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class DetectionRequest(BaseModel):
    response_text: str = Field(
        ...,
        description="The LLM response to evaluate for hallucinations.",
        min_length=10,
        max_length=10000,
    )
    reference_context: Optional[str] = Field(
        None,
        description="Trusted reference context to ground claims against.",
    )
    reference_answer: Optional[str] = Field(
        None,
        description="Known-correct answer for semantic comparison.",
    )
    task_id: Optional[str] = Field(
        None,
        description="Optional task identifier for tracking.",
    )


class ClaimScoreResponse(BaseModel):
    claim: str
    hallucination_probability: float
    flagged: bool
    confidence_level: str
    explanation: str


class DetectionResponse(BaseModel):
    task_id: Optional[str]
    response_flagged: bool
    risk_level: str
    flagged_claim_count: int
    total_claim_count: int
    max_hallucination_probability: float
    mean_hallucination_probability: float
    escalate_to_human: bool
    claim_scores: List[dict]
    processing_time_ms: float
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    mock_mode: bool
    flag_threshold: float
    version: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        mock_mode=USE_MOCK,
        flag_threshold=FLAG_THRESHOLD,
        version="1.0.0",
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect_hallucination(request: DetectionRequest):
    """
    Run hallucination detection on an LLM response.

    Pipeline:
    1. Extract atomic factual claims from response
    2. Ground each claim against reference context (if provided)
    3. Score semantic similarity vs reference answer (if provided)
    4. Aggregate signals using Bayesian updating
    5. Return per-claim scores + response-level summary

    Note: escalate_to_human=True means the response should be
    reviewed by a human annotator before being used downstream.
    """
    start_time = time.time()

    try:
        result = detector.detect(
            response_text=request.response_text,
            reference_context=request.reference_context,
            reference_answer=request.reference_answer,
        )
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Detection pipeline error: {str(e)}"
        )

    processing_time_ms = (time.time() - start_time) * 1000

    logger.info(
        f"task_id={request.task_id} "
        f"flagged={result.response_flagged} "
        f"risk={result.risk_level} "
        f"claims={result.total_claim_count} "
        f"time={processing_time_ms:.1f}ms"
    )

    return DetectionResponse(
        task_id=request.task_id,
        response_flagged=result.response_flagged,
        risk_level=result.risk_level,
        flagged_claim_count=result.flagged_claim_count,
        total_claim_count=result.total_claim_count,
        max_hallucination_probability=result.max_hallucination_probability,
        mean_hallucination_probability=result.mean_hallucination_probability,
        escalate_to_human=result.escalate_to_human,
        claim_scores=result.claim_scores,
        processing_time_ms=round(processing_time_ms, 2),
        metadata=result.metadata,
    )


@app.post("/detect/batch")
async def detect_batch(requests: List[DetectionRequest]):
    """
    Batch detection endpoint for pipeline use.
    Evaluates multiple responses in sequence.
    """
    results = []
    for req in requests:
        try:
            result = detector.detect(
                response_text=req.response_text,
                reference_context=req.reference_context,
                reference_answer=req.reference_answer,
            )
            results.append(
                {
                    "task_id": req.task_id,
                    "success": True,
                    "result": result.to_dict(),
                }
            )
        except Exception as e:
            results.append(
                {
                    "task_id": req.task_id,
                    "success": False,
                    "error": str(e),
                }
            )

    flagged_count = sum(
        1 for r in results if r.get("success") and r["result"]["response_flagged"]
    )

    return {
        "total": len(results),
        "flagged": flagged_count,
        "results": results,
    }
