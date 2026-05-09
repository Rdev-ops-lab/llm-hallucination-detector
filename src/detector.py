"""
LLM Hallucination Detector — Main Pipeline.

Orchestrates claim extraction, retrieval grounding,
semantic scoring, and Bayesian aggregation into a
single end-to-end hallucination detection pipeline.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

from .claim_extractor import ClaimExtractor, MockClaimExtractor
from .retrieval_grounder import RetrievalGrounder, MockRetrievalGrounder
from .bayesian_scorer import BayesianHallucinationScorer, EvidenceSignal


@dataclass
class DetectionResult:
    """Full hallucination detection result for a response."""

    response_text: str
    response_flagged: bool
    risk_level: str
    flagged_claim_count: int
    total_claim_count: int
    max_hallucination_probability: float
    mean_hallucination_probability: float
    escalate_to_human: bool
    claim_scores: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "response_flagged": self.response_flagged,
            "risk_level": self.risk_level,
            "flagged_claim_count": self.flagged_claim_count,
            "total_claim_count": self.total_claim_count,
            "max_hallucination_probability": self.max_hallucination_probability,
            "mean_hallucination_probability": self.mean_hallucination_probability,
            "escalate_to_human": self.escalate_to_human,
            "claim_scores": self.claim_scores,
            "metadata": self.metadata,
        }


class HallucinationDetector:
    """
    End-to-end hallucination detection pipeline.

    Architecture:
    1. Claim Extraction   — decompose response into atomic claims
    2. Retrieval Grounding — check each claim against reference docs
    3. Semantic Similarity — measure linguistic overlap with reference
    4. Bayesian Scoring   — aggregate signals into posterior probability

    Key design decision: response-level scoring was hiding partial
    hallucinations. Moving to claim-level evaluation dropped false
    negatives noticeably because one fabricated claim inside a
    mostly-correct response now surfaces instead of being diluted.
    """

    def __init__(
        self,
        use_mock: bool = False,
        flag_threshold: float = 0.45,
        model_name: str = "gpt-3.5-turbo",
    ):
        """
        Args:
            use_mock: Use mock extractors (no API key needed, for testing).
            flag_threshold: Hallucination probability threshold for flagging.
            model_name: OpenAI model for extraction and grounding.
        """
        self.use_mock = use_mock
        self.flag_threshold = flag_threshold

        if use_mock:
            self.extractor = MockClaimExtractor()
            self.grounder = MockRetrievalGrounder()
            self.embeddings = None
        else:
            self.extractor = ClaimExtractor(model_name=model_name)
            self.grounder = RetrievalGrounder(model_name=model_name)
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )

        self.scorer = BayesianHallucinationScorer(flag_threshold=flag_threshold)

    def load_reference_documents(self, documents: List[str]) -> None:
        """
        Load reference documents into retrieval index.

        Args:
            documents: List of trusted reference text strings.
        """
        self.grounder.build_index(documents)

    def detect(
        self,
        response_text: str,
        reference_context: Optional[str] = None,
        reference_answer: Optional[str] = None,
    ) -> DetectionResult:
        """
        Run full hallucination detection on a response.

        Args:
            response_text: The LLM response to evaluate.
            reference_context: Trusted context to ground claims against.
            reference_answer: Known-correct answer for semantic comparison.

        Returns:
            DetectionResult with per-claim scores and response summary.
        """
        # Step 1: Extract atomic claims
        claims = self.extractor.extract(response_text)

        if not claims:
            return self._empty_result(response_text)

        # Step 2: Build evidence signals per claim
        claims_with_signals = []

        for claim in claims:
            signals = []

            # Signal 1: Retrieval grounding
            retrieval_score = self.grounder.ground_claim(claim, reference_context)
            signals.append(
                EvidenceSignal(
                    name="retrieval_grounding",
                    support_score=retrieval_score,
                    weight=2.0,  # High weight — most reliable signal
                    confidence=0.9,
                )
            )

            # Signal 2: Semantic similarity vs reference answer
            if reference_answer and not self.use_mock:
                sem_score = self._semantic_similarity(claim, reference_answer)
                signals.append(
                    EvidenceSignal(
                        name="semantic_similarity",
                        support_score=sem_score,
                        weight=1.0,
                        confidence=0.7,  # Lower confidence — can hide fabrication
                    )
                )
            elif self.use_mock:
                # Mock semantic signal for testing
                signals.append(
                    EvidenceSignal(
                        name="semantic_similarity",
                        support_score=0.75,
                        weight=1.0,
                        confidence=0.7,
                    )
                )

            # Signal 3: Uncertainty marker detection (rule-based)
            uncertainty_score = self._check_uncertainty_markers(claim)
            signals.append(
                EvidenceSignal(
                    name="uncertainty_markers",
                    support_score=uncertainty_score,
                    weight=0.5,
                    confidence=0.8,
                )
            )

            claims_with_signals.append((claim, signals))

        # Step 3: Bayesian scoring
        result = self.scorer.score_response(claims_with_signals)

        return DetectionResult(
            response_text=response_text,
            response_flagged=result["response_flagged"],
            risk_level=result["risk_level"],
            flagged_claim_count=result["flagged_claim_count"],
            total_claim_count=result["total_claim_count"],
            max_hallucination_probability=result["max_hallucination_probability"],
            mean_hallucination_probability=result["mean_hallucination_probability"],
            escalate_to_human=result["escalate_to_human"],
            claim_scores=result["claim_scores"],
            metadata={
                "flag_threshold": self.flag_threshold,
                "claims_extracted": len(claims),
                "use_mock": self.use_mock,
            },
        )

    def _semantic_similarity(self, claim: str, reference: str) -> float:
        """Compute cosine similarity between claim and reference embeddings."""
        try:
            embeddings = self.embeddings.embed_documents([claim, reference])
            sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _check_uncertainty_markers(claim: str) -> float:
        """
        Rule-based check for overconfidence markers.

        Claims with hedging language get higher support scores
        (appropriate uncertainty = less likely to be hallucinated).
        Claims stated with false certainty on weak ground get
        lower support scores.
        """
        claim_lower = claim.lower()

        # Overconfidence markers — no hedging on potentially weak claims
        overconfidence_markers = [
            "always",
            "never",
            "100%",
            "proven",
            "definitely",
            "certainly",
            "it is a fact",
            "studies show",
            "research shows",
            "scientists say",
            "experts agree",
        ]

        # Appropriate hedging markers
        hedging_markers = [
            "may",
            "might",
            "could",
            "possibly",
            "approximately",
            "around",
            "roughly",
            "according to",
            "suggests",
            "indicates",
        ]

        overconfidence_count = sum(
            1 for m in overconfidence_markers if m in claim_lower
        )
        hedging_count = sum(1 for m in hedging_markers if m in claim_lower)

        # Short neutral claims get neutral score
        if not overconfidence_count and not hedging_count:
            return 0.6

        # Appropriate hedging = supportive signal
        if hedging_count > 0 and overconfidence_count == 0:
            return 0.8

        # Overconfidence without hedging = slight negative signal
        if overconfidence_count > 0:
            return max(0.2, 0.6 - (overconfidence_count * 0.15))

        return 0.6

    @staticmethod
    def _empty_result(response_text: str) -> DetectionResult:
        return DetectionResult(
            response_text=response_text,
            response_flagged=False,
            risk_level="LOW",
            flagged_claim_count=0,
            total_claim_count=0,
            max_hallucination_probability=0.0,
            mean_hallucination_probability=0.0,
            escalate_to_human=False,
            claim_scores=[],
            metadata={"note": "No claims extracted from response."},
        )
