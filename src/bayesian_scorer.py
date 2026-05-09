"""
Bayesian Confidence Scorer for Hallucination Detection.

Uses Beta distribution to aggregate multiple evidence signals
independently, updating posterior probability of hallucination
for each claim.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class EvidenceSignal:
    """Single evidence signal for a claim."""
    name: str
    support_score: float      # 0.0 = no support, 1.0 = full support
    confidence: float = 1.0   # how much to trust this signal
    weight: float = 1.0       # relative importance of this signal


@dataclass
class ClaimScore:
    """Bayesian scoring result for a single claim."""
    claim: str
    hallucination_probability: float
    signals: List[EvidenceSignal]
    flagged: bool
    confidence_level: str     # HIGH / MEDIUM / LOW
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "hallucination_probability": round(self.hallucination_probability, 4),
            "flagged": self.flagged,
            "confidence_level": self.confidence_level,
            "explanation": self.explanation,
            "signals": [
                {
                    "name": s.name,
                    "support_score": round(s.support_score, 4),
                    "weight": s.weight
                }
                for s in self.signals
            ]
        }


class BayesianHallucinationScorer:
    """
    Aggregates multiple evidence signals using Bayesian updating.

    Instead of averaging signals (which hides partial hallucinations),
    each signal independently updates a Beta distribution posterior.
    A claim with high semantic similarity but low retrieval support
    will still be flagged — the weak signal raises suspicion even
    when other signals look good.
    """

    def __init__(
        self,
        flag_threshold: float = 0.45,
        prior_alpha: float = 1.0,
        prior_beta: float = 2.0,
    ):
        """
        Args:
            flag_threshold: Posterior hallucination probability above
                            which a claim is flagged.
            prior_alpha: Beta prior alpha (support evidence count).
            prior_beta: Beta prior beta (against evidence count).
                        Default prior is slightly skeptical — Beta(1,2).
        """
        self.flag_threshold = flag_threshold
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def score_claim(
        self,
        claim: str,
        signals: List[EvidenceSignal]
    ) -> ClaimScore:
        """
        Score a single claim given evidence signals.

        Bayesian updating logic:
        - Start with prior Beta(alpha, beta)
        - Each signal contributes weighted evidence
        - High support_score increases alpha (evidence for truth)
        - Low support_score increases beta (evidence for hallucination)
        - Final posterior mean = alpha / (alpha + beta)
        - Hallucination probability = 1 - posterior_mean
        """
        alpha = self.prior_alpha
        beta = self.prior_beta

        for signal in signals:
            effective_weight = signal.weight * signal.confidence
            support = signal.support_score
            against = 1.0 - signal.support_score

            alpha += effective_weight * support
            beta += effective_weight * against

        # Posterior mean of Beta distribution = alpha / (alpha + beta)
        truth_probability = alpha / (alpha + beta)
        hallucination_probability = 1.0 - truth_probability

        flagged = hallucination_probability >= self.flag_threshold

        # Confidence level based on total evidence weight
        total_evidence = sum(s.weight * s.confidence for s in signals)
        if total_evidence >= 4.0:
            confidence_level = "HIGH"
        elif total_evidence >= 2.0:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        explanation = self._build_explanation(
            signals, hallucination_probability, flagged
        )

        return ClaimScore(
            claim=claim,
            hallucination_probability=hallucination_probability,
            signals=signals,
            flagged=flagged,
            confidence_level=confidence_level,
            explanation=explanation,
        )

    def _build_explanation(
        self,
        signals: List[EvidenceSignal],
        prob: float,
        flagged: bool
    ) -> str:
        if not signals:
            return "No evidence signals provided."

        weak_signals = [s for s in signals if s.support_score < 0.4]
        strong_signals = [s for s in signals if s.support_score >= 0.7]

        parts = []
        if flagged:
            parts.append(
                f"Flagged as likely hallucination "
                f"(probability: {prob:.1%})."
            )
        else:
            parts.append(
                f"Claim appears grounded "
                f"(hallucination probability: {prob:.1%})."
            )

        if weak_signals:
            names = ", ".join(s.name for s in weak_signals)
            parts.append(f"Weak support from: {names}.")

        if strong_signals:
            names = ", ".join(s.name for s in strong_signals)
            parts.append(f"Strong support from: {names}.")

        return " ".join(parts)

    def score_response(
        self,
        claims_with_signals: List[tuple]
    ) -> dict:
        """
        Score all claims in a response.

        Args:
            claims_with_signals: List of (claim_text, List[EvidenceSignal])

        Returns:
            Aggregated result with per-claim scores and response-level summary.
        """
        claim_scores = []
        for claim, signals in claims_with_signals:
            score = self.score_claim(claim, signals)
            claim_scores.append(score)

        flagged_claims = [s for s in claim_scores if s.flagged]
        hallucination_probs = [s.hallucination_probability for s in claim_scores]

        # Response-level risk: max probability drives overall assessment
        # (don't average — one bad claim should surface)
        max_prob = max(hallucination_probs) if hallucination_probs else 0.0
        mean_prob = np.mean(hallucination_probs) if hallucination_probs else 0.0

        response_flagged = len(flagged_claims) > 0

        return {
            "response_flagged": response_flagged,
            "flagged_claim_count": len(flagged_claims),
            "total_claim_count": len(claim_scores),
            "max_hallucination_probability": round(float(max_prob), 4),
            "mean_hallucination_probability": round(float(mean_prob), 4),
            "risk_level": self._risk_level(max_prob),
            "claim_scores": [s.to_dict() for s in claim_scores],
            "escalate_to_human": response_flagged or max_prob > 0.6,
        }

    @staticmethod
    def _risk_level(max_prob: float) -> str:
        if max_prob >= 0.7:
            return "HIGH"
        elif max_prob >= 0.45:
            return "MEDIUM"
        else:
            return "LOW"
