"""
Tests for Bayesian Hallucination Scorer.
"""

import pytest
from src.bayesian_scorer import (
    BayesianHallucinationScorer,
    EvidenceSignal,
    ClaimScore,
)


@pytest.fixture
def scorer():
    return BayesianHallucinationScorer(flag_threshold=0.45)


class TestEvidenceSignal:
    def test_signal_creation(self):
        signal = EvidenceSignal(
            name="retrieval_grounding",
            support_score=0.8,
            weight=2.0,
        )
        assert signal.name == "retrieval_grounding"
        assert signal.support_score == 0.8
        assert signal.weight == 2.0

    def test_signal_defaults(self):
        signal = EvidenceSignal(name="test", support_score=0.5)
        assert signal.confidence == 1.0
        assert signal.weight == 1.0


class TestBayesianScorer:
    def test_strongly_supported_claim_not_flagged(self, scorer):
        signals = [
            EvidenceSignal("retrieval", support_score=0.95, weight=2.0),
            EvidenceSignal("semantic", support_score=0.90, weight=1.0),
        ]
        result = scorer.score_claim("The capital of France is Paris.", signals)
        assert result.hallucination_probability < 0.45
        assert result.flagged is False

    def test_unsupported_claim_flagged(self, scorer):
        signals = [
            EvidenceSignal("retrieval", support_score=0.05, weight=2.0),
            EvidenceSignal("semantic", support_score=0.10, weight=1.0),
        ]
        result = scorer.score_claim("The study showed 99% efficacy.", signals)
        assert result.hallucination_probability >= 0.45
        assert result.flagged is True

    def test_conflicting_signals_surface_hallucination(self, scorer):
        """
        Key test: high semantic similarity + low retrieval support.
        A simple average would hide the hallucination.
        Bayesian updating should still raise probability.
        """
        signals = [
            EvidenceSignal("semantic_similarity", support_score=0.88, weight=1.0),
            EvidenceSignal("retrieval_grounding", support_score=0.05, weight=2.0),
        ]
        result = scorer.score_claim(
            "Studies show this drug has 87% efficacy rate.", signals
        )
        # Despite high semantic similarity, low retrieval should flag it
        assert result.hallucination_probability > 0.35
        assert result.explanation != ""

    def test_response_level_scoring(self, scorer):
        claims_with_signals = [
            ("The sky is blue.", [
                EvidenceSignal("retrieval", support_score=0.95, weight=2.0),
            ]),
            ("Studies prove 100% cure rate.", [
                EvidenceSignal("retrieval", support_score=0.02, weight=2.0),
                EvidenceSignal("semantic", support_score=0.15, weight=1.0),
            ]),
        ]
        result = scorer.score_response(claims_with_signals)
        assert result["response_flagged"] is True
        assert result["flagged_claim_count"] == 1
        assert result["total_claim_count"] == 2
        assert result["escalate_to_human"] is True

    def test_max_probability_drives_risk_not_mean(self, scorer):
        """
        One bad claim inside mostly-correct response
        should still surface — not get diluted by averages.
        """
        claims_with_signals = [
            (f"Correct claim {i}", [
                EvidenceSignal("retrieval", support_score=0.92, weight=2.0),
            ])
            for i in range(9)
        ]
        # Add one highly hallucinated claim
        claims_with_signals.append(("Fabricated statistic.", [
            EvidenceSignal("retrieval", support_score=0.01, weight=2.0),
        ]))

        result = scorer.score_response(claims_with_signals)
        assert result["response_flagged"] is True
        assert result["max_hallucination_probability"] > result["mean_hallucination_probability"]

    def test_confidence_levels(self, scorer):
        low_evidence = [EvidenceSignal("sig", support_score=0.3, weight=0.5)]
        high_evidence = [
            EvidenceSignal("sig1", support_score=0.3, weight=2.0),
            EvidenceSignal("sig2", support_score=0.3, weight=2.0),
        ]
        low_result = scorer.score_claim("claim", low_evidence)
        high_result = scorer.score_claim("claim", high_evidence)
        assert low_result.confidence_level == "LOW"
        assert high_result.confidence_level == "HIGH"

    def test_empty_signals_returns_prior(self, scorer):
        result = scorer.score_claim("A claim with no evidence.", [])
        # With Beta(1,2) prior, truth_probability = 1/3, hallucination = 2/3
        assert 0.6 < result.hallucination_probability < 0.7

    def test_risk_levels(self, scorer):
        assert scorer._risk_level(0.75) == "HIGH"
        assert scorer._risk_level(0.50) == "MEDIUM"
        assert scorer._risk_level(0.20) == "LOW"
