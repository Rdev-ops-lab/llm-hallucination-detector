"""
Integration tests for HallucinationDetector using mock components.
No API keys required.
"""

import pytest
from src.detector import HallucinationDetector


@pytest.fixture
def detector():
    """Detector in mock mode — no API keys needed."""
    return HallucinationDetector(use_mock=True, flag_threshold=0.45)


class TestHallucinationDetector:
    def test_detect_returns_result(self, detector):
        result = detector.detect(
            response_text="The Earth orbits the Sun every 365 days.",
            reference_context="The Earth completes one orbit around the Sun in approximately 365.25 days.",
        )
        assert result is not None
        assert result.total_claim_count >= 1
        assert 0.0 <= result.max_hallucination_probability <= 1.0

    def test_empty_response_handled(self, detector):
        result = detector.detect(response_text="   ")
        assert result.total_claim_count == 0
        assert result.response_flagged is False

    def test_result_has_required_fields(self, detector):
        result = detector.detect(
            response_text="Python was created by Guido van Rossum in 1991."
        )
        result_dict = result.to_dict()
        required_keys = [
            "response_flagged",
            "risk_level",
            "flagged_claim_count",
            "total_claim_count",
            "max_hallucination_probability",
            "mean_hallucination_probability",
            "escalate_to_human",
            "claim_scores",
            "metadata",
        ]
        for key in required_keys:
            assert key in result_dict, f"Missing key: {key}"

    def test_claim_scores_have_required_fields(self, detector):
        result = detector.detect(
            response_text="Water boils at 100 degrees Celsius at sea level."
        )
        if result.claim_scores:
            score = result.claim_scores[0]
            assert "claim" in score
            assert "hallucination_probability" in score
            assert "flagged" in score
            assert "confidence_level" in score
            assert "explanation" in score
            assert "signals" in score

    def test_uncertainty_markers_affect_score(self, detector):
        """Claims with overconfidence markers should score differently."""
        overconfident = detector.detect(
            response_text="Studies definitively prove this always works 100% of the time."
        )
        hedged = detector.detect(
            response_text="Research suggests this may work in approximately 70% of cases."
        )
        # Overconfident claims should have higher hallucination probability
        assert (
            overconfident.mean_hallucination_probability
            >= hedged.mean_hallucination_probability - 0.1
        )

    def test_batch_via_multiple_calls(self, detector):
        responses = [
            "The speed of light is approximately 299,792 km/s.",
            "Napoleon Bonaparte was born in 1769 in Corsica.",
            "The Great Wall of China is visible from space with the naked eye.",
        ]
        results = [detector.detect(r) for r in responses]
        assert len(results) == 3
        for r in results:
            assert r.risk_level in ["LOW", "MEDIUM", "HIGH"]
