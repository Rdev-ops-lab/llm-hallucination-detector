"""
Sample usage of the Hallucination Detector.

Run with mock mode (no API key needed):
    USE_MOCK=true python examples/sample_usage.py

Run with real OpenAI:
    OPENAI_API_KEY=your_key python examples/sample_usage.py
"""

import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import HallucinationDetector

USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
test_cases = [
    {
        "name": "Factual response — should pass",
        "response": (
            "Python was created by Guido van Rossum and first released in 1991. "
            "It emphasizes code readability and uses significant indentation. "
            "Python supports multiple programming paradigms."
        ),
        "reference": (
            "Python is a high-level programming language created by Guido van Rossum, "
            "with its first release in 1991. It is known for its clear syntax and "
            "readability, and supports procedural, object-oriented, and functional programming."
        ),
    },
    {
        "name": "Partially hallucinated response — fabricated statistic",
        "response": (
            "Python is a popular programming language used in data science. "
            "Studies show that 94% of all Fortune 500 companies use Python as their "
            "primary language. It was first released in 1991 by Guido van Rossum."
        ),
        "reference": (
            "Python is widely used in data science and machine learning. "
            "Many large companies use Python, but exact adoption statistics vary. "
            "Python was first released in 1991."
        ),
    },
    {
        "name": "Overconfident claim — no hedging",
        "response": (
            "Scientists have definitely proven that this treatment always works. "
            "100% of patients recover completely. This is an established medical fact."
        ),
        "reference": (
            "Early research suggests the treatment may be effective in some patients, "
            "but large-scale trials are still ongoing."
        ),
    },
]


def run_examples():
    print(f"Running examples in {'MOCK' if USE_MOCK else 'REAL'} mode\n")
    print("=" * 60)

    detector = HallucinationDetector(use_mock=USE_MOCK, flag_threshold=0.45)

    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}] {case['name']}")
        print("-" * 60)

        result = detector.detect(
            response_text=case["response"],
            reference_context=case.get("reference"),
        )

        print(f"Flagged:          {result.response_flagged}")
        print(f"Risk Level:       {result.risk_level}")
        print(f"Claims:           {result.total_claim_count} total, "
              f"{result.flagged_claim_count} flagged")
        print(f"Max Hallucination Prob: {result.max_hallucination_probability:.1%}")
        print(f"Escalate to Human: {result.escalate_to_human}")

        if result.claim_scores:
            print("\nClaim breakdown:")
            for j, claim in enumerate(result.claim_scores, 1):
                flag_icon = "⚠️ " if claim["flagged"] else "✓  "
                print(
                    f"  {flag_icon} [{claim['confidence_level']}] "
                    f"{claim['claim'][:80]}..."
                    if len(claim["claim"]) > 80
                    else f"  {flag_icon} [{claim['confidence_level']}] {claim['claim']}"
                )
                print(f"       → {claim['explanation']}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    run_examples()
