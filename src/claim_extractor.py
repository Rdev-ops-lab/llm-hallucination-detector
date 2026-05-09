"""
Claim Extractor using LangChain.

Decomposes an LLM response into atomic factual claims
before hallucination scoring. Claim-level evaluation
catches partial hallucinations that response-level
scoring hides.
"""

import os
import json
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


CLAIM_EXTRACTION_PROMPT = """\
You are an expert at decomposing text into atomic factual claims.

Given a response, extract every individual factual claim as a \
separate item. Each claim should:
- Be a single, verifiable statement
- Be self-contained (understandable without the original text)
- Contain exactly one checkable fact

Return ONLY a JSON array of strings. No explanation. No markdown.

Response to decompose:
{response}

JSON array of atomic claims:"""


class ClaimExtractor:
    """
    Extracts atomic factual claims from LLM responses using LangChain.

    Claim-level decomposition is critical because partial hallucinations
    hide inside otherwise correct responses. A response that is 90%
    accurate can still contain a fabricated statistic or unsupported
    causal claim that response-level scoring would miss entirely.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.prompt = ChatPromptTemplate.from_template(CLAIM_EXTRACTION_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def extract(self, response_text: str) -> List[str]:
        """
        Extract atomic claims from a response.

        Args:
            response_text: The LLM response to decompose.

        Returns:
            List of atomic factual claim strings.
        """
        if not response_text or not response_text.strip():
            return []

        raw_output = self.chain.invoke({"response": response_text})
        claims = self._parse_claims(raw_output)
        return claims

    def _parse_claims(self, raw: str) -> List[str]:
        """Parse JSON array from LLM output safely."""
        raw = raw.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if str(c).strip()]
        except json.JSONDecodeError:
            # Fallback: split by newlines if JSON parsing fails
            lines = [
                line.strip().lstrip("-•1234567890. ")
                for line in raw.split("\n")
                if line.strip()
            ]
            return [l for l in lines if len(l) > 10]

        return []


class MockClaimExtractor:
    """
    Mock extractor for testing without API keys.
    Splits on periods as a simple fallback.
    """

    def extract(self, response_text: str) -> List[str]:
        sentences = [
            s.strip()
            for s in response_text.replace("\n", " ").split(".")
            if len(s.strip()) > 15
        ]
        return sentences
