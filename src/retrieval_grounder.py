"""
Retrieval Grounder.

Checks each atomic claim against a reference corpus
or provided context to produce a retrieval support score.
Used as one signal in the Bayesian hallucination scorer.
"""

import os
from typing import List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

GROUNDING_PROMPT = """\
You are a fact-checking assistant.

Given a factual claim and reference context, determine how well \
the claim is supported by the context.

Return ONLY a float between 0.0 and 1.0:
- 1.0 = fully supported by context
- 0.5 = partially supported or ambiguous
- 0.0 = not supported, contradicted, or context is irrelevant

Claim: {claim}

Reference context:
{context}

Support score (0.0 to 1.0):"""


class RetrievalGrounder:
    """
    Scores how well each claim is supported by retrieved context.

    Retrieval grounding is a critical signal because it catches
    fabricated statistics and unsupported causal claims that
    semantic similarity scoring misses entirely.

    One key lesson: aggressive retrieval weighting causes false
    positives. We use this as one signal in a Bayesian ensemble,
    not as a standalone gate.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-3-small",
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.prompt = ChatPromptTemplate.from_template(GROUNDING_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.vectorstore: Optional[FAISS] = None

    def build_index(self, documents: List[str]) -> None:
        """
        Build a FAISS vector index from reference documents.

        Args:
            documents: List of reference text strings.
        """
        docs = [Document(page_content=d) for d in documents]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def ground_claim(
        self,
        claim: str,
        reference_context: Optional[str] = None,
        top_k: int = 3,
    ) -> float:
        """
        Score how well a claim is supported by reference material.

        Args:
            claim: Atomic factual claim to check.
            reference_context: Direct context string. If None,
                               retrieves from vectorstore.
            top_k: Number of documents to retrieve if using vectorstore.

        Returns:
            Support score between 0.0 and 1.0.
        """
        if reference_context is None:
            if self.vectorstore is None:
                # No reference available — return neutral score
                return 0.5
            retrieved = self.vectorstore.similarity_search(claim, k=top_k)
            reference_context = "\n\n".join(doc.page_content for doc in retrieved)

        raw = self.chain.invoke({"claim": claim, "context": reference_context})

        return self._parse_score(raw)

    def ground_claims(
        self,
        claims: List[str],
        reference_context: Optional[str] = None,
    ) -> List[float]:
        """
        Score a list of claims against reference material.

        Returns list of support scores, one per claim.
        """
        return [self.ground_claim(claim, reference_context) for claim in claims]

    @staticmethod
    def _parse_score(raw: str) -> float:
        """Parse float score from LLM output safely."""
        raw = raw.strip()
        try:
            score = float(raw)
            return max(0.0, min(1.0, score))
        except ValueError:
            # Try to find a float in the text
            import re

            matches = re.findall(r"\d+\.?\d*", raw)
            if matches:
                score = float(matches[0])
                return max(0.0, min(1.0, score))
            return 0.5


class MockRetrievalGrounder:
    """
    Mock grounder for testing without API keys.
    Returns configurable scores for testing.
    """

    def __init__(self, default_score: float = 0.7):
        self.default_score = default_score

    def build_index(self, documents: List[str]) -> None:
        pass

    def ground_claim(
        self, claim: str, reference_context: Optional[str] = None, top_k: int = 3
    ) -> float:
        return self.default_score

    def ground_claims(
        self, claims: List[str], reference_context: Optional[str] = None
    ) -> List[float]:
        return [self.default_score for _ in claims]
