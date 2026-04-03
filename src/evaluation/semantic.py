from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class SemanticScore:
    similarity: float
    passed: bool
    threshold: float
    candidate: str
    reference: str


class SemanticEvaluator:
    """Compares two texts by meaning rather than exact string matching."""

    _instances: dict[str, SemanticEvaluator] = {}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @classmethod
    def get_instance(cls, model_name: str = "all-MiniLM-L6-v2") -> SemanticEvaluator:
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name)
        return cls._instances[model_name]

    def embed(self, text: str) -> NDArray[np.float32]:
        result = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(result, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        result = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(result, dtype=np.float32)

    def similarity(self, text_a: str, text_b: str) -> float:
        embeddings = self.embed_batch([text_a, text_b])
        score = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, score))

    def evaluate(
        self,
        candidate: str,
        reference: str,
        threshold: float = 0.75,
    ) -> SemanticScore:
        score = self.similarity(candidate, reference)
        return SemanticScore(
            similarity=round(score, 4),
            passed=score >= threshold,
            threshold=threshold,
            candidate=candidate,
            reference=reference,
        )

    def evaluate_best_match(
        self,
        candidate: str,
        references: list[str],
        threshold: float = 0.75,
    ) -> SemanticScore:
        if not references:
            return SemanticScore(
                similarity=0.0,
                passed=False,
                threshold=threshold,
                candidate=candidate,
                reference="",
            )

        all_texts = [candidate] + references
        embeddings = self.embed_batch(all_texts)
        candidate_emb = embeddings[0]
        reference_embs = embeddings[1:]
        similarities = np.dot(reference_embs, candidate_emb)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_score = max(0.0, min(1.0, best_score))

        return SemanticScore(
            similarity=round(best_score, 4),
            passed=best_score >= threshold,
            threshold=threshold,
            candidate=candidate,
            reference=references[best_idx],
        )