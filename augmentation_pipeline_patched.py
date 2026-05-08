"""
Patched augmentation pipeline copy.

This wrapper keeps the original file untouched while fixing:
1. RAG retrieval leakage of the target problem.
2. RAG prompt confounding by removing extra CoT instructions.
"""

import random
from typing import Any, Dict, List, Optional

import augmentation_pipeline as base_pipeline

logger = base_pipeline.logger


class AugmentationPipeline(base_pipeline.AugmentationPipeline):
    """Patched prompt builder for cleaner ablation behavior."""

    def _rag_context(
        self,
        signature: str,
        docstring: str,
        problem: Dict[str, Any],
        all_problems: Optional[List[Dict[str, Any]]],
    ) -> str:
        retrieved = self._retrieve_similar(problem, all_problems, top_k=3)
        retrieved_str = "\n\n".join(
            f"Similar function {i+1}:\n```python\n{p.get('code', p.get('solution', ''))}\n```"
            for i, p in enumerate(retrieved)
        ) if retrieved else ""

        context_block = self.RAG_API_CONTEXT
        if retrieved_str:
            context_block += f"\n\n## Similar Functions from Dataset\n{retrieved_str}"

        return (
            f"Use the following context to help implement the function:\n\n"
            f"{context_block}\n\n"
            f"Now complete this function:\n\n"
            f"{signature}\n"
            f'    """{docstring}"""\n\n'
            f"Provide only the complete function code."
        )

    def _retrieve_similar(
        self,
        problem: Dict[str, Any],
        all_problems: Optional[List[Dict[str, Any]]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieve similar problems while always excluding the target problem."""
        if not all_problems:
            return []

        others = [p for p in all_problems if p.get("id") != problem.get("id")]
        if not others:
            return []

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            problem_ids = tuple(p.get("id") for p in all_problems)
            if self._rag_index is None or self._rag_index.get("problem_ids") != problem_ids:
                model = SentenceTransformer("all-MiniLM-L6-v2")
                corpus = [
                    f"{p.get('signature', '')} {p.get('docstring', '')}"
                    for p in all_problems
                ]
                embeddings = np.asarray(model.encode(corpus, show_progress_bar=False))
                self._rag_index = {
                    "model": model,
                    "embeddings": embeddings,
                    "problems": all_problems,
                    "problem_ids": problem_ids,
                }

            query = f"{problem.get('signature', '')} {problem.get('docstring', '')}"
            q_emb = np.asarray(self._rag_index["model"].encode([query]))
            sims = np.dot(self._rag_index["embeddings"], q_emb.T).flatten()

            ranked = []
            for idx in np.argsort(sims)[::-1]:
                candidate = self._rag_index["problems"][idx]
                if candidate.get("id") == problem.get("id"):
                    continue
                ranked.append(candidate)
                if len(ranked) >= top_k:
                    break

            return ranked

        except ImportError:
            logger.warning("sentence-transformers not available; using random retrieval for RAG.")
            return random.sample(others, min(top_k, len(others)))
