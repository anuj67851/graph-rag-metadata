import logging
from typing import List

# We still use the base model from langchain-huggingface
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from app.core.config import settings
from app.models.query_models import SourceChunk

logger = logging.getLogger(__name__)

class ReRanker:
    """
    A modular component for re-ranking retrieved documents using a cross-encoder model.
    This version directly computes scores and attaches them to the chunks.
    """
    def __init__(self):
        reranking_config = settings.RETRIEVAL_PIPELINE.get('reranking', {})
        model_repo = reranking_config.get('model_repo', 'cross-encoder/ms-marco-MiniLM-L-6-v2')

        # We only need the model itself now, not the high-level compressor
        self.model = HuggingFaceCrossEncoder(
            model_name=model_repo,
            model_kwargs={'device': 'cpu'}
        )
        logger.info(f"ReRanker initialized with model: {model_repo}")

    def rerank_chunks(self, query: str, chunks: List[SourceChunk]) -> List[SourceChunk]:
        """
        Re-ranks a list of SourceChunk objects by scoring them against the query,
        updating their scores, and returning the sorted list.

        Args:
            query: The original user query.
            chunks: The list of SourceChunk objects from the initial search.

        Returns:
            A sorted and trimmed list of the most relevant SourceChunk objects with updated scores.
        """
        if not chunks:
            return []

        # 1. Create pairs of [query, chunk_text] for the cross-encoder to score.
        query_chunk_pairs = [[query, chunk.chunk_text] for chunk in chunks]

        # 2. Get the new, more accurate scores from the model.
        # The model's score() method returns a list of float scores.
        new_scores: List[float] = self.model.score(query_chunk_pairs)

        # 3. Attach the new scores back to our original SourceChunk objects.
        for i, chunk in enumerate(chunks):
            chunk.score = new_scores[i]

        # 4. Sort the chunks in descending order based on their new scores.
        chunks.sort(key=lambda x: x.score, reverse=True)

        # 5. Trim the list to the final desired number of chunks.
        reranking_config = settings.RETRIEVAL_PIPELINE.get('reranking', {})
        final_top_n = reranking_config.get('final_top_n', 3)
        final_chunks = chunks[:final_top_n]

        logger.info(f"Re-ranked {len(chunks)} chunks down to {len(final_chunks)}.")
        for chunk in final_chunks:
            logger.debug(f"Re-ranked chunk '{chunk.source_document}' score: {chunk.score:.4f}")

        return final_chunks

# --- Singleton Management (remains the same) ---
_reranker_instance: ReRanker = None

def get_reranker() -> ReRanker:
    reranking_enabled = settings.RETRIEVAL_PIPELINE.get('reranking', {}).get('enabled', False)
    if not reranking_enabled:
        return None

    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = ReRanker()
    return _reranker_instance