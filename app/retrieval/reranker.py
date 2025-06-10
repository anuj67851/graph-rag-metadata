import logging
from typing import List
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.documents import Document

from app.core.config import settings
from app.models.query_models import SourceChunk

logger = logging.getLogger(__name__)

class ReRanker:
    """
    A modular component for re-ranking retrieved documents using a cross-encoder model.
    """
    def __init__(self):
        reranking_config = settings.RETRIEVAL_PIPELINE['reranking']
        model_repo = reranking_config['model_repo']

        self.reranker = CrossEncoderReranker(
            model=model_repo,
            top_n=reranking_config['final_top_n']
        )
        logger.info(f"ReRanker initialized with model: {model_repo}")

    def rerank_chunks(self, query: str, chunks: List[SourceChunk]) -> List[SourceChunk]:
        """
        Re-ranks a list of SourceChunk objects based on their relevance to a query.

        Args:
            query: The original user query.
            chunks: The list of SourceChunk objects retrieved from the initial search.

        Returns:
            A sorted and trimmed list of the most relevant SourceChunk objects.
        """
        if not chunks:
            return []

        # Convert our SourceChunk objects to LangChain's Document format
        documents_to_rerank = [
            Document(page_content=chunk.chunk_text, metadata={"original_chunk": chunk})
            for chunk in chunks
        ]

        # Use the LangChain reranker to compress (sort and trim) the documents
        reranked_docs = self.reranker.compress_documents(
            documents=documents_to_rerank,
            query=query
        )

        # Extract our original SourceChunk objects from the metadata of the re-ranked docs
        final_chunks = [doc.metadata["original_chunk"] for doc in reranked_docs]

        logger.info(f"Re-ranked {len(chunks)} chunks down to {len(final_chunks)}.")
        return final_chunks

# --- Singleton Management ---
_reranker_instance: ReRanker = None

def get_reranker() -> ReRanker:
    """Provides a singleton instance of the ReRanker."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = ReRanker()
    return _reranker_instance