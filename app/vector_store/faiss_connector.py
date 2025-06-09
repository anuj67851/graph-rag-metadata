import faiss
import numpy as np
import os
import json
import logging
from typing import List, Optional, Dict, Any

from app.core.config import settings
from app.llm_integration.openai_connector import get_text_embedding

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FaissConnector:
    def __init__(self, index_file_path: str, metadata_file_path: str, embedding_dimension: int):
        self.index_file_path = index_file_path
        self.metadata_file_path = metadata_file_path
        self.embedding_dimension = embedding_dimension

        self.index: Optional[faiss.Index] = None
        # This list stores the metadata dictionary for each vector, indexed by its position in FAISS.
        self.metadata: List[Dict[str, Any]] = []

        self._load_or_initialize()

    def _initialize(self):
        """Initializes a new FAISS index and clears metadata."""
        logger.info(f"Initializing new FAISS index with dimension {self.embedding_dimension}.")
        # IndexFlatL2 is a basic index for L2 distance. Good for starting.
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.metadata = []

    def _load_or_initialize(self):
        """Loads the FAISS index and metadata from disk if they exist, otherwise initializes them."""
        index_dir = os.path.dirname(self.index_file_path)
        if not os.path.exists(index_dir):
            logger.info(f"Creating directory for FAISS index: {index_dir}")
            os.makedirs(index_dir, exist_ok=True)

        try:
            if os.path.exists(self.index_file_path) and os.path.exists(self.metadata_file_path):
                logger.info(f"Loading FAISS index from {self.index_file_path}")
                self.index = faiss.read_index(self.index_file_path)

                logger.info(f"Loading metadata from {self.metadata_file_path}")
                with open(self.metadata_file_path, 'r') as f:
                    self.metadata = json.load(f)

                if self.index.d != self.embedding_dimension:
                    logger.warning(
                        f"Loaded FAISS index dimension ({self.index.d}) "
                        f"differs from configured dimension ({self.embedding_dimension}). Re-initializing."
                    )
                    self._initialize()
                elif self.index.ntotal != len(self.metadata):
                    logger.warning(
                        f"FAISS index size ({self.index.ntotal}) "
                        f"differs from metadata size ({len(self.metadata)}). Re-initializing."
                    )
                    self._initialize()
                else:
                    logger.info(f"Successfully loaded FAISS index with {self.index.ntotal} vectors and metadata.")
            else:
                logger.info("FAISS index or metadata file not found. Initializing a new index.")
                self._initialize()
        except Exception as e:
            logger.error(f"Error during FAISS index loading: {e}. Initializing a new index.", exc_info=True)
            self._initialize()

    def save(self):
        """Saves the FAISS index and metadata to disk."""
        if self.index is not None:
            try:
                logger.info(f"Saving FAISS index to {self.index_file_path} with {self.index.ntotal} vectors.")
                faiss.write_index(self.index, self.index_file_path)

                logger.info(f"Saving metadata to {self.metadata_file_path}")
                with open(self.metadata_file_path, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
                logger.info("FAISS index and metadata saved successfully.")
            except Exception as e:
                logger.error(f"Error saving FAISS index or metadata: {e}", exc_info=True)
        else:
            logger.warning("Attempted to save FAISS index, but it's not initialized.")

    async def add_chunk_embeddings_batch(self, chunks_data: List[Dict[str, Any]]):
        """
        Generates embeddings for a batch of text chunks and adds them to the FAISS index.

        Args:
            chunks_data: A list of dictionaries, where each dict contains:
                         'chunk_text', 'source_document', 'entity_ids'.
        """
        if self.index is None:
            logger.error("FAISS index not initialized. Cannot add embeddings.")
            return

        texts_to_embed = [chunk['chunk_text'] for chunk in chunks_data]
        if not texts_to_embed:
            return

        embeddings: List[Optional[List[float]]] = []
        for text in texts_to_embed:
            embedding = await get_text_embedding(text)
            embeddings.append(embedding)

        valid_embeddings = []
        valid_metadata = []
        for i, emb in enumerate(embeddings):
            if emb and len(emb) == self.embedding_dimension:
                valid_embeddings.append(emb)
                valid_metadata.append(chunks_data[i])
            else:
                logger.warning(f"Skipping chunk due to invalid embedding. Source: {chunks_data[i].get('source_document')}")

        if not valid_embeddings:
            logger.warning("No valid embeddings were generated for the batch.")
            return

        vectors = np.array(valid_embeddings, dtype=np.float32)
        try:
            self.index.add(vectors)
            self.metadata.extend(valid_metadata)
            logger.info(f"Added {len(valid_embeddings)} new vectors to FAISS. Total vectors: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"FAISS error adding vectors: {e}", exc_info=True)

    async def search_similar_chunks(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Generates embedding for the query text and searches FAISS for similar chunks.

        Args:
            query_text: The text to search for.
            top_k: The number of top similar chunks to return.

        Returns:
            A list of metadata dictionaries for the most similar chunks.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.info("FAISS index is empty or not initialized. Returning empty search results.")
            return []

        actual_top_k = min(top_k, self.index.ntotal)
        if actual_top_k == 0:
            return []

        query_embedding = await get_text_embedding(query_text)

        if not query_embedding or len(query_embedding) != self.embedding_dimension:
            logger.warning("Could not generate query embedding or dimension mismatch. Returning empty search results.")
            return []

        query_vector = np.array([query_embedding], dtype=np.float32)

        try:
            distances, indices = self.index.search(query_vector, actual_top_k)
        except Exception as e:
            logger.error(f"FAISS search error: {e}", exc_info=True)
            return []

        results = []
        if len(indices[0]) > 0:
            for i in range(len(indices[0])):
                faiss_idx = indices[0][i]
                if faiss_idx != -1 and faiss_idx < len(self.metadata):
                    retrieved_metadata = self.metadata[faiss_idx].copy()
                    retrieved_metadata['score'] = float(distances[0][i])
                    results.append(retrieved_metadata)
        return results

    def get_index_size(self) -> int:
        return self.index.ntotal if self.index else 0

# --- FastAPI App Lifecycle Management ---
faiss_connector_instance: Optional[FaissConnector] = None

async def get_faiss_connector() -> FaissConnector:
    """Provides access to the FaissConnector singleton instance."""
    global faiss_connector_instance
    if faiss_connector_instance is None:
        if settings.EMBEDDING_DIMENSION is None:
            raise ValueError("EMBEDDING_DIMENSION is not set in settings. FAISS Connector cannot be initialized.")

        faiss_connector_instance = FaissConnector(
            index_file_path=settings.FAISS_INDEX_PATH,
            metadata_file_path=settings.FAISS_INDEX_PATH + "_metadata.json",
            embedding_dimension=settings.EMBEDDING_DIMENSION
        )
        logger.info("FAISS Connector instance created.")
    return faiss_connector_instance

async def init_vector_store():
    """Called on FastAPI app startup to initialize/load the FAISS index."""
    await get_faiss_connector()
    logger.info("FAISS Vector Store Initialized and ready.")

async def save_vector_store_on_shutdown():
    """Called on FastAPI app shutdown to save the FAISS index."""
    connector = await get_faiss_connector()
    if connector:
        connector.save()
    logger.info("FAISS Vector Store save attempt on shutdown completed.")


if __name__ == "__main__":
    import asyncio

    async def test_faiss_connector():
        print("--- Testing FAISS Connector (Chunk-based) ---")
        if not settings.OPENAI_API_KEY:
            print("OPENAI_API_KEY not set. Skipping tests.")
            return

        connector = await get_faiss_connector()
        print(f"Initial FAISS index size: {connector.get_index_size()}")

        # Test adding chunk embeddings
        chunks_to_add = [
            {
                "chunk_text": "Alpha Corporation is a global leader in artificial intelligence and machine learning solutions.",
                "source_document": "report-2023.pdf",
                "entity_ids": ["Alpha Corporation"]
            },
            {
                "chunk_text": "The company announced Project Nova, an initiative focused on quantum computing research.",
                "source_document": "report-2023.pdf",
                "entity_ids": ["Project Nova"]
            },
            {
                "chunk_text": "Beta Systems, a primary competitor, specializes in cloud infrastructure and data analytics.",
                "source_document": "market-analysis.docx",
                "entity_ids": ["Beta Systems"]
            }
        ]

        print("\n1. Adding chunk embeddings to FAISS...")
        # A real implementation would filter out chunks already processed
        # For this test, we assume a fresh start or add all.
        await connector.add_chunk_embeddings_batch(chunks_to_add)

        print(f"FAISS index size after additions: {connector.get_index_size()}")
        print(f"Metadata entries: {len(connector.metadata)}")

        # Test searching
        print("\n2. Searching for 'AI research company'...")
        search_query = "AI research company"
        search_results = await connector.search_similar_chunks(search_query, top_k=2)
        if search_results:
            print(f"   Search results for '{search_query}':")
            for chunk_meta in search_results:
                print(f"   - Score: {chunk_meta['score']:.4f}, Doc: {chunk_meta['source_document']}, Text: '{chunk_meta['chunk_text'][:40]}...'")
        else:
            print(f"   No search results for '{search_query}'.")

        print("\n3. Manually testing save()...")
        connector.save()

        print("\n4. Testing loading index (simulating restart)...")
        global faiss_connector_instance
        faiss_connector_instance = None

        new_connector = await get_faiss_connector()
        print(f"   FAISS index size after loading: {new_connector.get_index_size()}")
        if new_connector.get_index_size() > 0:
            print("   Data appears to have loaded correctly.")
        else:
            print("   Data did not load as expected or index was empty.")

        print("\n--- FAISS Connector Test Finished ---")

    if settings.OPENAI_API_KEY:
        asyncio.run(test_faiss_connector())
    else:
        print("Skipping FAISS connector tests as OPENAI_API_KEY is not set.")