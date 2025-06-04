import faiss
import numpy as np
import os
import json
import logging
from typing import List, Tuple, Optional, Dict

from app.core.config import settings
from app.llm_integration.openai_connector import get_text_embedding

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FaissConnector:
    def __init__(self, index_file_path: str, id_map_file_path: str, embedding_dimension: int):
        self.index_file_path = index_file_path
        self.id_map_file_path = id_map_file_path
        self.embedding_dimension = embedding_dimension

        self.index: Optional[faiss.Index] = None
        # self.id_map maps the sequential FAISS index ID (0, 1, 2...)
        # to our application's entity identifier (e.g., canonical_name).
        self.id_map: List[str] = []
        # For quick lookups to see if an entity ID (canonical_name) is already in the index
        # and to find its FAISS index. Key: entity_id, Value: faiss_vector_idx
        self.entity_id_to_faiss_idx: Dict[str, int] = {}

        self._load_or_initialize_index()

    def _initialize_index(self):
        """Initializes a new FAISS index and clears maps."""
        logger.info(f"Initializing new FAISS index with dimension {self.embedding_dimension}.")
        # IndexFlatL2 is a basic index for L2 distance.
        # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat.
        # IndexIDMap can be wrapped around another index to use custom 64-bit IDs directly,
        # which simplifies updates/deletions if your app uses such IDs.
        # For now, IndexFlatL2 with external id_map is simpler to start.
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.id_map = []
        self.entity_id_to_faiss_idx = {}

    def _load_or_initialize_index(self):
        """Loads the FAISS index and ID map from disk if they exist, otherwise initializes them."""
        index_dir = os.path.dirname(self.index_file_path)
        if not os.path.exists(index_dir):
            logger.info(f"Creating directory for FAISS index: {index_dir}")
            os.makedirs(index_dir, exist_ok=True)

        try:
            if os.path.exists(self.index_file_path) and os.path.exists(self.id_map_file_path):
                logger.info(f"Loading FAISS index from {self.index_file_path}")
                self.index = faiss.read_index(self.index_file_path)

                logger.info(f"Loading ID map from {self.id_map_file_path}")
                with open(self.id_map_file_path, 'r') as f:
                    loaded_id_map_data = json.load(f)
                    self.id_map = loaded_id_map_data.get("id_map", [])
                    self.entity_id_to_faiss_idx = loaded_id_map_data.get("entity_id_to_faiss_idx", {})

                if self.index.d != self.embedding_dimension:
                    logger.warning(
                        f"Loaded FAISS index dimension ({self.index.d}) "
                        f"differs from configured dimension ({self.embedding_dimension}). Re-initializing."
                    )
                    self._initialize_index()
                elif self.index.ntotal != len(self.id_map):
                    logger.warning(
                        f"FAISS index size ({self.index.ntotal}) "
                        f"differs from ID map size ({len(self.id_map)}). Re-initializing."
                    )
                    self._initialize_index()
                else:
                    logger.info(f"Successfully loaded FAISS index with {self.index.ntotal} vectors and ID map.")
            else:
                logger.info("FAISS index or ID map file not found. Initializing a new index.")
                self._initialize_index()
        except Exception as e:
            logger.error(f"Error during FAISS index loading: {e}. Initializing a new index.", exc_info=True)
            self._initialize_index()

    def save_index(self):
        """Saves the FAISS index and ID map to disk."""
        if self.index is not None:
            try:
                logger.info(f"Saving FAISS index to {self.index_file_path} with {self.index.ntotal} vectors.")
                faiss.write_index(self.index, self.index_file_path)

                logger.info(f"Saving ID map to {self.id_map_file_path}")
                id_map_data = {
                    "id_map": self.id_map,
                    "entity_id_to_faiss_idx": self.entity_id_to_faiss_idx
                }
                with open(self.id_map_file_path, 'w') as f:
                    json.dump(id_map_data, f, indent=2)
                logger.info("FAISS index and ID map saved successfully.")
            except Exception as e:
                logger.error(f"Error saving FAISS index or ID map: {e}", exc_info=True)
        else:
            logger.warning("Attempted to save FAISS index, but it's not initialized.")

    async def add_or_update_entity_embedding(self, entity_id: str, entity_text_representation: str):
        """
        Generates an embedding for the entity's text and adds or updates it in the FAISS index.
        'entity_id' is typically the canonical_name.

        If the entity_id already exists, its vector is updated.
        Note: Updating vectors in IndexFlatL2 typically means removing the old one and adding the new one.
        FAISS `update` method is not available for all index types.
        A more robust update for IndexFlatL2 would require rebuilding or careful ID management.
        This implementation uses a remove-and-add approach if supported, or rebuilds for simplicity if not directly supported.
        For IndexFlatL2, true "update" of a specific vector at an index is not directly supported.
        We can "reconstruct" the index without the vector or add new and manage IDs.
        This version will attempt to remove by ID if the index supports it, otherwise it might be complex.
        Given IndexFlatL2, a common strategy for updates is to rebuild periodically or if many updates.
        This function will demonstrate adding new, and a placeholder for update.
        A simpler "update" is to re-add and let searches find the latest if IDs are not strictly managed with faiss IDs.

        For now, if entity_id exists, we'll log a warning and potentially re-add,
        which is not a true update for IndexFlatL2 and can lead to stale vectors
        if not handled carefully by re-indexing periodically or using a different FAISS index type.

        A strategy for updates: if entity_id is in `entity_id_to_faiss_idx`,
        we cannot directly update the vector at `faiss_idx` in IndexFlatL2.
        We would need to rebuild the index without that specific vector and then add the new one.
        Or, use an index that supports removal by ID (like IndexIDMap wrapping an index that supports remove_ids).

        This implementation will overwrite if using `faiss.IndexIDMap` in future.
        For `IndexFlatL2`, we'll effectively "add" and the old one might still be there.
        A cleaner `add_or_update` for `IndexFlatL2` usually involves rebuilding the index excluding the old vector.
        This is complex for real-time.

        Let's simplify: if an ID exists, we will not re-add to avoid duplicate vectors without a removal mechanism.
        The service layer should decide if a full re-index from Neo4j is needed.
        """
        if self.index is None:
            logger.error("FAISS index not initialized. Cannot add embedding.")
            return

        if entity_id in self.entity_id_to_faiss_idx:
            # This means the entity exists. For a true update with IndexFlatL2,
            # you'd need to rebuild the index excluding the old vector.
            # This is complex. A simpler strategy is periodic full rebuilds from the source of truth (Neo4j).
            # For now, we will skip re-adding if it exists to prevent duplicate vectors if this method is called multiple times.
            # Or, if you want to replace, you'd need a mechanism to find the old vector's index and a strategy to handle it.
            # Let's assume we are only adding new unique entities for now, or the service layer handles idempotency.
            # logger.info(f"Entity ID '{entity_id}' already exists in FAISS. To update, consider re-indexing or using an index type that supports direct updates/removals.")

            # For a basic "update by replace" if we allow re-adding the same entity_id
            # (which requires careful handling of faiss_idx if it changes):
            # 1. Get new embedding
            # 2. (If possible) Remove old vector at self.entity_id_to_faiss_idx[entity_id]
            # 3. Add new vector, update maps.
            # This is hard with IndexFlatL2.

            # A simple approach if we expect updates: just re-embed and add.
            # This means the old vector for this entity_id is still there.
            # The search will find both. This is usually not desired.
            # Solution: Use IndexIDMap2 which allows adding with specific IDs.
            # faiss.IndexIDMap2(faiss.IndexFlatL2(self.embedding_dimension))
            # then self.index.add_with_ids(vector, np.array([custom_id_for_entity]))
            # and self.index.remove_ids(np.array([custom_id_for_entity_to_remove]))

            # For now, let's make `add_entity_embedding` idempotent for simplicity.
            # If you call it again with the same entity_id, it won't add a duplicate vector.
            # The "update" part would imply that the `entity_text_representation` has changed,
            # requiring re-embedding and replacing the old vector.
            logger.warning(f"Entity ID '{entity_id}' found. Update logic for IndexFlatL2 is complex (requires re-indexing or specific index types). Current call will not update the existing vector for this ID in this simplified example.")
            return # Or proceed to re-embed and replace if a strategy is defined.

        embedding = await get_text_embedding(entity_text_representation)

        if embedding and len(embedding) == self.embedding_dimension:
            vector = np.array([embedding], dtype=np.float32)
            try:
                self.index.add(vector)
                current_faiss_idx = self.index.ntotal - 1 # FAISS adds sequentially
                self.id_map.append(entity_id)
                self.entity_id_to_faiss_idx[entity_id] = current_faiss_idx
                logger.info(f"Added embedding for entity '{entity_id}' at FAISS index {current_faiss_idx}. Total vectors: {self.index.ntotal}")
            except Exception as e:
                logger.error(f"FAISS error adding vector for entity '{entity_id}': {e}", exc_info=True)
        elif not embedding:
            logger.warning(f"Could not generate embedding for entity '{entity_id}'. Not added to FAISS.")
        else: # Dimension mismatch
            logger.warning(
                f"Embedding dimension mismatch for entity '{entity_id}'. "
                f"Expected {self.embedding_dimension}, got {len(embedding)}. Not added."
            )


    async def search_similar_entities(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Generates embedding for the query text and searches FAISS for similar entities.

        Args:
            query_text: The text to search for.
            top_k: The number of top similar entities to return.

        Returns:
            A list of tuples: (entity_id, similarity_score).
            For IndexFlatL2, the score is L2 distance (lower is better).
        """
        if self.index is None or self.index.ntotal == 0:
            logger.info("FAISS index is empty or not initialized. Returning empty search results.")
            return []

        actual_top_k = min(top_k, self.index.ntotal) # Cannot get more results than in index
        if actual_top_k == 0:
            return []

        query_embedding = await get_text_embedding(query_text)

        if not query_embedding or len(query_embedding) != self.embedding_dimension:
            logger.warning("Could not generate query embedding or dimension mismatch. Returning empty search results.")
            return []

        query_vector = np.array([query_embedding], dtype=np.float32)

        try:
            # D: distances, I: indices (0-based index into the vectors added to FAISS)
            distances, indices = self.index.search(query_vector, actual_top_k)
        except Exception as e:
            logger.error(f"FAISS search error: {e}", exc_info=True)
            return []

        results = []
        if len(indices[0]) > 0:
            for i in range(len(indices[0])):
                faiss_idx = indices[0][i]
                if faiss_idx != -1 and faiss_idx < len(self.id_map): # faiss_idx can be -1 if fewer than k results
                    entity_id = self.id_map[faiss_idx]
                    distance_score = float(distances[0][i])
                    # For L2 distance, lower is better.
                    # If cosine similarity is desired, use IndexFlatIP and normalize vectors,
                    # or convert L2: sim = 1 / (1 + L2_distance) or exp(-L2_distance)
                    # This score needs to be handled by the caller (e.g., thresholding).
                    results.append((entity_id, distance_score))
                else:
                    logger.debug(f"Invalid FAISS index {faiss_idx} encountered in search results.")
        return results

    def get_index_size(self) -> int:
        return self.index.ntotal if self.index else 0

# --- FastAPI App Lifecycle Management ---
# This approach uses a global instance.
faiss_connector_instance: Optional[FaissConnector] = None

async def get_faiss_connector() -> FaissConnector:
    """Provides access to the FaissConnector singleton instance."""
    global faiss_connector_instance
    if faiss_connector_instance is None:
        # Ensure settings are fully loaded, especially EMBEDDING_DIMENSION
        if settings.EMBEDDING_DIMENSION is None:
            # This should have been derived or set in config.py
            raise ValueError("EMBEDDING_DIMENSION is not set in settings. FAISS Connector cannot be initialized.")

        faiss_connector_instance = FaissConnector(
            index_file_path=settings.FAISS_INDEX_PATH,
            id_map_file_path=settings.FAISS_INDEX_PATH + "_id_map.json", # Convention for id map file
            embedding_dimension=settings.EMBEDDING_DIMENSION
        )
        logger.info("FAISS Connector instance created.")
    return faiss_connector_instance

async def init_vector_store():
    """Called on FastAPI app startup to initialize/load the FAISS index."""
    await get_faiss_connector() # This will trigger initialization if not already done
    logger.info("FAISS Vector Store Initialized and ready.")

async def save_vector_store_on_shutdown():
    """Called on FastAPI app shutdown to save the FAISS index."""
    connector = await get_faiss_connector() # Ensures instance exists
    if connector:
        connector.save_index()
    logger.info("FAISS Vector Store save attempt on shutdown completed.")


if __name__ == "__main__":
    import asyncio

    async def test_faiss_connector():
        print("--- Testing FAISS Connector ---")
        # This test relies on a valid OpenAI API key for get_text_embedding
        # and proper configuration in settings.
        if not settings.OPENAI_API_KEY:
            print("OPENAI_API_KEY not set. Skipping FAISS connector tests that require embeddings.")
            return

        connector = await get_faiss_connector()
        print(f"Initial FAISS index size: {connector.get_index_size()}")

        # Test adding embeddings
        entity_data = {
            "AlphaCorp_ID": "Alpha Corporation is a leader in AI.",
            "ProjectNova_ID": "Project Nova focuses on quantum computing research.",
            "BetaSystems_ID": "Beta Systems develops innovative software solutions."
        }

        print("\n1. Adding entities to FAISS...")
        for entity_id, text_repr in entity_data.items():
            # In a real app, check if entity_id already processed to avoid duplicates if not desired
            if entity_id not in connector.entity_id_to_faiss_idx:
                await connector.add_or_update_entity_embedding(entity_id, text_repr)
            else:
                print(f"   Entity '{entity_id}' already in index, skipping add for test.")


        print(f"FAISS index size after additions: {connector.get_index_size()}")
        print(f"ID Map: {connector.id_map}")
        print(f"Entity ID to FAISS Index Map: {connector.entity_id_to_faiss_idx}")


        # Test searching
        print("\n2. Searching for 'AI research company'...")
        search_query = "AI research company"
        search_results = await connector.search_similar_entities(search_query, top_k=2)
        if search_results:
            print(f"   Search results for '{search_query}':")
            for eid, score in search_results:
                print(f"   - Entity ID: {eid}, Score (L2 Distance): {score:.4f}")
        else:
            print(f"   No search results for '{search_query}' or search failed.")

        print("\n3. Searching for 'quantum project'...")
        search_query_2 = "quantum project"
        search_results_2 = await connector.search_similar_entities(search_query_2, top_k=2)
        if search_results_2:
            print(f"   Search results for '{search_query_2}':")
            for eid, score in search_results_2:
                print(f"   - Entity ID: {eid}, Score (L2 Distance): {score:.4f}")
        else:
            print(f"   No search results for '{search_query_2}' or search failed.")

        # Test saving the index (will be called on shutdown in app)
        print("\n4. Manually testing save_index()...")
        connector.save_index()

        # Test loading by creating a new instance (simulates app restart)
        print("\n5. Testing loading index (simulating restart)...")
        global faiss_connector_instance # Allow reassignment for test
        faiss_connector_instance = None # Clear current instance

        new_connector = await get_faiss_connector()
        print(f"   FAISS index size after loading: {new_connector.get_index_size()}")
        print(f"   ID Map loaded: {new_connector.id_map}")
        if new_connector.get_index_size() > 0 and "AlphaCorp_ID" in new_connector.id_map :
            print("   Data appears to have loaded correctly.")
        else:
            print("   Data did not load as expected or index was empty.")
            if new_connector.get_index_size() == 0 and len(entity_data)>0 :
                print("    WARN: Index is empty after reload, check save/load paths and permissions.")


        # Clean up dummy index files for test (optional)
        # if os.path.exists(settings.FAISS_INDEX_PATH):
        #     os.remove(settings.FAISS_INDEX_PATH)
        # if os.path.exists(settings.FAISS_INDEX_PATH + "_id_map.json"):
        #     os.remove(settings.FAISS_INDEX_PATH + "_id_map.json")
        # print("\nDummy index files cleaned up.")

        print("\n--- FAISS Connector Test Finished ---")

    if settings.OPENAI_API_KEY: # Requires OpenAI for embeddings
        asyncio.run(test_faiss_connector())
    else:
        print("Skipping FAISS connector tests as OPENAI_API_KEY is not set (needed for embeddings).")