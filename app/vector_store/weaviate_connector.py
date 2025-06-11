import weaviate
import logging
from typing import List, Optional, Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)

class WeaviateConnector:
    """
    A connector for managing vector storage and search with a Weaviate instance.
    This class provides a modular interface for vector database operations.
    """
    _client: Optional[weaviate.Client] = None

    def _get_client(self) -> weaviate.Client:
        """Establishes and returns the Weaviate client."""
        if self._client is None:
            try:
                self._client = weaviate.Client(
                    url=f"http://{settings.WEAVIATE_HOST}:{settings.WEAVIATE_PORT}",
                    timeout_config=(10, 240) # (connect_timeout, read_timeout)
                )
                if not self._client.is_ready():
                    raise ConnectionError("Weaviate is not ready.")
                logger.info(f"Weaviate client connected to http://{settings.WEAVIATE_HOST}:{settings.WEAVIATE_PORT}")
            except Exception as e:
                logger.error(f"Failed to connect to Weaviate: {e}", exc_info=True)
                raise
        return self._client

    def _ensure_schema_exists(self):
        """
        Ensures the required class schema exists in Weaviate, creating it if necessary.
        This is where the embedding model is configured.
        """
        client = self._get_client()
        class_name = settings.WEAVIATE_CLASS_NAME

        try:
            # Check if class already exists
            client.schema.get(class_name)
            logger.info(f"Weaviate class '{class_name}' already exists.")
        except weaviate.exceptions.UnexpectedStatusCodeException as e:
            if e.status_code == 404:
                logger.info(f"Weaviate class '{class_name}' not found. Creating now...")

                class_obj = {
                    "class": class_name,
                    # Configure Weaviate to use the text2vec-transformers module
                    "vectorizer": "text2vec-transformers",
                    "moduleConfig": {
                        "text2vec-transformers": {
                            # Setting the model to use for vectorization.
                            "model": settings.EMBEDDING_MODEL_REPO,
                            "vectorizeClassName": False
                        }
                    },
                    "properties": [
                        {
                            "name": "chunk_text",
                            "dataType": ["text"],
                            "description": "The actual text content of the chunk.",
                        },
                        {
                            "name": "source_document",
                            "dataType": ["string"],
                            "description": "The filename of the source document for this chunk.",
                        },
                        {
                            "name": "entity_ids",
                            "dataType": ["string[]"],
                            "description": "A list of canonical entity names found in this chunk.",
                        }
                    ]
                }
                client.schema.create_class(class_obj)
                logger.info(f"Successfully created Weaviate class '{class_name}' with vectorizer '{settings.EMBEDDING_MODEL_REPO}'.")
            else:
                raise e

    async def add_chunk_batch(self, chunks_data: List[Dict[str, Any]]):
        """
        Adds a batch of text chunks to Weaviate. Weaviate handles the embedding internally.

        Args:
            chunks_data: A list of dictionaries, where each dict has keys that match
                         the properties in the Weaviate schema (e.g., 'chunk_text', 'source_document').
        """
        client = self._get_client()
        class_name = settings.WEAVIATE_CLASS_NAME

        with client.batch as batch:
            batch.batch_size = 100 # Configure batch size as needed
            for chunk in chunks_data:
                properties = {
                    "chunk_text": chunk.get("chunk_text"),
                    "source_document": chunk.get("source_document"),
                    "entity_ids": chunk.get("entity_ids", [])
                }
                batch.add_data_object(properties, class_name)

        logger.info(f"Added {len(chunks_data)} chunks to Weaviate class '{class_name}'.")

    async def search_similar_chunks(
            self,
            query_concepts: List[str],
            top_k: int = 5,
            filter_filenames: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for similar chunks using Weaviate's nearText search.
        This function now handles both global and filtered searches robustly.
        """
        client = self._get_client()
        class_name = settings.WEAVIATE_CLASS_NAME

        near_text_filter = {"concepts": query_concepts}

        where_filter = None
        if filter_filenames:
            logger.info(f"Applying search filter for documents: {filter_filenames}")
            where_filter = {
                "path": ["source_document"],
                "operator": "ContainsAny",
                "valueString": filter_filenames
            }

        try:
            # 1. Start with the base query builder
            query_builder = (
                client.query
                .get(class_name, ["chunk_text", "source_document", "entity_ids"])
                .with_near_text(near_text_filter)
                .with_limit(top_k)
                .with_additional(["score", "distance", "certainty"])
            )

            # 2. Conditionally add the .with_where() clause ONLY if a filter exists
            if where_filter is not None:
                query_builder = query_builder.with_where(where_filter)

            # 3. Execute the fully constructed query
            result = query_builder.do()

            search_results = result.get("data", {}).get("Get", {}).get(class_name, [])
            reformatted_results = []
            if search_results:
                for res in search_results:
                    # Prioritize 'certainty' from nearText, but fall back to 'score' for other search types
                    additional_props = res.get('_additional', {})
                    score = 0.0  # Default to 0.0
                    # For nearText, 'certainty' is the primary similarity score (0 to 1).
                    if 'certainty' in additional_props and additional_props['certainty'] is not None:
                        score = additional_props['certainty']
                    # Fallback for other potential search methods that might use 'score'.
                    elif 'score' in additional_props and additional_props['score'] is not None:
                        score = additional_props['score']

                    reformatted_results.append({
                        "chunk_text": res.get('chunk_text'),
                        "source_document": res.get('source_document'),
                        "entity_ids": res.get('entity_ids', []),
                        "score": score
                    })
            return reformatted_results

        except Exception as e:
            logger.error(f"Weaviate search error: {e}", exc_info=True)
            return []

    async def delete_chunks_by_filename(self, filename: str) -> int:
        """Deletes all chunk objects associated with a specific filename."""
        client = self._get_client()
        class_name = settings.WEAVIATE_CLASS_NAME

        where_filter = {
            "path": ["source_document"],
            "operator": "Equal",
            "valueString": filename
        }

        # The 'dryRun' parameter can be used to count items before deleting
        delete_result = client.batch.delete_objects(
            class_name=class_name,
            where=where_filter,
            output='verbose'
        )

        num_deleted = delete_result.get('results', {}).get('successful', 0)
        logger.info(f"Deleted {num_deleted} chunks from Weaviate for file '{filename}'.")
        return num_deleted

    async def search_chunks_per_document(
            self,
            query_concepts: List[str],
            filenames: List[str],
            per_file_limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Performs a separate search for each specified document and returns the top
        N chunks from each, effectively diversifying the result set.

        Args:
            query_concepts: The list of search queries.
            filenames: The list of document filenames to search within.
            per_file_limit: The number of top chunks to return from each document.

        Returns:
            A combined list of the top chunks from each specified document.
        """
        all_results = []
        # Use a set to keep track of the text of chunks we've already added
        # This prevents returning the exact same chunk text from different searches
        seen_chunk_texts = set()

        for filename in filenames:
            # For each file, perform a targeted search
            logger.info(f"Performing targeted search in file: '{filename}' for concepts: {query_concepts}")

            # The search_similar_chunks method already supports filtering, so we can reuse it
            results_for_file = await self.search_similar_chunks(
                query_concepts=query_concepts,
                top_k=per_file_limit,
                filter_filenames=[filename] # Filter for just this one file
            )

            for res in results_for_file:
                if res['chunk_text'] not in seen_chunk_texts:
                    all_results.append(res)
                    seen_chunk_texts.add(res['chunk_text'])

        # Optionally, re-sort all collected results by score to have the best ones first
        all_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        logger.info(f"Retrieved {len(all_results)} chunks via per-document search.")
        return all_results

# --- Singleton Management ---
_weaviate_connector_instance: Optional[WeaviateConnector] = None

def get_weaviate_connector() -> WeaviateConnector:
    """Provides a singleton instance of the WeaviateConnector."""
    global _weaviate_connector_instance
    if _weaviate_connector_instance is None:
        _weaviate_connector_instance = WeaviateConnector()
    return _weaviate_connector_instance

async def init_vector_store():
    """Called on FastAPI app startup to initialize the Weaviate client and schema."""
    connector = get_weaviate_connector()
    connector._get_client()
    connector._ensure_schema_exists()
    logger.info("Weaviate Vector Store Initialized and ready.")

async def save_vector_store_on_shutdown():
    """Weaviate is a service, so there's no file to save on shutdown."""
    logger.info("Weaviate is a persistent service; no save action needed on shutdown.")