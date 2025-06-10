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

        Args:
            query_concepts: The queries to search for.
            top_k: The number of top similar chunks to return.
            filter_filenames: An optional list of filenames to restrict the search to.

        Returns:
            A list of result dictionaries, each containing the chunk data and search score.
        """
        client = self._get_client()
        class_name = settings.WEAVIATE_CLASS_NAME

        # The FIX: Pass the list directly to the 'concepts' key
        near_text_filter = {"concepts": query_concepts}

        # Build the 'where' filter if filenames are provided
        where_filter = None
        if filter_filenames:
            where_filter = {
                "path": ["source_document"],
                "operator": "ContainsAny",
                "valueString": filter_filenames
            }

        try:
            result = (
                client.query
                .get(class_name, ["chunk_text", "source_document", "entity_ids"])
                .with_near_text(near_text_filter)
                .with_limit(top_k)
                .with_additional(["score"]) # 'score' is a Weaviate-native similarity metric
                .with_where(where_filter)
                .do()
            )

            search_results = result["data"]["Get"][class_name]
            # Reformat the results to match the 'SourceChunk' model
            reformatted_results = []
            for res in search_results:
                reformatted_results.append({
                    "chunk_text": res.get('chunk_text'),
                    "source_document": res.get('source_document'),
                    "entity_ids": res.get('entity_ids', []),
                    "score": res.get('_additional', {}).get('score', 0.0)
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