import weaviate
import logging
from typing import List, Optional, Dict, Any
import numpy as np

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
                    "vectorizer": "text2vec-transformers",
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "model": settings.EMBEDDING_MODEL_REPO,
                            "vectorizeClassName": False,
                            # Ensure proper inference URL
                            "inferenceUrl": f"http://{settings.WEAVIATE_HOST.replace('weaviate', 'transformers-inference')}:8080"
                        }
                    },
                    # Explicit BM25 configuration - important for hybrid search
                    "invertedIndexConfig": {
                        "bm25": {
                            "enabled": True,
                            "b": 0.75,
                            "k1": 1.2
                        },
                        "cleanupIntervalSeconds": 60,
                        "stopwords": {
                            "additions": None,
                            "preset": "en",
                            "removals": None
                        }
                    },
                    "properties": [
                        {
                            "name": "chunk_text",
                            "dataType": ["text"],
                            "description": "The actual text content of the chunk.",
                            "tokenization": "word",
                            # Important: ensure this field is indexed for BM25
                            "indexSearchable": True,
                            "indexFilterable": False
                        },
                        {
                            "name": "source_document",
                            "dataType": ["string"],
                            "description": "The filename of the source document for this chunk.",
                            "indexSearchable": False,
                            "indexFilterable": True
                        },
                        {
                            "name": "entity_ids",
                            "dataType": ["string[]"],
                            "description": "A list of canonical entity names found in this chunk.",
                            "indexSearchable": False,
                            "indexFilterable": True
                        }
                    ]
                }
                client.schema.create_class(class_obj)
                logger.info(f"Successfully created Weaviate class '{class_name}' with vectorizer '{settings.EMBEDDING_MODEL_REPO}' and BM25 enabled.")
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

    async def get_vector_for_concepts(self, concepts: List[str]) -> Optional[List[float]]:
        """
        Asks Weaviate to generate a single, averaged vector for a list of text concepts.
        This is the correct way to "use" the external vectorizer.
        """
        client = self._get_client()
        class_name = settings.WEAVIATE_CLASS_NAME

        # We perform a dummy query using nearText. Weaviate calculates the
        # vector for the concepts internally. We ask for the vector back.
        try:
            result = (
                client.query
                .get(class_name, []) # No properties needed
                .with_near_text({"concepts": concepts})
                .with_limit(1) # We only need one result to grab the vector from
                .with_additional("vector")
                .do() # In async context, you might need await .ado()
            )

            # Extract the vector from the query that was actually performed
            if (result and result.get("data", {}).get("Get", {}).get(class_name) and
                    result["data"]["Get"][class_name][0]["_additional"]["vector"]):
                vector = result["data"]["Get"][class_name][0]["_additional"]["vector"]
                logger.info(f"Successfully generated a vector for {len(concepts)} concepts.")
                return vector
            else:
                logger.warning("Could not retrieve a vector from Weaviate for the given concepts.")
                return None
        except Exception as e:
            logger.error(f"Error generating vector via Weaviate: {e}", exc_info=True)
            return None

    async def search_similar_chunks(
            self,
            query: str,
            alpha: float,
            top_k: int = 5,
            filter_filenames: Optional[List[str]] = None,
            search_vector: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches for similar chunks using Weaviate's hybrid search. It can use a
        different set of concepts for the keyword and vector parts of the search.
        """
        client = self._get_client()
        class_name = settings.WEAVIATE_CLASS_NAME

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
                .with_hybrid(
                    query=query,               # Use original query for precise keyword (BM25) search.
                    alpha=alpha,                      # The balance parameter.
                    vector=search_vector,
                )
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
            query: str,
            alpha: float,
            filenames: List[str],
            per_file_limit: int = 3,
            search_vector: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs a separate hybrid search for each specified document.
        """
        all_results = []
        # Use a set to keep track of the text of chunks we've already added
        # This prevents returning the exact same chunk text from different searches
        seen_chunk_texts = set()

        for filename in filenames:
            # For each file, perform a targeted search
            logger.info(f"Performing targeted hybrid search in file: '{filename}'")

            # The search_similar_chunks method already supports filtering, so we can reuse it
            results_for_file = await self.search_similar_chunks(
                query=query,
                alpha=alpha,
                top_k=per_file_limit,
                filter_filenames=[filename],
                search_vector=search_vector,
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