import logging
import hashlib
from collections import defaultdict
from typing import List, Optional

from app.core.config import settings
from app.llm_integration.openai_connector import generate_response_from_context, generate_expanded_queries_from_context
from app.retrieval.reranker import get_reranker
from app.vector_store.weaviate_connector import get_weaviate_connector, WeaviateConnector
from app.graph_db.neo4j_connector import get_neo4j_connector, Neo4jConnector
from app.caching.redis_connector import get_redis_connector, RedisConnector
from app.models.query_models import QueryRequest, QueryResponse, SourceChunk, VectorSearchRequest
from app.models.common_models import Subgraph

logger = logging.getLogger(__name__)

def _create_cache_key(query: str, filenames: Optional[List[str]]) -> str:
    """Creates a consistent, hash-based key for caching."""
    key_string = query
    if filenames:
        key_string += "".join(sorted(filenames))
    return "query:" + hashlib.sha256(key_string.encode()).hexdigest()

async def process_user_query(query_request: QueryRequest) -> QueryResponse:
    """
    Processes a user query through a sophisticated, multi-stage RAG pipeline.

    The pipeline consists of the following stages:
    1.  **Cache Check**: Checks Redis for a cached response to this exact query to return immediately.
    2.  **Query Expansion (Optional)**:
        a. Performs an initial, small search to find context.
        b. Uses an LLM to generate multiple, context-aware query variations.
    3.  **Main Retrieval**:
        - If file filters are applied, uses a **per-document retrieval** strategy to fetch a diverse set of candidate chunks from each specified file.
        - If no filters are applied, performs a **global retrieval** across all documents.
    4.  **Re-ranking (Optional)**:
        - Uses a powerful cross-encoder model to re-score all candidate chunks for relevance.
        - If per-document retrieval was used, the re-ranking is also done on a per-document basis to preserve source diversity.
    5.  **Graph Augmentation**: Fetches N-hop subgraphs from Neo4j for entities found in the final, re-ranked chunks.
    6.  **Final Answer Generation**: Synthesizes the final text chunks and graph context into a coherent answer using a powerful LLM.
    7.  **Cache Population**: Caches the final response in Redis for future requests.

    Args:
        query_request: The user's request, containing the query and optional file filters.

    Returns:
        A QueryResponse object containing the final answer and all contextual sources.
    """
    # --- Initial Setup ---
    user_query = query_request.query
    filter_filenames = query_request.filter_filenames
    pipeline_config = settings.RETRIEVAL_PIPELINE
    query_expansion_config = pipeline_config.get('query_expansion', {})
    reranking_config = pipeline_config.get('reranking', {})
    logger.info(f"Processing query: '{user_query}' with filters: {filter_filenames}")

    # --- Stage 1: Cache Check ---
    redis_conn: RedisConnector = get_redis_connector()
    cache_key = _create_cache_key(user_query, filter_filenames)
    if cached_response := await redis_conn.get_query_cache(cache_key):
        logger.info("Returning cached response.")
        return cached_response

    # --- Get Database Connectors ---
    weaviate_conn: WeaviateConnector = get_weaviate_connector()
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()

    # --- Stage 2: Query Expansion (Optional) ---
    queries_for_main_search = [user_query]
    if query_expansion_config.get('enabled'):
        logger.info("Query expansion enabled. Fetching context for expansion...")
        context_chunks_meta = await weaviate_conn.search_similar_chunks(
            query_concepts=[user_query],
            top_k=pipeline_config.get('context_chunks_for_expansion', 3),
            filter_filenames=filter_filenames
        )
        if context_chunks_meta:
            context_text = "\n---\n".join([chunk['chunk_text'] for chunk in context_chunks_meta])
            expanded_queries = await generate_expanded_queries_from_context(user_query, context_text)
            queries_for_main_search.extend(expanded_queries)
            queries_for_main_search = list(set(queries_for_main_search))

    logger.info(f"Performing main search with {len(queries_for_main_search)} queries.")

    # --- Stage 3: Main Retrieval ---
    candidate_chunks_meta = []
    if filter_filenames:
        logger.info("Using per-document retrieval strategy.")
        candidate_chunks_meta = await weaviate_conn.search_chunks_per_document(
            query_concepts=queries_for_main_search,
            filenames=filter_filenames,
            per_file_limit=pipeline_config.get('candidates_per_doc', 5)
        )
    else:
        logger.info("Using global retrieval strategy.")
        candidate_chunks_meta = await weaviate_conn.search_similar_chunks(
            query_concepts=queries_for_main_search,
            top_k=pipeline_config.get('main_search_top_k', 15),
            filter_filenames=None
        )

    if not candidate_chunks_meta:
        return QueryResponse(llm_answer="Could not find any relevant information.", subgraph_context=Subgraph(), source_chunks=[])

    candidate_chunks = [SourceChunk(**meta) for meta in candidate_chunks_meta]

    # --- Stage 4: Re-ranking (Optional) ---
    final_chunks_for_context = []
    reranker = get_reranker() if reranking_config.get('enabled') else None

    if reranker:
        if filter_filenames:
            logger.info("Applying per-document re-ranking strategy to preserve diversity.")
            chunks_by_doc = defaultdict(list)
            for chunk in candidate_chunks:
                chunks_by_doc[chunk.source_document].append(chunk)

            # This is the configurable number of chunks to keep from each document.
            top_n_per_doc = reranking_config.get('top_n_per_reranked_doc', 3)
            for doc_chunks in chunks_by_doc.values():
                reranked_group = reranker.rerank_chunks(user_query, doc_chunks)
                final_chunks_for_context.extend(reranked_group[:top_n_per_doc])
            logger.info(f"Re-ranked and selected top {top_n_per_doc} chunk(s) from {len(chunks_by_doc)} documents.")
        else:
            logger.info("Applying global re-ranking strategy.")
            final_chunks_for_context = reranker.rerank_chunks(user_query, candidate_chunks)
    else:
        logger.info("Re-ranking is disabled. Using top results from initial search.")
        candidate_chunks.sort(key=lambda x: x.score, reverse=True)
        final_chunks_for_context = candidate_chunks[:reranking_config.get('final_top_n', 3)]

    # --- Stage 5, 6, 7: Final Assembly, Generation, and Caching ---
    if not final_chunks_for_context:
        return QueryResponse(llm_answer="Found some initial information, but could not refine it to a final answer.", subgraph_context=Subgraph(), source_chunks=[])

    final_chunks_for_context.sort(key=lambda x: x.score, reverse=True)

    entity_ids_to_fetch = set(eid for chunk in final_chunks_for_context for eid in chunk.entity_ids if eid)
    retrieved_subgraph = Subgraph()
    if entity_ids_to_fetch:
        retrieved_subgraph = await neo4j_conn.get_subgraph_for_entities(list(entity_ids_to_fetch), settings.ENTITY_INFO_HOP_DEPTH)

    combined_context = _format_context_for_llm(final_chunks_for_context, retrieved_subgraph)
    final_answer_text = await generate_response_from_context(user_query, combined_context) or "I found relevant information but had an issue formulating a response."

    final_response = QueryResponse(
        llm_answer=final_answer_text,
        subgraph_context=retrieved_subgraph,
        source_chunks=final_chunks_for_context
    )
    await redis_conn.set_query_cache(cache_key, final_response)
    return final_response


async def perform_raw_vector_search(search_request: VectorSearchRequest) -> List[SourceChunk]:
    """
    Performs a direct vector search against the vector store. If requested,
    it will also perform a second-pass re-ranking on the initial results.
    """
    weaviate_conn = get_weaviate_connector()

    # If reranking is enabled, we should fetch more initial candidates
    # to give the reranker a better pool of documents to work with.
    initial_top_k = search_request.top_k
    if search_request.enable_reranking:
        # Fetch more candidates, e.g., 3x the final desired count, but no more than 50
        initial_top_k = max(search_request.top_k, min(search_request.rerank_top_n * 3, 50))
        logger.info(f"Reranking enabled. Fetching {initial_top_k} initial candidates.")

    # Step 1: Initial vector search
    search_results_meta = await weaviate_conn.search_similar_chunks(
        query_concepts=[search_request.query],
        top_k=initial_top_k,
        filter_filenames=search_request.filter_filenames
    )

    if not search_results_meta:
        return []

    initial_chunks = [SourceChunk(**res) for res in search_results_meta]

    # Step 2: Optional Re-ranking
    if not search_request.enable_reranking:
        logger.info(f"Reranking disabled. Returning top {search_request.top_k} vector search results.")
        # Return results trimmed to the original requested top_k
        return initial_chunks[:search_request.top_k]

    reranker = get_reranker()
    if not reranker:
        logger.warning("Reranking was requested, but the reranker is not enabled in the configuration. Returning top vector search results instead.")
        return initial_chunks[:search_request.rerank_top_n]

    logger.info("Performing re-ranking on initial results...")
    reranked_chunks = reranker.rerank_chunks(search_request.query, initial_chunks)

    # Step 3: Trim to the final desired count
    final_chunks = reranked_chunks[:search_request.rerank_top_n]

    return final_chunks


def _format_context_for_llm(source_chunks: List[SourceChunk], subgraph: Subgraph) -> str:
    """Formats the retrieved chunks and subgraph into a single text block for the LLM."""
    context_parts = []
    if source_chunks:
        context_parts.append("--- Start of Retrieved Textual Context ---")
        for i, chunk in enumerate(source_chunks):
            context_parts.append(f"\n[Source Chunk {i+1} from '{chunk.source_document}' | Similarity Score: {chunk.score:.4f}]:\n{chunk.chunk_text}")
        context_parts.append("\n--- End of Retrieved Textual Context ---")
    if not subgraph.is_empty():
        context_parts.append("\n\n--- Start of Retrieved Knowledge Graph Context ---")
        if subgraph.nodes:
            context_parts.append("\nEntities Found:")
            for node in subgraph.nodes:
                context_parts.append(f"- Node: [Name: {node.label}, Type: {node.type}]")
        if subgraph.edges:
            context_parts.append("\nRelationships Found:")
            for edge in subgraph.edges:
                context_parts.append(f"- Relationship: ({edge.source}) -[{edge.label}]-> ({edge.target})")
        context_parts.append("--- End of Retrieved Knowledge Graph Context ---")
    if not context_parts:
        return "No relevant information was found in the knowledge base to answer this query."
    return "\n".join(context_parts)