import logging
import hashlib
from typing import List, Optional

from app.core.config import settings
from app.llm_integration.openai_connector import generate_response_from_context, generate_expanded_queries_from_context
from app.retrieval.reranker import ReRanker, get_reranker
from app.vector_store.weaviate_connector import get_weaviate_connector, WeaviateConnector
from app.graph_db.neo4j_connector import get_neo4j_connector, Neo4jConnector
from app.caching.redis_connector import get_redis_connector, RedisConnector
from app.models.query_models import QueryRequest, QueryResponse, SourceChunk
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
    Processes a user query via a multi-stage RAG pipeline with expansion and re-ranking.
    """
    user_query = query_request.query
    filter_filenames = query_request.filter_filenames

    pipeline_config = settings.RETRIEVAL_PIPELINE
    query_expansion_config = pipeline_config.get('query_expansion', {})
    reranking_config = pipeline_config.get('reranking', {})

    logger.info(f"Processing query: '{user_query}' with filters: {filter_filenames}")

    # Get all necessary connectors
    redis_conn: RedisConnector = get_redis_connector()
    weaviate_conn: WeaviateConnector = get_weaviate_connector()
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()

    # --- Check Cache ---
    cache_key = _create_cache_key(user_query, filter_filenames)
    cached_response = await redis_conn.get_query_cache(cache_key)
    if cached_response:
        return cached_response

    # --- Stage 1: Initial Context Retrieval for Expansion ---
    context_chunks_for_expansion = []
    if query_expansion_config.get('enabled'):
        context_chunks_meta = await weaviate_conn.search_similar_chunks(
            query_concepts=[user_query,],
            top_k=pipeline_config.get('context_chunks_for_expansion', 2),
            filter_filenames=filter_filenames
        )
        context_chunks_for_expansion = [chunk['chunk_text'] for chunk in context_chunks_meta]

    # --- Stage 2: Context-Aware Query Expansion ---
    queries_for_main_search = [user_query]
    if query_expansion_config.get('enabled') and context_chunks_for_expansion:
        context_text = "\n---\n".join(context_chunks_for_expansion)
        expanded_queries = await generate_expanded_queries_from_context(user_query, context_text)
        queries_for_main_search.extend(expanded_queries)
        # Remove duplicates
        queries_for_main_search = list(set(queries_for_main_search))

    logger.info(f"Performing main search with {len(queries_for_main_search)} queries: {queries_for_main_search}")

    # --- Stage 3: Main Search ---
    candidate_chunks_meta = await weaviate_conn.search_similar_chunks(
        query_concepts=queries_for_main_search, # Weaviate can take multiple concepts
        top_k=pipeline_config.get('main_search_top_k', 15),
        filter_filenames=filter_filenames
    )
    if not candidate_chunks_meta:
        logger.warning("Main search returned no candidate chunks.")
        return QueryResponse(llm_answer="Could not find any relevant information.", subgraph_context=Subgraph(), source_chunks=[])

    candidate_chunks = [SourceChunk(**meta) for meta in candidate_chunks_meta]

    # --- Stage 4: Re-ranking ---
    final_chunks_for_context = candidate_chunks
    if reranking_config.get('enabled'):
        reranker: ReRanker = get_reranker()
        final_chunks_for_context = reranker.rerank_chunks(user_query, candidate_chunks)

    # --- Final Context Assembly and Answer Generation ---
    if not final_chunks_for_context:
        logger.warning("No chunks remained after re-ranking.")
        return QueryResponse(llm_answer="Found some initial information, but could not refine it to a final answer.", subgraph_context=Subgraph(), source_chunks=[])

    # Augment with graph context
    entity_ids_to_fetch = set(eid for chunk in final_chunks_for_context for eid in chunk.entity_ids if eid)
    retrieved_subgraph = Subgraph()
    if entity_ids_to_fetch:
        retrieved_subgraph = await neo4j_conn.get_subgraph_for_entities(
            canonical_names=list(entity_ids_to_fetch),
            hop_depth=settings.ENTITY_INFO_HOP_DEPTH
        )

    # Generate final answer
    combined_context = _format_context_for_llm(final_chunks_for_context, retrieved_subgraph)
    final_answer_text = await generate_response_from_context(user_query, combined_context)
    if not final_answer_text:
        final_answer_text = "I found relevant information but had an issue formulating a response."

    # Construct and cache final response
    final_response = QueryResponse(
        llm_answer=final_answer_text,
        subgraph_context=retrieved_subgraph,
        source_chunks=final_chunks_for_context
    )
    await redis_conn.set_query_cache(cache_key, final_response)

    return final_response


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