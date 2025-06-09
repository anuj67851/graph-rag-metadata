import logging
from typing import List, Optional

from app.core.config import settings
from app.llm_integration.openai_connector import generate_response_from_context
from app.vector_store.faiss_connector import get_faiss_connector, FaissConnector
from app.graph_db.neo4j_connector import get_neo4j_connector, Neo4jConnector
from app.models.query_models import QueryRequest, QueryResponse, SourceChunk
from app.models.common_models import Subgraph

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _format_context_for_llm(source_chunks: List[SourceChunk], subgraph: Subgraph) -> str:
    """
    Formats the retrieved chunks and subgraph into a single text block for the LLM.
    """
    context_parts = []

    # 1. Add Source Text Chunks
    if source_chunks:
        context_parts.append("--- Start of Retrieved Textual Context ---")
        for i, chunk in enumerate(source_chunks):
            context_parts.append(f"\n[Source Chunk {i+1} from '{chunk.source_document}']:\n{chunk.chunk_text}")
        context_parts.append("\n--- End of Retrieved Textual Context ---")

    # 2. Add Knowledge Graph Context
    if not subgraph.is_empty():
        context_parts.append("\n\n--- Start of Retrieved Knowledge Graph Context ---")
        if subgraph.nodes:
            context_parts.append("\nEntities Found:")
            for node in subgraph.nodes:
                props_str = f"Name: {node.label}, Type: {node.type}"
                context_parts.append(f"- Node: [{props_str}]")
        if subgraph.edges:
            context_parts.append("\nRelationships Found:")
            for edge in subgraph.edges:
                context_parts.append(f"- Relationship: ({edge.source}) -[{edge.label}]-> ({edge.target})")
        context_parts.append("--- End of Retrieved Knowledge Graph Context ---")

    if not context_parts:
        return "No relevant information was found in the knowledge base to answer this query."

    return "\n".join(context_parts)


async def process_user_query(query_request: QueryRequest) -> QueryResponse:
    """
    Main service function to process a user's query using the new RAG pipeline.
    1. Search for relevant text chunks in FAISS.
    2. Extract entity IDs from the metadata of these chunks.
    3. Retrieve a subgraph for these entities from Neo4j.
    4. Combine chunks and subgraph into a context.
    5. Generate a final answer with an LLM.
    """
    user_query = query_request.query
    logger.info(f"Processing user query: '{user_query}'")

    faiss_conn: FaissConnector = await get_faiss_connector()
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()

    # 1. Search FAISS for relevant text chunks
    retrieved_chunks_meta = await faiss_conn.search_similar_chunks(
        query_text=user_query,
        top_k=settings.SEMANTIC_SEARCH_TOP_K
    )
    if not retrieved_chunks_meta:
        logger.info("No relevant chunks found in FAISS for the query.")
        return QueryResponse(
            llm_answer="I could not find any relevant information in the uploaded documents to answer your question.",
            subgraph_context=Subgraph(),
            source_chunks=[]
        )

    source_chunks = [SourceChunk(**meta) for meta in retrieved_chunks_meta]

    # 2. Extract entity IDs from chunk metadata to augment with graph context
    entity_ids_to_fetch = set()
    for chunk in source_chunks:
        # The metadata in FAISS connector now directly contains 'entity_ids'
        if chunk.chunk_text in [meta['chunk_text'] for meta in retrieved_chunks_meta]:
            # Find corresponding metadata to get entity_ids
            for meta in retrieved_chunks_meta:
                if meta['chunk_text'] == chunk.chunk_text:
                    entity_ids_to_fetch.update(meta.get('entity_ids', []))
                    break


    # 3. Retrieve subgraph from Neo4j based on entities found in chunks
    retrieved_subgraph = Subgraph()
    if entity_ids_to_fetch:
        logger.info(f"Fetching subgraph for entities found in chunks: {list(entity_ids_to_fetch)}")
        retrieved_subgraph = await neo4j_conn.get_subgraph_for_entities(
            canonical_names=list(entity_ids_to_fetch),
            hop_depth=settings.ENTITY_INFO_HOP_DEPTH
        )
    else:
        logger.info("No entities were linked to the retrieved chunks. Proceeding with text context only.")

    # 4. Format the combined context for the LLM
    combined_context = _format_context_for_llm(source_chunks, retrieved_subgraph)
    logger.debug(f"Formatted LLM context:\n{combined_context}")

    # 5. Generate final answer using LLM
    final_answer_text: Optional[str] = await generate_response_from_context(user_query, combined_context)

    if final_answer_text is None:
        logger.error("LLM failed to generate a final answer.")
        final_answer_text = "I found some relevant information but encountered an issue while trying to formulate a response."

    return QueryResponse(
        llm_answer=final_answer_text,
        subgraph_context=retrieved_subgraph,
        source_chunks=source_chunks
    )