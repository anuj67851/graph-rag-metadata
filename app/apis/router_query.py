import logging
from typing import List

from fastapi import APIRouter, HTTPException, Body, status
from app.services.query_service import process_user_query, perform_raw_vector_search
from app.models.query_models import QueryRequest, QueryResponse, VectorSearchRequest, SourceChunk

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["Querying"],
    responses={
        400: {"description": "Bad Request. The query may be empty or malformed."},
        500: {"description": "Internal Server Error."}
    }
)

@router.post(
    "/",
    response_model=QueryResponse,
    summary="Process a query using the full RAG pipeline.",
    response_description="A structured response containing the LLM-generated answer, source chunks, and graph context."
)
async def handle_user_query(
        query_request: QueryRequest = Body(..., description="The user's query and optional document filters.")
):
    """
    Accepts a user query and processes it through the complete RAG pipeline:
    1.  **Cache Check**: Looks for a previously cached response.
    2.  **Retrieval**: Performs a hybrid search to find relevant text chunks from the vector store.
    3.  **Graph Augmentation**: Fetches related entities and relationships from the knowledge graph to enrich the context.
    4.  **LLM Generation**: Synthesizes the text and graph context into a coherent, final answer using a large language model.
    5.  **Cache Population**: Caches the new response for future requests.
    """
    if not query_request.query or not query_request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty.")

    logger.info(f"Received query request: '{query_request.query}' with filters: {query_request.filter_filenames}")

    try:
        query_response = await process_user_query(query_request)
        logger.info("Successfully processed query. Returning response.")
        return query_response
    except Exception as e:
        logger.error(f"Unexpected error processing query '{query_request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected server error occurred: {e}")

@router.post(
    "/vector_search",
    response_model=List[SourceChunk],
    summary="Perform a raw vector search for text chunks.",
    response_description="A list of the most similar SourceChunk objects found in the vector store."
)
async def direct_vector_search(
        search_request: VectorSearchRequest = Body(..., description="The search query, top_k, and optional document filters.")
):
    """
    Bypasses the full RAG pipeline and performs a direct vector similarity search
    against the Weaviate vector store.

    This is useful for debugging, testing chunking strategies, or for applications
    that only need raw semantic search results without LLM-based answer generation.
    The returned chunks include their similarity score.
    """
    logger.info(f"Received direct vector search request: '{search_request.query}' with top_k={search_request.top_k}")
    try:
        return await perform_raw_vector_search(search_request)
    except Exception as e:
        logger.error(f"Error during direct vector search for query '{search_request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected server error occurred during vector search: {e}")