import logging
from fastapi import APIRouter, HTTPException, Body, status
from app.services.query_service import process_user_query
from app.models.query_models import QueryRequest, QueryResponse

# Initialize logging
logger = logging.getLogger(__name__)

# Create an APIRouter instance for query-related endpoints
router = APIRouter(
    prefix="/query",
    tags=["Querying"],
)

@router.post(
    "/", # POST to /query/
    response_model=QueryResponse,
    summary="Process a user's natural language query against the knowledge base.",
    description=(
            "Accepts a user query and an optional list of filenames to filter by. "
            "The query is processed through the RAG pipeline (caching, semantic chunk retrieval, "
            "graph augmentation, LLM response generation) and returns a structured response."
    )
)
async def handle_user_query(
        query_request: QueryRequest = Body(..., description="The user's query and optional filters.")
):
    """
    Endpoint to process a user query and return a RAG-based answer.
    """
    if not query_request.query or not query_request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty."
        )

    logger.info(
        f"Received query request: '{query_request.query}' with filters: {query_request.filter_filenames}"
    )

    try:
        # The query_request object, now containing optional filters,
        # is passed directly to the service.
        query_response: QueryResponse = await process_user_query(query_request)

        logger.info(f"Successfully processed query. Returning response.")
        return query_response

    except Exception as e:
        logger.error(f"Unexpected error processing query '{query_request.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {str(e)}"
        )