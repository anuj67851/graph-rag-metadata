import logging
from fastapi import APIRouter, HTTPException, Body, status
from app.services.query_service import process_user_query
from app.models.query_models import QueryRequest, QueryResponse

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Create an APIRouter instance for query-related endpoints
router = APIRouter(
    prefix="/query",
    tags=["Querying"],
    responses={404: {"description": "Not found"}},
)

@router.post(
    "/", # POST to /query/
    response_model=QueryResponse,
    summary="Process a user's natural language query against the knowledge base.",
    description=(
            "Accepts a user query, processes it through the new RAG pipeline "
            "(semantic chunk retrieval, graph augmentation, LLM response generation), "
            "and returns a structured response including the answer, source text chunks, "
            "and an explorable subgraph."
    )
)
async def handle_user_query(
        query_request: QueryRequest = Body(..., description="The user's query.")
):
    """
    Endpoint to process a user query and return a RAG-based answer.
    """
    if not query_request.query or not query_request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty."
        )

    logger.info(f"Received query request: '{query_request.query}'")

    try:
        # Call the refactored query service to process the query
        query_response: QueryResponse = await process_user_query(query_request)
        logger.info(f"Successfully processed query. LLM Answer (snippet): '{query_response.llm_answer[:100]}...'")
        return query_response

    except Exception as e:
        logger.error(f"Unexpected error processing query '{query_request.query}': {e}", exc_info=True)
        # For unhandled server errors, a 500 is most appropriate.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {str(e)}"
        )

if __name__ == "__main__":
    # The __main__ block for ad-hoc testing remains conceptually the same,
    # but would now test the new chunk-based workflow.
    print("Query router defined. Run with a FastAPI application (e.g., app/main.py) to test endpoints.")