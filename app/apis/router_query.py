import logging
from fastapi import APIRouter, HTTPException, Body, status
from app.services.query_service import process_user_query
from app.models.query_models import QueryRequest, QueryResponse

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Create an APIRouter instance for query-related endpoints
router = APIRouter(
    # prefix=f"{settings.API_V1_STR}/query", # Using API_V1_STR from settings
    prefix="/query", # Simpler prefix
    tags=["Querying"],
    responses={404: {"description": "Not found"}},
)

@router.post(
    "/", # POST to /query/
    response_model=QueryResponse,
    summary="Process a user's natural language query against the knowledge graph.",
    description=(
            "Accepts a user query, processes it through the RAG pipeline "
            "(entity linking, intent classification, subgraph retrieval, LLM response generation), "
            "and returns a structured response including the answer, a explorable subgraph, "
            "and the textual context provided to the LLM."
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
        # Call the query service to process the query
        query_response: QueryResponse = await process_user_query(query_request)

        # The query_service should handle its own internal errors and formulate
        # an appropriate QueryResponse, potentially with an error message in llm_answer.
        # We don't necessarily need to raise HTTPExceptions here unless process_user_query itself raises one
        # that we want to propagate or convert.

        logger.info(f"Successfully processed query. LLM Answer (snippet): '{query_response.llm_answer[:100]}...'")
        return query_response

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions that might be explicitly raised by deeper layers
        # (though ideally, services return data or business errors, not HTTP exceptions)
        logger.warning(f"HTTPException caught during query processing: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error processing query '{query_request.query}': {e}", exc_info=True)
        # Return a generic error response structure consistent with QueryResponse
        # This helps the client (Streamlit UI) handle errors more gracefully
        # as it always expects a QueryResponse-like object.
        # However, for unhandled server errors, a 500 might be more appropriate.
        # For now, let's raise a 500 for unexpected errors.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {str(e)}"
        )

if __name__ == "__main__":
    # This router is meant to be run by Uvicorn as part of a FastAPI app.
    # Example of how to run this router for quick manual testing:
    # from fastapi import FastAPI
    # import uvicorn
    #
    # temp_app = FastAPI(title="Query Router Test API")
    # temp_app.include_router(router)
    #
    # # For this test to be meaningful, you need:
    # # 1. All services and connectors (OpenAI, Neo4j, FAISS) to be functional.
    # # 2. Neo4j and FAISS to be populated with data (e.g., by running ingestion_service).
    # # 3. .env, config.yaml, schema.yaml, prompts.yaml properly set up.
    #
    # @temp_app.on_event("startup")
    # async def startup_event():
    #     # Initialize connectors (as would happen in the main app)
    #     from app.llm_integrations.openai_connector import async_client as openai_async_client # Check init
    #     from app.graph_db.neo4j_connector import init_neo4j_driver
    #     from app.vector_store.faiss_connector import init_vector_store
    #     if not openai_async_client: print("Warning: OpenAI client not initialized in connector.")
    #     await init_neo4j_driver()
    #     await init_vector_store()
    #     print("Connectors initialized for test.")
    #
    # @temp_app.on_event("shutdown")
    # async def shutdown_event():
    #     from app.graph_db.neo4j_connector import close_neo4j_driver
    #     from app.vector_store.faiss_connector import save_vector_store_on_shutdown
    #     await close_neo4j_driver()
    #     await save_vector_store_on_shutdown()
    #     print("Connectors shut down for test.")
    #
    # print("Starting temporary FastAPI server for query router on http://127.0.0.1:8002")
    # print("Try POSTing a JSON payload like {'query': 'Your question here'} to http://127.0.0.1:8002/query/")
    # uvicorn.run(temp_app, host="127.0.0.1", port=8002)

    # Example curl command (ensure FastAPI server is running with this router):
    # curl -X POST -H "Content-Type: application/json" \
    # -d '{"query": "Tell me about Alpha Corp"}' \
    # http://127.0.0.1:8000/api/v1/query/
    # (Adjust URL based on your main app's prefix and port)
    print("Query router defined. Run with a FastAPI application (e.g., app/main.py) to test endpoints.")