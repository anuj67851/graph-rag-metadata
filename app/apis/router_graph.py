import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query as FastAPIQuery, status # Renamed Query to avoid conflict
from app.services.graph_service import get_full_graph_sample, get_top_n_busiest_nodes
from app.models.common_models import Subgraph
from app.core.config import settings # For API prefix, default limits, etc.

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Create an APIRouter instance for graph exploration-related endpoints
router = APIRouter(
    # prefix=f"{settings.API_V1_STR}/graph", # Using API_V1_STR from settings
    prefix="/graph", # Simpler prefix
    tags=["Graph Exploration"],
    responses={404: {"description": "Not found"}},
)

@router.get(
    "/full_sample",
    response_model=Subgraph,
    summary="Get a sample of the full knowledge graph.",
    description=(
            "Retrieves a limited sample of nodes and edges from the entire graph. "
            "Useful for getting a general overview or for visualization of a subset. "
            "The sample size is controlled by 'node_limit' and 'edge_limit' query parameters."
    )
)
async def get_full_graph_sample_endpoint(
        node_limit: Optional[int] = FastAPIQuery(
            default=settings.DEFAULT_FULL_GRAPH_NODE_LIMIT if hasattr(settings, 'DEFAULT_FULL_GRAPH_NODE_LIMIT') else 200,
            description="Maximum number of nodes to return in the sample.",
            ge=10, # Minimum reasonable limit
            le=1000 # Maximum reasonable limit to prevent overload
        ),
        edge_limit: Optional[int] = FastAPIQuery(
            default=settings.DEFAULT_FULL_GRAPH_EDGE_LIMIT if hasattr(settings, 'DEFAULT_FULL_GRAPH_EDGE_LIMIT') else 400,
            description="Maximum number of edges to return in the sample.",
            ge=10,
            le=2000
        )
):
    """
    Endpoint to fetch a sample of the full graph.
    Uses default limits from settings if available, otherwise hardcoded defaults here.
    """
    logger.info(f"Request received for full graph sample with node_limit={node_limit}, edge_limit={edge_limit}")
    try:
        subgraph_data = await get_full_graph_sample(node_limit=node_limit, edge_limit=edge_limit)
        if subgraph_data.is_empty():
            logger.info("Full graph sample query returned no data (graph might be empty or limits too restrictive).")
            # Return empty subgraph, not necessarily an error
        return subgraph_data
    except Exception as e:
        logger.error(f"Error fetching full graph sample: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching the graph sample: {str(e)}"
        )

@router.get(
    "/busiest_nodes",
    response_model=Subgraph,
    summary="Get the top N busiest (most connected) nodes and their 1-hop neighborhood.",
    description=(
            "Identifies the N nodes with the highest degree (most relationships) "
            "and returns a subgraph containing these nodes and their immediate neighbors."
    )
)
async def get_busiest_nodes_endpoint(
        top_n: Optional[int] = FastAPIQuery(
            default=10, # A common default for "top N" lists
            description="Number of busiest nodes to identify.",
            ge=1,
            le=50 # Max limit for busiest nodes to keep subgraph manageable
        )
):
    """
    Endpoint to fetch the busiest nodes and their local neighborhood.
    """
    logger.info(f"Request received for top {top_n} busiest nodes.")
    try:
        subgraph_data = await get_top_n_busiest_nodes(top_n=top_n)
        if subgraph_data.is_empty():
            logger.info("Busiest nodes query returned no data (graph might be empty).")
            # Return empty subgraph
        return subgraph_data
    except Exception as e:
        logger.error(f"Error fetching busiest nodes: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching busiest nodes: {str(e)}"
        )

if __name__ == "__main__":
    # This router is meant to be run by Uvicorn as part of a FastAPI app.
    # Example of how to run this router for quick manual testing:
    # from fastapi import FastAPI
    # import uvicorn
    #
    # temp_app = FastAPI(title="Graph Router Test API")
    # temp_app.include_router(router)
    #
    # # For this test to be meaningful, you need:
    # # 1. Neo4j service and connector to be functional.
    # # 2. Neo4j to be populated with data (e.g., by running ingestion_service).
    # # 3. .env, config.yaml for Neo4j connection.
    #
    # @temp_app.on_event("startup")
    # async def startup_event():
    #     from app.graph_db.neo4j_connector import init_neo4j_driver
    #     await init_neo4j_driver()
    #     print("Neo4j connector initialized for graph router test.")
    #
    # @temp_app.on_event("shutdown")
    # async def shutdown_event():
    #     from app.graph_db.neo4j_connector import close_neo4j_driver
    #     await close_neo_j_driver()
    #     print("Neo4j connector shut down for graph router test.")
    #
    # print("Starting temporary FastAPI server for graph router on http://127.0.0.1:8003")
    # print("Try GET requests like:")
    # print("  http://127.0.0.1:8003/graph/full_sample?node_limit=30&edge_limit=50")
    # print("  http://127.0.0.1:8003/graph/busiest_nodes?top_n=5")
    # uvicorn.run(temp_app, host="127.0.0.1", port=8003)

    # Example curl commands (ensure FastAPI server is running with this router):
    # curl -X GET "http://127.0.0.1:8000/api/v1/graph/full_sample?node_limit=20&edge_limit=30"
    # curl -X GET "http://127.0.0.1:8000/api/v1/graph/busiest_nodes?top_n=3"
    # (Adjust URL based on your main app's prefix and port)
    print("Graph exploration router defined. Run with a FastAPI application (e.g., app/main.py) to test endpoints.")