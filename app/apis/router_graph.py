import logging
from typing import Optional, List, Dict

from fastapi import APIRouter, HTTPException, Query
from app.models.common_models import Subgraph
from app.core.config import settings
from app.services.graph_service import (
    get_full_graph_sample,
    get_top_n_busiest_nodes,
    get_node_neighborhood_subgraph,
    get_current_graph_schema,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/graph",
    tags=["Graph Exploration"],
    responses={
        404: {"description": "Resource not found."},
        500: {"description": "Internal Server Error."}
    }
)

@router.get(
    "/full_sample",
    response_model=Subgraph,
    summary="Get a sample of the full knowledge graph.",
    response_description="A Subgraph object containing a sample of nodes and edges."
)
async def get_full_graph_sample_endpoint(
        node_limit: int = Query(
            default=settings.DEFAULT_FULL_GRAPH_NODE_LIMIT,
            description="Maximum number of nodes to return in the sample.",
            ge=10, le=1000
        ),
        edge_limit: int = Query(
            default=settings.DEFAULT_FULL_GRAPH_EDGE_LIMIT,
            description="Maximum number of edges to return in the sample.",
            ge=10, le=2000
        ),
        filenames: Optional[List[str]] = Query(None, alias="filenames", description="Optional list of source documents to filter the graph by.")
):
    """
    Retrieves a limited, random sample of nodes and edges from the entire graph,
    optionally filtered by source documents. This is useful for getting a general
    overview or for visualizing a subset of the knowledge base.
    """
    try:
        subgraph_data = await get_full_graph_sample(node_limit=node_limit, edge_limit=edge_limit, filenames=filenames)
        return subgraph_data
    except Exception as e:
        logger.error(f"Error fetching full graph sample: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching the graph sample: {e}")

@router.get(
    "/busiest_nodes",
    response_model=Subgraph,
    summary="Get the most connected nodes (and their neighbors).",
    response_description="A Subgraph object centered around the busiest nodes."
)
async def get_busiest_nodes_endpoint(
        top_n: int = Query(default=10, ge=1, le=50, description="The number of busiest nodes to identify."),
        filenames: Optional[List[str]] = Query(None, alias="filenames", description="Optional list of source documents to filter the graph by.")
):
    """
    Identifies the 'top_n' nodes with the highest degree (most relationships)
    and returns a subgraph containing these nodes and their immediate neighbors (1-hop).
    This helps to quickly find the central entities in the graph.
    """
    try:
        subgraph_data = await get_top_n_busiest_nodes(top_n=top_n, filenames=filenames)
        return subgraph_data
    except Exception as e:
        logger.error(f"Error fetching busiest nodes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching busiest nodes: {e}")

@router.get(
    "/node/{node_id}",
    response_model=Subgraph,
    summary="Explore the neighborhood of a specific node.",
    response_description="A Subgraph object centered around the requested node."
)
async def get_node_neighborhood(
        node_id: str,
        hop_depth: int = Query(default=1, ge=0, le=3, description="How many relationship 'hops' to explore out from the central node.")
):
    """
    Retrieves a subgraph centered on a specific node, identified by its unique
    canonical name (e.g., "Project Chimera"). This allows for targeted exploration
    of the graph starting from a known entity.
    """
    try:
        subgraph_data = await get_node_neighborhood_subgraph(node_id, hop_depth)
        if subgraph_data.is_empty():
            raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found in graph.")
        return subgraph_data
    except Exception as e:
        logger.error(f"Error fetching node neighborhood for '{node_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching the node neighborhood: {e}")

@router.get(
    "/schema",
    response_model=Dict[str, List[str]],
    summary="Discover the graph's dynamic schema.",
    response_description="A dictionary containing lists of all unique node labels and relationship types."
)
async def get_dynamic_graph_schema():
    """
    Inspects the current graph and returns all unique node labels (entity types)
    and relationship types that are actually present in the database.
    This is useful for dynamically populating UI filters or understanding the
    graph's content.
    """
    try:
        return await get_current_graph_schema()
    except Exception as e:
        logger.error(f"Error fetching graph schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while fetching the graph schema.")