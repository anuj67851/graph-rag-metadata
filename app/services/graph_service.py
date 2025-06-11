import logging
from typing import Optional, List, Dict

from app.graph_db.neo4j_connector import get_neo4j_connector
from app.models.common_models import Subgraph

logger = logging.getLogger(__name__)

DEFAULT_BUSIEST_NODES_NEIGHBOR_HOP_DEPTH = 1

async def get_full_graph_sample(node_limit: int, edge_limit: int, filenames: Optional[List[str]] = None) -> Subgraph:
    """
    Retrieves a sample of the full graph by calling the data access layer.
    This service layer function is now database-agnostic.
    """
    # The service layer now just orchestrates the call.
    neo4j_conn = await get_neo4j_connector()
    logger.info(f"Fetching full graph sample via connector (node_limit={node_limit}, edge_limit={edge_limit})")
    return await neo4j_conn.get_full_graph_sample(node_limit=node_limit, edge_limit=edge_limit, filenames=filenames)


async def get_top_n_busiest_nodes(top_n: int = 10, filenames: Optional[List[str]] = None) -> Subgraph:
    """
    Retrieves the busiest nodes and their neighborhood by calling the connector.
    This service is now clean of Cypher.
    """
    neo4j_conn = await get_neo4j_connector()
    logger.info(f"Fetching top {top_n} busiest nodes via connector with filters: {filenames}")

    return await neo4j_conn.get_top_n_busiest_nodes_subgraph(
        top_n=top_n,
        hop_depth=DEFAULT_BUSIEST_NODES_NEIGHBOR_HOP_DEPTH,
        filenames=filenames
    )

async def get_node_neighborhood_subgraph(node_id: str, hop_depth: int) -> Subgraph:
    """
    Retrieves the N-hop neighborhood for a specific entity from the graph.
    """
    logger.info(f"Fetching {hop_depth}-hop neighborhood for node '{node_id}'.")
    neo4j_conn = await get_neo4j_connector()
    return await neo4j_conn.get_subgraph_for_entities([node_id], hop_depth)

async def get_current_graph_schema() -> Dict[str, List[str]]:
    """
    Retrieves the dynamic schema (all node labels and relationship types) from the graph.
    """
    logger.info("Fetching dynamic graph schema from connector.")
    neo4j_conn = await get_neo4j_connector()
    return await neo4j_conn.get_graph_schema()

async def safely_remove_file_references(filename: str):
    """
    Orchestrates the safe removal of file references from the graph
    by making a high-level call to the Neo4j connector.
    This function is now free of any database-specific logic.
    """
    logger.info(f"Service: Orchestrating safe removal of references for file: '{filename}'.")
    neo4j_conn = await get_neo4j_connector()
    await neo4j_conn.safely_remove_file_references(filename)
    logger.info(f"Service: Completed orchestration for safe removal of '{filename}'.")