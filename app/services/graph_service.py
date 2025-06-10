import logging
from typing import Optional, List

from app.graph_db.neo4j_connector import get_neo4j_connector, Neo4jConnector
from app.models.common_models import Subgraph

logger = logging.getLogger(__name__)

DEFAULT_BUSIEST_NODES_NEIGHBOR_HOP_DEPTH = 1

async def get_full_graph_sample(node_limit: int, edge_limit: int, filenames: Optional[List[str]] = None) -> Subgraph:
    """
    Retrieves a sample of the full graph using a robust query.
    """
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()
    logger.info(f"Fetching full graph sample (node_limit={node_limit}, edge_limit={edge_limit})")

    match_clause = "MATCH (n)"
    if filenames:
        match_clause = "MATCH (n) WHERE any(file IN n.source_document_filename WHERE file IN $filenames)"

    # This query first finds a set of edges, then returns those edges
    # along with their start and end nodes. This is a common sampling strategy.
    query = f"""
    {match_clause}
    WITH n LIMIT $node_limit
    MATCH (n)-[r]-(m)
    RETURN s, t, r
    LIMIT $edge_limit
    """
    params = {"node_limit": node_limit, "edge_limit": edge_limit, "filenames": filenames}
    records = await neo4j_conn.execute_query(query, params)

    return neo4j_conn._process_records_to_subgraph(records)


async def get_top_n_busiest_nodes(top_n: int = 10, filenames: Optional[List[str]] = None) -> Subgraph:
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()
    logger.info(f"Fetching top {top_n} busiest nodes with filters: {filenames}")

    # Build the MATCH clause conditionally
    match_clause = "MATCH (n)"
    if filenames:
        match_clause = "MATCH (n) WHERE any(file IN n.source_document_filename WHERE file IN $filenames)"

    query_busiest_nodes = f"""
    {match_clause}
    WITH n
    WHERE n.canonical_name IS NOT NULL
    RETURN n.canonical_name AS canonical_name, size((n)--()) AS degree
    ORDER BY degree DESC
    LIMIT toInteger($top_n)
    """
    busiest_nodes_results = await neo4j_conn.execute_query(query_busiest_nodes, {"top_n": top_n, "filenames": filenames})

    if not busiest_nodes_results:
        return Subgraph()

    busiest_canonical_names = [record["canonical_name"] for record in busiest_nodes_results]

    return await neo4j_conn.get_subgraph_for_entities(
        canonical_names=busiest_canonical_names,
        hop_depth=DEFAULT_BUSIEST_NODES_NEIGHBOR_HOP_DEPTH
    )

async def safely_remove_file_references(filename: str):
    """
    Finds nodes/relationships referencing a file and safely removes them.
    If a node/relationship is only referenced by this file, it's deleted.
    Otherwise, just the filename reference is removed from its property list.
    """
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()
    logger.info(f"Initiating safe removal of references for file: '{filename}' in Neo4j.")
    query = """
    // Process Nodes first
    MATCH (n) WHERE $filename IN n.source_document_filename
    WITH n, size(n.source_document_filename) AS source_count
    // If this file is the only source, detach and delete the node
    FOREACH (_ IN CASE WHEN source_count = 1 THEN [1] ELSE [] END |
        DETACH DELETE n
    )
    // If there are other sources, just remove the filename from the list
    FOREACH (_ IN CASE WHEN source_count > 1 THEN [1] ELSE [] END |
        SET n.source_document_filename = [file IN n.source_document_filename WHERE file <> $filename]
    )

    // Then, process Relationships
    WITH 'nodes done' as marker
    MATCH ()-[r]-() WHERE $filename IN r.source_document_filename
    WITH r, size(r.source_document_filename) AS source_count
    // If this file is the only source, delete the relationship
    FOREACH (_ IN CASE WHEN source_count = 1 THEN [1] ELSE [] END |
        DELETE r
    )
    // If there are other sources, remove the filename from the list
    FOREACH (_ IN CASE WHEN source_count > 1 THEN [1] ELSE [] END |
        SET r.source_document_filename = [file IN r.source_document_filename WHERE file <> $filename]
    )
    """
    params = {"filename": filename}
    await neo4j_conn.execute_query(query, params)
    logger.info(f"Neo4j safe removal process completed for '{filename}'.")