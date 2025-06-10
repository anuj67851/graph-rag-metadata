import logging

from app.graph_db.neo4j_connector import get_neo4j_connector, Neo4jConnector
from app.models.common_models import Subgraph

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_BUSIEST_NODES_NEIGHBOR_HOP_DEPTH = 1

async def get_full_graph_sample(node_limit: int, edge_limit: int) -> Subgraph:
    """
    Retrieves a sample of the full graph using a robust query.
    It fetches a set of edges and ensures all their source and target nodes are included.
    """
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()
    logger.info(f"Fetching full graph sample (node_limit={node_limit}, edge_limit={edge_limit})")

    query = """
    MATCH (s)-[r]->(t)
    WITH s, r, t
    LIMIT $edge_limit
    // Collect all nodes and relationships from the edge-limited sample
    WITH collect(r) as rels, collect(s) + collect(t) as all_nodes
    UNWIND all_nodes as node
    WITH rels, collect(DISTINCT node) as unique_nodes
    // Return the relationships and the limited list of unique nodes
    RETURN rels, unique_nodes[..$node_limit] as nodes
    """
    params = {"node_limit": node_limit, "edge_limit": edge_limit}
    results = await neo4j_conn.execute_query(query, params)

    if not results or not results[0].get('nodes'):
        return Subgraph()

    # The query returns a single record with keys 'nodes' and 'rels'
    # which contain lists of graph objects.
    # We create a list of single-item records to pass to the robust processor.
    record = results[0]
    nodes_and_rels_records = record.get('nodes', []) + record.get('rels', [])

    return neo4j_conn._process_subgraph_results_full_graph_sample(nodes_and_rels_records)
async def get_top_n_busiest_nodes(top_n: int = 10) -> Subgraph:
    """
    Retrieves the top N busiest (most connected) nodes and their 1-hop neighborhood.
    """
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()
    logger.info(f"Fetching top {top_n} busiest nodes and their neighborhood.")

    query_busiest_nodes = """
    MATCH (n)
    WHERE n.canonical_name IS NOT NULL
    RETURN
      n.canonical_name   AS canonical_name,
      COUNT{(n)-[]-()}          AS degree
    ORDER BY degree DESC
    LIMIT toInteger($top_n)
    """
    params_busiest = {"top_n": top_n}
    busiest_nodes_results = await neo4j_conn.execute_query(query_busiest_nodes, params_busiest)

    if not busiest_nodes_results:
        logger.info("No nodes found or graph is empty.")
        return Subgraph()

    busiest_canonical_names = [record["canonical_name"] for record in busiest_nodes_results]
    if not busiest_canonical_names:
        return Subgraph()

    logger.info(f"Top {len(busiest_canonical_names)} busiest canonical names found: {busiest_canonical_names}")

    subgraph_result = await neo4j_conn.get_subgraph_for_entities(
        canonical_names=busiest_canonical_names,
        hop_depth=DEFAULT_BUSIEST_NODES_NEIGHBOR_HOP_DEPTH
    )

    logger.info(f"Returning subgraph for busiest nodes with {len(subgraph_result.nodes)} nodes and {len(subgraph_result.edges)} edges.")
    return subgraph_result
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