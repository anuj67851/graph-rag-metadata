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

    return neo4j_conn._process_subgraph_results(nodes_and_rels_records)


async def get_top_n_busiest_nodes(top_n: int = 10) -> Subgraph:
    """
    Retrieves the top N busiest (most connected) nodes and their 1-hop neighborhood.
    """
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()
    logger.info(f"Fetching top {top_n} busiest nodes and their neighborhood.")

    query_busiest_nodes = """
    MATCH (n)
    WHERE n.canonical_name IS NOT NULL
    WITH n, size((n)--()) as degree
    ORDER BY degree DESC
    LIMIT $top_n
    RETURN n.canonical_name AS canonical_name
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