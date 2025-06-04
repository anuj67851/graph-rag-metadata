import logging
from typing import List, Dict

from app.graph_db.neo4j_connector import get_neo4j_connector, Neo4jConnector
from app.models.common_models import Subgraph, Node as PydanticNode, Edge as PydanticEdge
from app.core.config import settings # For potential limits or configurations

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default limits for graph exploration to prevent overwhelming the system/UI
DEFAULT_FULL_GRAPH_NODE_LIMIT = 200
DEFAULT_FULL_GRAPH_EDGE_LIMIT = 400
DEFAULT_BUSIEST_NODES_NEIGHBOR_HOP_DEPTH = 1


async def get_full_graph_sample(
        node_limit: int = DEFAULT_FULL_GRAPH_NODE_LIMIT,
        edge_limit: int = DEFAULT_FULL_GRAPH_EDGE_LIMIT
) -> Subgraph:
    """
    Retrieves a sample of the full graph, limited by a number of nodes and edges.
    This is a simplified representation and might not be a connected graph if limits are hit.
    """
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()
    logger.info(f"Fetching full graph sample (node_limit={node_limit}, edge_limit={edge_limit})")

    # Query to get a sample of nodes and their relationships
    # This is a basic approach. More sophisticated sampling might be needed for very large graphs.
    # We fetch some nodes, and then relationships between *those* nodes.
    query = (
        f"MATCH (n) "
        f"WITH n LIMIT $node_limit " # Get a limited set of nodes first
        f"MATCH (n)-[r]-(m) " # Then get relationships connected to these nodes
        f"WHERE id(n) < id(m) " # Avoid duplicate relationships (direction-agnostic for this fetch)
        # This ensures each pair is considered once. For directed, remove this.
        f"RETURN n, r, m " # Return nodes involved in these relationships and the relationships
        f"LIMIT $edge_limit" # Limit the number of relationships (and consequently, involved nodes)
    )
    # A slightly different approach:
    # query = (
    #     "MATCH (n)-[r]->(m) "
    #     "RETURN n, r, m "
    #     f"LIMIT $edge_limit" # This would prioritize edges, nodes come along
    # )
    # Simpler still, just get N nodes and M edges without trying to connect them initially:
    # query_nodes = "MATCH (n) RETURN n LIMIT $node_limit"
    # query_edges = "MATCH ()-[r]->() RETURN r, startNode(r) as s, endNode(r) as t LIMIT $edge_limit"


    # For this implementation, let's use a query that fetches nodes and then their relationships
    # up to a limit. This will likely give a more connected sample.
    # Using an APOC approach for a more connected sample might be better if available:
    # "MATCH (n) WITH n LIMIT $node_limit CALL apoc.path.subgraphNodes(n, {maxLevel:1}) YIELD nodes, relationships ..."
    # But let's stick to simpler Cypher for broader compatibility first for this "full graph" view.

    # Let's try fetching distinct nodes up to a limit, then relationships among them.
    query_nodes_and_rels = (
        "MATCH (n) "
        "WITH n LIMIT $node_limit " # Sample N nodes
        "OPTIONAL MATCH (n)-[r]-(m) " # Get relationships for these nodes
        "WHERE id(n) < id(m) " # Avoid duplicate relationships if undirected view
        "RETURN DISTINCT n as node1, r as rel, m as node2 " # Return nodes and relationships
        # "LIMIT $edge_limit" # Applying edge limit here is tricky with optional match
        # The overall number of results will be limited by the combination.
    )
    # This query is still not perfect for a "full sample" as edge_limit isn't directly applied.
    # A better way for a limited full graph sample might be to get all nodes and edges
    # and then sample them in Python, or use more complex Cypher with collection and slicing.

    # For now, let's use a query that grabs edges and their nodes, limited by edge count.
    # This tends to give a more "graphy" sample.
    final_query = (
        "MATCH (s)-[r]->(t) " # Get directed relationships and their start/end nodes
        "RETURN s, r, t "
        f"LIMIT $limit" # Limit the number of relationships (edges) returned
    )

    params = {"limit": edge_limit} # Limit is on relationships for this query
    results = await neo4j_conn.execute_query(final_query, params)

    pydantic_nodes_map: Dict[str, PydanticNode] = {}
    pydantic_edges: List[PydanticEdge] = []

    for record in results:
        source_node_proxy = record.get('s')
        target_node_proxy = record.get('t')
        rel_proxy = record.get('r')

        if source_node_proxy:
            s_node = neo4j_conn._convert_neo4j_node_to_pydantic({"n": source_node_proxy}, 'n')
            if s_node and s_node.id not in pydantic_nodes_map:
                pydantic_nodes_map[s_node.id] = s_node

        if target_node_proxy:
            t_node = neo4j_conn._convert_neo4j_node_to_pydantic({"n": target_node_proxy}, 'n')
            if t_node and t_node.id not in pydantic_nodes_map:
                pydantic_nodes_map[t_node.id] = t_node

        if rel_proxy and source_node_proxy and target_node_proxy:
            # For PydanticEdge conversion, we need canonical names of source/target
            # The _convert_neo4j_relationship_to_pydantic expects these in the record if not using node objects
            # Here we have the node objects, so we can extract their IDs (canonical_names)
            s_canonical = source_node_proxy.get("canonical_name", str(source_node_proxy.id))
            t_canonical = target_node_proxy.get("canonical_name", str(target_node_proxy.id))

            edge_record_like = {
                "r": rel_proxy,
                "s_canonical_name": s_canonical,
                "t_canonical_name": t_canonical
            }
            edge = neo4j_conn._convert_neo4j_relationship_to_pydantic(edge_record_like, 'r')
            if edge:
                pydantic_edges.append(edge)

    # Further node limit enforcement after collecting all unique nodes from edges
    final_nodes = list(pydantic_nodes_map.values())
    if len(final_nodes) > node_limit:
        logger.warning(f"Full graph sample node count ({len(final_nodes)}) exceeded limit ({node_limit}). Truncating node list.")
        # This truncation might leave some edges pointing to non-existent nodes in the truncated list.
        # Proper sampling is complex. For UI, this might be acceptable.
        final_nodes = final_nodes[:node_limit]
        # Optionally, filter edges to only include those between the final_nodes
        final_node_ids = {node.id for node in final_nodes}
        pydantic_edges = [edge for edge in pydantic_edges if edge.source in final_node_ids and edge.target in final_node_ids]


    logger.info(f"Returning full graph sample with {len(final_nodes)} nodes and {len(pydantic_edges)} edges.")
    return Subgraph(nodes=final_nodes, edges=pydantic_edges)


async def get_top_n_busiest_nodes(top_n: int = 10) -> Subgraph:
    """
    Retrieves the top N busiest (most connected) nodes and their 1-hop neighborhood.
    """
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()
    logger.info(f"Fetching top {top_n} busiest nodes and their neighborhood.")

    # 1. Find top N busiest nodes (highest degree)
    # Degree calculation: size((n)---()) counts all relationships (incoming and outgoing).
    # Use (n)-[]->() for outgoing, ()-[]->(n) for incoming if specific direction is wanted.
    query_busiest_nodes = (
        "MATCH (n) "
        "WITH n, size((n)--()) AS degree " # Calculate degree
        "ORDER BY degree DESC "
        f"LIMIT $top_n "
        "RETURN n.canonical_name AS canonical_name, degree" # Return canonical name and degree
    )
    params_busiest = {"top_n": top_n}
    busiest_nodes_results = await neo4j_conn.execute_query(query_busiest_nodes, params_busiest)

    if not busiest_nodes_results:
        logger.info("No nodes found or graph is empty.")
        return Subgraph()

    busiest_canonical_names = [record["canonical_name"] for record in busiest_nodes_results if record.get("canonical_name")]
    logger.info(f"Top {len(busiest_canonical_names)} busiest canonical names found: {busiest_canonical_names}")

    if not busiest_canonical_names:
        return Subgraph()

    # 2. For these busiest nodes, get their 1-hop subgraph using the existing method.
    # The `get_subgraph_for_entities` uses APOC by default for N-hop.
    # If APOC is not used, the alternative N-hop query in neo4j_connector will be used.
    subgraph_result = await neo4j_conn.get_subgraph_for_entities(
        canonical_names=busiest_canonical_names,
        hop_depth=DEFAULT_BUSIEST_NODES_NEIGHBOR_HOP_DEPTH # Typically 1-hop for "busiest nodes" view
    )

    logger.info(f"Returning subgraph for busiest nodes with {len(subgraph_result.nodes)} nodes and {len(subgraph_result.edges)} edges.")
    return subgraph_result


if __name__ == "__main__":
    import asyncio

    async def test_graph_service():
        print("--- Testing Graph Service ---")
        # Requires Neo4j instance with some data.
        # Run ingestion_service test first to populate.

        if not settings.NEO4J_URI:
            print("NEO4J_URI not set. Skipping graph service test.")
            return

        try:
            # Test 1: Get Full Graph Sample
            print("\n1. Testing Get Full Graph Sample...")
            full_graph_sample = await get_full_graph_sample(node_limit=50, edge_limit=100) # Smaller limits for test
            if not full_graph_sample.is_empty():
                print(f"   Full graph sample: {len(full_graph_sample.nodes)} nodes, {len(full_graph_sample.edges)} edges.")
                # print("   Sample Nodes (first 5):")
                # for node in full_graph_sample.nodes[:5]:
                #     print(f"     - {node.id} ({node.type})")
            else:
                print("   Full graph sample is empty (or graph has no data).")

            # Test 2: Get Top N Busiest Nodes
            print("\n2. Testing Get Top N Busiest Nodes (N=5)...")
            busiest_nodes_subgraph = await get_top_n_busiest_nodes(top_n=5)
            if not busiest_nodes_subgraph.is_empty():
                print(f"   Busiest nodes subgraph: {len(busiest_nodes_subgraph.nodes)} nodes, {len(busiest_nodes_subgraph.edges)} edges.")
                print("   Central Busiest Nodes (IDs found in subgraph):")
                # The subgraph will contain more than N nodes due to 1-hop neighbors.
                # We'd need to cross-reference with the busiest_canonical_names if we want to list only the N.
                # For now, just showing counts.
                # for node in busiest_nodes_subgraph.nodes: # Example of printing some details
                #     if node.properties.get('degree') is not None: # Degree not added by current get_subgraph
                #         print(f"     - {node.id} (Degree: {node.properties.get('degree')})")
            else:
                print("   Busiest nodes subgraph is empty (or graph has no data).")

        except Exception as e:
            print(f"Error during graph service test: {e}", exc_info=True)
        finally:
            try:
                neo4j_conn_main = await get_neo4j_connector()
                await neo4j_conn_main.close_driver()
            except Exception as e_close:
                print(f"Error closing Neo4j driver in test: {e_close}")

        print("\n--- Graph Service Test Finished ---")

    asyncio.run(test_graph_service())