import logging
from typing import List, Dict, Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.graph import Node as Neo4jNodeObject, Relationship as Neo4jRelationshipObject
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from app.core.config import settings
from app.models.common_models import Node as PydanticNode, Edge as PydanticEdge, Subgraph

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Neo4jConnector:
    _driver: Optional[AsyncDriver] = None

    async def initialize_driver(self):
        """Establishes and verifies the connection to the Neo4j database."""
        if self._driver is not None:
            try:
                await self._driver.verify_connectivity()
                logger.debug("Existing Neo4j driver connectivity verified.")
                return
            except (ServiceUnavailable, Neo4jError) as e:
                logger.warning(f"Existing Neo4j driver failed connectivity: {e}. Re-initializing.")
                await self.close_driver()
        try:
            logger.info(f"Initializing Neo4j driver at {settings.NEO4J_URI}")
            self._driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
            await self._driver.verify_connectivity()
            logger.info("Neo4j driver initialized and connected successfully.")
        except (ServiceUnavailable, Neo4jError) as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}", exc_info=True)
            self._driver = None
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Neo4j driver initialization: {e}", exc_info=True)
            self._driver = None
            raise

    async def _get_driver(self) -> AsyncDriver:
        if self._driver is None:
            await self.initialize_driver()
        if self._driver is None: # Check again after trying to initialize
            raise ServiceUnavailable("Neo4j driver is not available and initialization failed.")
        return self._driver

    async def close_driver(self):
        """Closes the Neo4j driver connection."""
        if self._driver:
            logger.info("Closing Neo4j driver.")
            await self._driver.close()
            self._driver = None

    async def _ensure_constraints(self):
        """Ensures unique constraints on node canonical names are set for defined entity types."""
        driver = await self._get_driver()
        entity_types = settings.SCHEMA.ENTITY_TYPES
        if not entity_types:
            logger.warning("No entity types found in schema. Skipping constraint creation.")
            return
        async with driver.session() as session:
            for entity_type in entity_types:
                constraint_name = f"constraint_unique_{entity_type.lower()}_canonical_name"
                query = f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{entity_type}) REQUIRE n.canonical_name IS UNIQUE"
                try:
                    logger.debug(f"Ensuring constraint: {query}")
                    await session.run(query)
                except Exception as e:
                    logger.error(f"Error ensuring constraint for {entity_type} ({constraint_name}): {e}")
            logger.info("Finished ensuring constraints on node labels.")

    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Executes a Cypher query and returns the results."""
        driver = await self._get_driver()
        try:
            async with driver.session() as session:
                logger.debug(f"Executing Cypher: {query} with params: {parameters}")
                result = await session.run(query, parameters)
                records_list = [record.data() async for record in result]
                await result.consume()
                return records_list
        except Exception as e:
            logger.error(f"Error during Cypher query execution: {e}\nQuery: {query}\nParams: {parameters}", exc_info=True)
            return [] # Return empty list on failure

    async def merge_entity(self, entity_type: str, canonical_name: str, properties: Dict[str, Any]) -> bool:
        """Merges a node into the graph, creating or updating it."""
        props = {"canonical_name": canonical_name, **properties}
        query = f"""
        MERGE (n:{entity_type} {{canonical_name: $canonical_name}})
        ON CREATE SET n = $props
        ON MATCH SET n += $props
        """
        try:
            await self.execute_query(query, {"canonical_name": canonical_name, "props": props})
            logger.info(f"Merged entity: {entity_type} - {canonical_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to merge entity {entity_type} - {canonical_name}: {e}")
            return False

    async def merge_relationship(self, s_type: str, s_name: str, t_type: str, t_name: str, r_type: str, props: Dict[str, Any]) -> bool:
        """Merges a relationship between two nodes."""
        query = f"""
        MATCH (s:{s_type} {{canonical_name: $s_name}}), (t:{t_type} {{canonical_name: $t_name}})
        MERGE (s)-[r:{r_type}]->(t)
        ON CREATE SET r = $props
        ON MATCH SET r += $props
        """
        params = {"s_name": s_name, "t_name": t_name, "props": props}
        try:
            await self.execute_query(query, params)
            logger.info(f"Merged relationship: ({s_name})-[{r_type}]->({t_name})")
            return True
        except Exception as e:
            logger.error(f"Failed to merge relationship ({s_name})-[{r_type}]->({t_name}): {e}")
            return False

    def _convert_neo4j_node_to_pydantic(self, node: Neo4jNodeObject) -> PydanticNode:
        """Converts a Neo4j Node object to a Pydantic Node model."""
        props = dict(node)
        # Prioritize defined schema types, fallback to first label.
        primary_type = "Unknown"
        node_labels = list(node.labels)
        for label in node_labels:
            if label in settings.SCHEMA.ENTITY_TYPES:
                primary_type = label
                break
        if primary_type == "Unknown" and node_labels:
            primary_type = node_labels[0]

        node_id = props.get("canonical_name", str(node.element_id))

        return PydanticNode(
            id=node_id,
            label=props.get("canonical_name", node_id),
            type=primary_type,
            properties=props
        )

    def _convert_neo4j_relationship_to_pydantic(self, rel: Neo4jRelationshipObject, nodes_map: Dict[str, PydanticNode]) -> Optional[PydanticEdge]:
        """Converts a Neo4j Relationship object to a Pydantic Edge model."""
        source_id = str(rel.start_node.element_id)
        target_id = str(rel.end_node.element_id)

        # We need the canonical_name of the start and end nodes for the Edge model.
        # We find them in the nodes_map we built from all nodes in the subgraph.
        source_node_model = nodes_map.get(source_id)
        target_node_model = nodes_map.get(target_id)

        if not source_node_model or not target_node_model:
            logger.warning(f"Could not find start/end node in map for relationship {rel.type}. Skipping edge.")
            return None

        return PydanticEdge(
            source=source_node_model.id, # Use canonical_name from the Pydantic model
            target=target_node_model.id, # Use canonical_name from the Pydantic model
            label=rel.type,
            properties=dict(rel)
        )

    def _process_subgraph_results(self, records: List[Dict[str, Any]]) -> Subgraph:
        """Processes raw query results (nodes and relationships) into a Subgraph object."""
        pydantic_nodes_map: Dict[str, PydanticNode] = {} # Key: element_id
        pydantic_edges: List[PydanticEdge] = []

        # First pass: collect all unique nodes
        for record in records:
            # A record can contain a node, a relationship, or both.
            for value in record.values():
                if isinstance(value, Neo4jNodeObject):
                    node_element_id = str(value.element_id)
                    if node_element_id not in pydantic_nodes_map:
                        pydantic_nodes_map[node_element_id] = self._convert_neo4j_node_to_pydantic(value)

        # Second pass: collect all relationships, using the node map for source/target info
        for record in records:
            for value in record.values():
                if isinstance(value, Neo4jRelationshipObject):
                    edge = self._convert_neo4j_relationship_to_pydantic(value, pydantic_nodes_map)
                    if edge:
                        pydantic_edges.append(edge)

        # Remove duplicate edges (can happen in path queries)
        unique_edges = list({(e.source, e.target, e.label): e for e in pydantic_edges}.values())

        return Subgraph(nodes=list(pydantic_nodes_map.values()), edges=unique_edges)

    async def get_subgraph_for_entities(self, canonical_names: List[str], hop_depth: int = 1) -> Subgraph:
        """Retrieves an N-hop subgraph for a list of seed entities."""
        if not canonical_names:
            return Subgraph()

        # This query finds the seed nodes, traverses N-hops, and returns all nodes and relationships found.
        # It's robust and avoids complex UNIONs.
        query = f"""
        MATCH (seed) WHERE seed.canonical_name IN $names
        CALL {{
            WITH seed
            MATCH path = (seed)-[*0..{hop_depth}]-(connected_node)
            UNWIND nodes(path) AS n
            UNWIND relationships(path) AS r
            RETURN collect(DISTINCT n) AS nodes, collect(DISTINCT r) AS rels
        }}
        RETURN nodes, rels
        """
        params = {"names": canonical_names}
        results = await self.execute_query(query, params)

        # The result will be one record per seed node, with lists of nodes/rels.
        # We need to flatten and combine them.
        all_nodes = []
        all_rels = []
        for record in results:
            all_nodes.extend(record.get('nodes', []))
            all_rels.extend(record.get('rels', []))

        # Re-package for the standard processor
        final_records = [{"nodes": all_nodes, "rels": all_rels}]
        return self._process_subgraph_results(final_records)

    async def find_shortest_paths(self, start_node_name: str, end_node_name: str, max_hops: int = 3) -> Subgraph:
        """Finds all shortest paths between two entities and returns the combined subgraph."""
        query = f"""
        MATCH (start {{canonical_name: $start_name}}), (end {{canonical_name: $end_name}})
        MATCH path = allShortestPaths((start)-[*1..{max_hops}]-(end))
        WITH nodes(path) as path_nodes, relationships(path) as path_rels
        UNWIND path_nodes as n
        UNWIND path_rels as r
        RETURN collect(DISTINCT n) as nodes, collect(DISTINCT r) as rels
        """
        params = {"start_name": start_node_name, "end_name": end_node_name}
        results = await self.execute_query(query, params)
        return self._process_subgraph_results(results)


# Global instance management
neo4j_connector_instance: Optional[Neo4jConnector] = None

async def get_neo4j_connector() -> Neo4jConnector:
    """Provides the singleton Neo4jConnector instance."""
    global neo4j_connector_instance
    if neo4j_connector_instance is None:
        neo4j_connector_instance = Neo4jConnector()
    return neo4j_connector_instance

async def init_neo4j_driver():
    """Initializes the driver and ensures constraints on startup."""
    try:
        connector = await get_neo4j_connector()
        await connector.initialize_driver()
        await connector._ensure_constraints()
    except Exception as e:
        logger.critical(f"Critical failure during Neo4j initialization: {e}", exc_info=True)
        # In a real app, you might want this to prevent startup.

async def close_neo4j_driver():
    """Closes the driver on shutdown."""
    connector = await get_neo4j_connector()
    if connector:
        await connector.close_driver()

if __name__ == "__main__":
    import asyncio

    async def test_neo4j_connector():
        print("--- Testing Neo4j Connector (Refined) ---")
        connector = await get_neo4j_connector()

        # Test data
        await connector.merge_entity("PERSON", "Alice", {"role": "Engineer"})
        await connector.merge_entity("ORGANIZATION", "ACME Corp", {"industry": "Software"})
        await connector.merge_relationship("PERSON", "Alice", "ORGANIZATION", "ACME Corp", "WORKS_FOR", {"year": 2024})

        print("\n1. Testing Get Subgraph for 'Alice'...")
        subgraph = await connector.get_subgraph_for_entities(["Alice"], hop_depth=1)
        if subgraph and not subgraph.is_empty():
            print(f"   Subgraph Nodes: {[node.id for node in subgraph.nodes]}")
            print(f"   Subgraph Edges Count: {len(subgraph.edges)}")
            for edge in subgraph.edges:
                print(f"     Edge: ({edge.source})-[:{edge.label}]->({edge.target})")
        else:
            print("   Subgraph for 'Alice' is empty or retrieval failed.")

        # Clean up test data
        await connector.execute_query("MATCH (n {canonical_name: 'Alice'}) DETACH DELETE n")
        await connector.execute_query("MATCH (n {canonical_name: 'ACME Corp'}) DETACH DELETE n")
        print("\nCleaned up test data.")

        await connector.close_driver()
        print("\n--- Neo4j Connector Test Finished ---")

    if settings.NEO4J_URI:
        asyncio.run(test_neo4j_connector())
    else:
        print("Skipping Neo4j connector tests: NEO4J_URI not set.")