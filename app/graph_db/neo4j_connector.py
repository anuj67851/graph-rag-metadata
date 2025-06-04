import logging
from typing import List, Dict, Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from app.core.config import settings
from app.models.common_models import Node as PydanticNode, Edge as PydanticEdge, Subgraph

logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Ensure logger is configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Neo4jConnector:
    _driver: Optional[AsyncDriver] = None

    async def initialize_driver(self):
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
            driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
            await driver.verify_connectivity()
            self._driver = driver
            logger.info("Neo4j driver initialized and connected successfully.")
        except (ServiceUnavailable, Neo4jError) as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            self._driver = None
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Neo4j driver initialization: {e}")
            self._driver = None
            raise

    async def _get_driver(self) -> AsyncDriver:
        if self._driver is None:
            logger.warning("Neo4j driver is None. Attempting to initialize now.")
            await self.initialize_driver()
            if self._driver is None:
                raise ServiceUnavailable("Neo4j driver is not available. Initialization failed.")
        return self._driver

    async def close_driver(self):
        if self._driver is not None:
            logger.info("Closing Neo4j driver.")
            await self._driver.close()
            self._driver = None

    async def _ensure_constraints(self):
        driver = await self._get_driver()
        entity_types = settings.SCHEMA.ENTITY_TYPES
        if not entity_types:
            logger.warning("No entity types defined in schema. Skipping constraint creation.")
            return

        async with driver.session() as session:
            for entity_type in entity_types:
                constraint_name = f"constraint_unique_{entity_type.lower()}_canonical_name"
                query = (
                    f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                    f"FOR (n:{entity_type}) REQUIRE n.canonical_name IS UNIQUE"
                )
                try:
                    logger.debug(f"Ensuring constraint for {entity_type} with query: {query}")
                    await session.run(query)
                except Exception as e: # Catch broad exceptions for constraint creation
                    logger.error(f"Error ensuring constraint for {entity_type} (Name: {constraint_name}): {e}")
            logger.info("Finished ensuring constraints for defined entity types.")

    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        driver = await self._get_driver()
        records_list = []
        try:
            async with driver.session() as session:
                logger.debug(f"Executing Cypher: {query} with params: {parameters}")
                result = await session.run(query, parameters)
                async for record in result:
                    records_list.append(record.data())
                await result.consume()
        except Exception as e: # Catch broad exceptions for query execution
            logger.error(f"Error during query execution: {e}\nQuery: {query}\nParams: {parameters}")
            # Optionally re-raise or return empty based on error handling strategy
        return records_list

    async def merge_entity(self, entity_type: str, canonical_name: str, properties: Dict[str, Any]) -> bool:
        full_properties = {"canonical_name": canonical_name, **properties}
        query = (
            f"MERGE (n:{entity_type} {{canonical_name: $canonical_name}}) "
            "ON CREATE SET n = $props "
            "ON MATCH SET n += $props"
        )
        params = {"canonical_name": canonical_name, "props": full_properties}
        try:
            await self.execute_query(query, params)
            logger.info(f"Merged entity: {entity_type} - {canonical_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to merge entity {entity_type} - {canonical_name}: {e}")
            return False

    async def merge_relationship(
            self,
            source_entity_type: str,
            source_canonical_name: str,
            target_entity_type: str,
            target_canonical_name: str,
            relationship_type: str,
            properties: Dict[str, Any]
    ) -> bool:
        query = (
            f"MATCH (s:{source_entity_type} {{canonical_name: $source_name}}), "
            f"(t:{target_entity_type} {{canonical_name: $target_name}}) "
            f"MERGE (s)-[r:{relationship_type}]->(t) "
            "ON CREATE SET r = $props "
            "ON MATCH SET r += $props"
        )
        params = {
            "source_name": source_canonical_name,
            "target_name": target_canonical_name,
            "props": properties
        }
        try:
            await self.execute_query(query, params)
            logger.info(f"Merged relationship: ({source_canonical_name})-[{relationship_type}]->({target_canonical_name})")
            return True
        except Exception as e:
            logger.error(f"Failed to merge relationship ({source_canonical_name})-[{relationship_type}]->({target_canonical_name}): {e}")
            return False

    def _convert_neo4j_node_to_pydantic(self, neo4j_node_proxy, node_alias: str = 'n') -> Optional[PydanticNode]:
        try:
            node_data = neo4j_node_proxy[node_alias]
            labels = list(node_data.labels)
            primary_type = "Unknown"
            if labels:
                primary_type = labels[0]
                for label in labels:
                    if label in settings.SCHEMA.ENTITY_TYPES:
                        primary_type = label
                        break
            props = dict(node_data)
            node_id = props.get("canonical_name", str(node_data.id))
            return PydanticNode(id=node_id, label=props.get("canonical_name", node_id), type=primary_type, properties=props)
        except Exception as e:
            logger.error(f"Error converting Neo4j node to Pydantic model: {e}. Node data: {neo4j_node_proxy}")
            return None

    def _convert_neo4j_relationship_to_pydantic(self, neo4j_rel_proxy, rel_alias: str = 'r') -> Optional[PydanticEdge]:
        try:
            rel_data = neo4j_rel_proxy[rel_alias]
            source_id = neo4j_rel_proxy.get("s_canonical_name", str(rel_data.start_node.id if rel_data.start_node else "UNKNOWN_SOURCE"))
            target_id = neo4j_rel_proxy.get("t_canonical_name", str(rel_data.end_node.id if rel_data.end_node else "UNKNOWN_TARGET"))
            return PydanticEdge(source=source_id, target=target_id, label=rel_data.type, properties=dict(rel_data))
        except Exception as e:
            logger.error(f"Error converting Neo4j relationship to Pydantic model: {e}. Rel data: {neo4j_rel_proxy}")
            return None

    async def get_subgraph_for_entities(self, canonical_names: List[str], hop_depth: int = 1) -> Subgraph:
        if not canonical_names: return Subgraph()
        query = (
            f"MATCH (center) WHERE center.canonical_name IN $names "
            f"CALL apoc.path.subgraphAll(center, {{maxLevel: $hops, relationshipFilter: '>'}}) YIELD nodes, relationships "
            f"UNWIND nodes AS n UNWIND relationships AS r "
            f"RETURN DISTINCT n, r, startNode(r).canonical_name AS s_canonical_name, endNode(r).canonical_name AS t_canonical_name"
        )
        params = {"names": canonical_names, "hops": hop_depth}
        results = await self.execute_query(query, params)
        return self._process_subgraph_results(results)

    async def find_shortest_paths(self, start_node_name: str, end_node_name: str, max_hops: int = 3) -> Subgraph:
        query = (
            f"MATCH (start {{canonical_name: $start_name}}), (end {{canonical_name: $end_name}}) "
            f"CALL apoc.algo.allShortestPaths(start, end, null, $max_hops) YIELD path "
            f"UNWIND nodes(path) AS n UNWIND relationships(path) AS r "
            f"RETURN DISTINCT n, r, startNode(r).canonical_name AS s_canonical_name, endNode(r).canonical_name AS t_canonical_name"
        )
        params = {"start_name": start_node_name, "end_name": end_node_name, "max_hops": max_hops}
        results = await self.execute_query(query, params)
        return self._process_subgraph_results(results)

    def _process_subgraph_results(self, results: List[Dict[str, Any]]) -> Subgraph:
        pydantic_nodes_map: Dict[str, PydanticNode] = {}
        temp_edges: List[PydanticEdge] = []
        processed_rel_ids = set()

        for record in results:
            node_proxy = record.get('n')
            if node_proxy:
                node = self._convert_neo4j_node_to_pydantic({"n": node_proxy}, 'n')
                if node and node.id not in pydantic_nodes_map:
                    pydantic_nodes_map[node.id] = node

            rel_proxy = record.get('r')
            if rel_proxy:
                # Neo4j relationship objects from unpivoted paths can be unique by their element_id
                rel_element_id = rel_proxy.element_id
                if rel_element_id not in processed_rel_ids:
                    # Pass the full record for s_canonical_name and t_canonical_name context
                    edge = self._convert_neo4j_relationship_to_pydantic(record, 'r')
                    if edge:
                        temp_edges.append(edge)
                        processed_rel_ids.add(rel_element_id)
        return Subgraph(nodes=list(pydantic_nodes_map.values()), edges=temp_edges)

neo4j_connector_instance: Optional[Neo4jConnector] = None

async def get_neo4j_connector() -> Neo4jConnector:
    global neo4j_connector_instance
    if neo4j_connector_instance is None:
        neo4j_connector_instance = Neo4jConnector()
    return neo4j_connector_instance

async def init_neo4j_driver():
    try:
        connector = await get_neo4j_connector()
        await connector.initialize_driver()
        if connector._driver is not None:
            await connector._ensure_constraints()
        else:
            logger.error("Neo4j driver could not be initialized during app startup. Constraints not ensured.")
    except Exception as e:
        logger.critical(f"Critical failure initializing Neo4j driver or constraints: {e}", exc_info=True)
        # Consider exiting the application if Neo4j is essential and fails to connect at startup
        # For now, it allows the app to start but Neo4j operations will fail.

async def close_neo4j_driver():
    if neo4j_connector_instance is not None:
        await neo4j_connector_instance.close_driver()

if __name__ == "__main__":
    import asyncio

    async def test_neo4j_connection():
        print("--- Testing Neo4j Connector (Corrected) ---")
        connector = await get_neo4j_connector()
        await connector.initialize_driver() # Explicit init for test

        if connector._driver is None:
            print("Failed to initialize Neo4j driver. Check connection settings and Neo4j server status.")
            return

        await connector._ensure_constraints()

        print("\n1. Testing basic Cypher query (Neo4j version)...")
        version_result = await connector.execute_query("CALL dbms.components() YIELD versions RETURN versions[0] AS version", {})
        if version_result: print(f"   Neo4j Version Info: {version_result[0]}")
        else: print("   Could not retrieve version info.")

        entity_props = {"description": "Test entity", "aliases": ["T1"], "source_document_filename": "test.py"}
        await connector.merge_entity("TEST_ENTITY", "TestNode1", entity_props)
        await connector.merge_entity("TEST_ENTITY", "TestNode2", entity_props)
        await connector.merge_relationship("TEST_ENTITY", "TestNode1", "TEST_ENTITY", "TestNode2", "RELATED_TO_TEST", {"context": "test"})

        print("\n2. Testing Get Subgraph for 'TestNode1' (APOC based)...")
        subgraph = await connector.get_subgraph_for_entities(["TestNode1"], hop_depth=1)
        if subgraph and not subgraph.is_empty():
            print(f"   Subgraph Nodes: {[node.id for node in subgraph.nodes]}")
            print(f"   Subgraph Edges: {len(subgraph.edges)}")
        else:
            print("   Subgraph retrieval failed or returned empty (check APOC install and node existence).")

        # Clean up
        # await connector.execute_query("MATCH (n:TEST_ENTITY) DETACH DELETE n")
        # print("Cleaned up TEST_ENTITY nodes.")

        await connector.close_driver()
        print("\n--- Neo4j Connector Test Finished ---")

    if settings.NEO4J_URI:
        asyncio.run(test_neo4j_connection())
    else:
        print("Skipping Neo4j connector tests as NEO4J_URI is not set.")