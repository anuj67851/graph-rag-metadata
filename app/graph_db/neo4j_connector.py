import json
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
            self._driver = None; raise
        except Exception as e:
            logger.error(f"Unexpected error during Neo4j driver initialization: {e}")
            self._driver = None; raise

    async def _get_driver(self) -> AsyncDriver:
        if self._driver is None:
            logger.warning("Neo4j driver is None. Attempting to initialize now.")
            await self.initialize_driver()
            if self._driver is None:
                raise ServiceUnavailable("Neo4j driver is not available. Initialization failed.")
        return self._driver

    async def close_driver(self):
        if self._driver is not None:
            logger.info("Closing Neo4j driver."); await self._driver.close(); self._driver = None

    async def _ensure_constraints(self):
        driver = await self._get_driver()
        entity_types = settings.SCHEMA.ENTITY_TYPES
        if not entity_types: logger.warning("No entity types in schema. Skipping constraints."); return
        async with driver.session() as session:
            for entity_type in entity_types:
                name = f"constraint_unique_{entity_type.lower()}_canonical_name"
                q = f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{entity_type}) REQUIRE n.canonical_name IS UNIQUE"
                try: logger.debug(f"Ensuring constraint for {entity_type}: {q}"); await session.run(q)
                except Exception as e: logger.error(f"Error ensuring constraint for {entity_type} ({name}): {e}")
            logger.info("Finished ensuring constraints.")

    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        driver = await self._get_driver()
        records_list = []
        try:
            async with driver.session() as session:
                logger.debug(f"Executing Cypher: {query} with params: {parameters}")
                result = await session.run(query, parameters)
                async for record in result: records_list.append(record.data())
                await result.consume()
        except Exception as e: logger.error(f"Error during query: {e}\nQuery: {query}\nParams: {parameters}")
        return records_list

    async def merge_entity(self, entity_type: str, canonical_name: str, properties: Dict[str, Any]) -> bool:
        props = {"canonical_name": canonical_name, **properties}
        q = f"MERGE (n:{entity_type} {{canonical_name: $canonical_name}}) ON CREATE SET n = $props ON MATCH SET n += $props"
        try: await self.execute_query(q, {"canonical_name": canonical_name, "props": props}); logger.info(f"Merged entity: {entity_type} - {canonical_name}"); return True
        except Exception as e: logger.error(f"Failed to merge entity {entity_type} - {canonical_name}: {e}"); return False

    async def merge_relationship(self,s_type: str,s_name: str,t_type: str,t_name: str,r_type: str,props: Dict[str, Any]) -> bool:
        q = (f"MATCH (s:{s_type} {{canonical_name: $s_name}}), (t:{t_type} {{canonical_name: $t_name}}) "
             f"MERGE (s)-[r:{r_type}]->(t) ON CREATE SET r = $props ON MATCH SET r += $props")
        params = {"s_name": s_name, "t_name": t_name, "props": props}
        try: await self.execute_query(q, params); logger.info(f"Merged rel: ({s_name})-[{r_type}]->({t_name})"); return True
        except Exception as e: logger.error(f"Failed to merge rel ({s_name})-[{r_type}]->({t_name}): {e}"); return False

    def _convert_neo4j_node_to_pydantic(self, node_input: Any) -> Optional[PydanticNode]:
        try:
            if isinstance(node_input, Neo4jNodeObject): # Direct Neo4j Node object
                node_obj = node_input
                props = dict(node_obj)
                node_labels = list(node_obj.labels)
                node_element_id = node_obj.element_id # Graph DB global ID string
            elif isinstance(node_input, dict): # Dictionary representation
                props = node_input
                node_labels = props.get("_labels", []) # Convention if labels are in dict
                node_element_id = props.get("element_id", props.get("id")) # If element_id or id is in dict
                if not node_labels and props.get("type"): node_labels = [props.get("type")] # Fallback
            else:
                logger.warning(f"Cannot convert to PydanticNode, unexpected input type: {type(node_input)}. Value: {str(node_input)[:200]}")
                return None

            primary_type = "Unknown"
            if node_labels:
                primary_type = node_labels[0]
                for label in node_labels:
                    if label in settings.SCHEMA.ENTITY_TYPES: primary_type = label; break
            elif props.get("type"): primary_type = props["type"]
            elif props.get("entity_type"): primary_type = props["entity_type"]

            node_id_val = props.get("canonical_name", str(node_element_id) if node_element_id else None)
            if node_id_val is None: logger.warning(f"Node missing canonical_name and element_id/id. Props: {props}"); return None

            return PydanticNode(id=str(node_id_val), label=props.get("canonical_name", str(node_id_val)), type=primary_type, properties=props)
        except Exception as e:
            logger.error(f"Error converting Neo4j val to PydanticNode: {e}. Input: {str(node_input)[:200]}", exc_info=True)
            return None

    def _convert_neo4j_relationship_to_pydantic(self, rel_input: Any, record_context: Dict[str, Any]) -> Optional[PydanticEdge]:
        try:
            rel_type_val = ""
            rel_props = {}
            if isinstance(rel_input, Neo4jRelationshipObject): # Direct Neo4j Relationship object
                rel_obj = rel_input
                rel_type_val = rel_obj.type
                rel_props = dict(rel_obj)
            elif isinstance(rel_input, dict): # Dictionary representation
                rel_props = rel_input
                rel_type_val = rel_props.get("type", "RELATED_TO") # Default if type missing in dict
            else:
                logger.warning(f"Cannot convert to PydanticEdge, unexpected input type: {type(rel_input)}. Value: {str(rel_input)[:200]}")
                return None

            source_id = record_context.get("s_canonical_name")
            target_id = record_context.get("t_canonical_name")

            # Fallbacks if s_canonical_name/t_canonical_name not in record_context (should be from query)
            # These fallbacks are less ideal as they rely on start_node/end_node being full node objects
            if source_id is None and isinstance(rel_input, Neo4jRelationshipObject) and rel_input.start_node:
                source_id = rel_input.start_node.get("canonical_name", str(rel_input.start_node.element_id))
            if target_id is None and isinstance(rel_input, Neo4jRelationshipObject) and rel_input.end_node:
                target_id = rel_input.end_node.get("canonical_name", str(rel_input.end_node.element_id))

            if source_id is None or target_id is None:
                logger.warning(f"Could not determine source/target ID for relationship. RecordCtx: {record_context}, RelInput: {str(rel_input)[:200]}")
                return None

            return PydanticEdge(source=str(source_id), target=str(target_id), label=rel_type_val, properties=rel_props)
        except Exception as e:
            logger.error(f"Error converting Neo4j val to PydanticEdge: {e}. Input: {str(rel_input)[:200]}", exc_info=True)
            return None

    async def get_subgraph_for_entities(self, canonical_names: List[str], hop_depth: int = 1) -> Subgraph:
        if not canonical_names: return Subgraph()
        logger.info(f"Getting subgraph for entities {canonical_names} (hop: {hop_depth}) using V7 standard Cypher.")

        # Query V7: Each part of UNION ALL is a direct MATCH, UNWIND, and RETURN DISTINCT.
        # No intermediate WITH clauses that might confuse the UNION structure.
        query = (
            # Part 1: Get all distinct nodes from the N-hop paths
                f"MATCH path = (center)-[*0..{hop_depth}]-(neighbor) "
                f"WHERE center.canonical_name IN $names "
                f"UNWIND nodes(path) AS n_item "
                f"RETURN DISTINCT n_item AS item, 'node' AS item_type, "
                f"       null AS s_canonical_name, null AS t_canonical_name, null AS rel_element_id "

                f"UNION ALL "

                # Part 2: Get all distinct relationships from the N-hop paths
                f"MATCH path = (center)-[*0..{hop_depth}]-(neighbor) "
                f"WHERE center.canonical_name IN $names "
                f"UNWIND relationships(path) AS r_item "
                f"RETURN DISTINCT r_item AS item, 'relationship' AS item_type, "
                f"       startNode(r_item).canonical_name AS s_canonical_name, "
                f"       endNode(r_item).canonical_name AS t_canonical_name, "
                f"       elementId(r_item) AS rel_element_id"
                )

        params = {"names": canonical_names}
        results = await self.execute_query(query, params)
        return self._process_subgraph_results_revised_v3(results)


    async def find_shortest_paths(self, start_node_name: str, end_node_name: str, max_hops: int = 3) -> Subgraph:
        logger.info(f"Finding shortest paths between {start_node_name} and {end_node_name} (max_hops: {max_hops}) using V7 standard Cypher.")

        default_node_label = settings.SCHEMA.ENTITY_TYPES[0] if settings.SCHEMA.ENTITY_TYPES and settings.SCHEMA.ENTITY_TYPES[0] else 'Node'

        query_sp = (
            # Part 1: Nodes from all shortest paths
            f"MATCH path = allShortestPaths((startNode:{default_node_label} {{canonical_name: $start_name}})-[*1..{max_hops}]-(endNode:{default_node_label} {{canonical_name: $end_name}})) "
            f"UNWIND nodes(path) AS node_item "
            f"RETURN DISTINCT node_item AS item, 'node' AS item_type, "
            f"       null AS s_canonical_name, null AS t_canonical_name, null AS rel_element_id "

            f"UNION ALL "

            # Part 2: Relationships from all shortest paths
            f"MATCH path = allShortestPaths((startNode:{default_node_label} {{canonical_name: $start_name}})-[*1..{max_hops}]-(endNode:{default_node_label} {{canonical_name: $end_name}})) "
            f"UNWIND relationships(path) AS rel_item "
            f"RETURN DISTINCT rel_item AS item, 'relationship' AS item_type, "
            f"       startNode(rel_item).canonical_name AS s_canonical_name, "
            f"       endNode(rel_item).canonical_name AS t_canonical_name, "
            f"       elementId(rel_item) AS rel_element_id"
        )

        params = {"start_name": start_node_name, "end_name": end_node_name}
        results = await self.execute_query(query_sp, params)

        if not results:
            logger.warning(f"No path found between {start_node_name} and {end_node_name}")
            return Subgraph()

        return self._process_subgraph_results_revised_v3(results)

    def _process_subgraph_results_revised_v3(self, results: List[Dict[str, Any]]) -> Subgraph:
        pydantic_nodes_map: Dict[str, PydanticNode] = {}
        temp_edges: List[PydanticEdge] = []
        processed_rel_element_ids = set()

        logger.debug(f"Processing {len(results)} records with V3 subgraph processor.")

        for i, record in enumerate(results):
            item = record.get('item')
            item_type = record.get('item_type')

            if item_type == 'node' and item:
                pydantic_node = self._convert_neo4j_node_to_pydantic(item)
                if pydantic_node and pydantic_node.id not in pydantic_nodes_map:
                    pydantic_nodes_map[pydantic_node.id] = pydantic_node
            elif item_type == 'relationship' and item:
                current_rel_unique_id = record.get('rel_element_id') # Get element_id directly from record

                if current_rel_unique_id is None: # Fallback if element_id is not returned (should be)
                    if isinstance(item, Neo4jRelationshipObject):
                        current_rel_unique_id = item.element_id
                    elif isinstance(item, dict) and 'element_id' in item:
                        current_rel_unique_id = item['element_id']
                    else:
                        logger.warning(f"Relationship missing element_id in record, falling back to hash. Item: {str(item)[:100]}")
                        try: current_rel_unique_id = hash(json.dumps(item, sort_keys=True, default=str))
                        except TypeError: logger.warning(f"Could not hash rel item: {str(item)[:100]}"); continue

                if current_rel_unique_id in processed_rel_element_ids:
                    continue

                pydantic_edge = self._convert_neo4j_relationship_to_pydantic(item, record) # Pass full record for s/t names
                if pydantic_edge:
                    temp_edges.append(pydantic_edge)
                    processed_rel_element_ids.add(current_rel_unique_id)

        # Ensure all nodes from edges are present in the nodes list (robustness check)
        all_node_ids_from_edges = set()
        for edge in temp_edges:
            all_node_ids_from_edges.add(edge.source)
            all_node_ids_from_edges.add(edge.target)

        for node_id in all_node_ids_from_edges:
            if node_id not in pydantic_nodes_map:
                # This case implies a relationship's start/end node was not captured by the node part of the query.
                # This *shouldn't* happen with the current query logic but if it does, we'd need to fetch the node.
                # For now, log a warning. In a production system, you might fetch missing nodes.
                logger.warning(f"Node '{node_id}' from an edge was not found in the initial node collection. Subgraph might be incomplete.")
                # To fix, you could add a step here to query for these missing nodes:
                # missing_nodes_data = await self.execute_query("MATCH (n) WHERE n.canonical_name = $id RETURN n", {"id": node_id})
                # if missing_nodes_data and missing_nodes_data[0].get('n'):
                #     p_node = self._convert_neo4j_node_to_pydantic(missing_nodes_data[0]['n'])
                #     if p_node: pydantic_nodes_map[p_node.id] = p_node


        logger.debug(f"Processed subgraph (V3): {len(pydantic_nodes_map)} nodes, {len(temp_edges)} edges.")
        return Subgraph(nodes=list(pydantic_nodes_map.values()), edges=temp_edges)


# Global instance management (remains the same)
neo4j_connector_instance: Optional[Neo4jConnector] = None
async def get_neo4j_connector() -> Neo4jConnector: # ... same ...
    global neo4j_connector_instance
    if neo4j_connector_instance is None:
        neo4j_connector_instance = Neo4jConnector()
    return neo4j_connector_instance
async def init_neo4j_driver(): # ... same ...
    try:
        connector = await get_neo4j_connector()
        await connector.initialize_driver()
        if connector._driver is not None: await connector._ensure_constraints()
        else: logger.error("Neo4j driver not initialized. Constraints not ensured.")
    except Exception as e: logger.critical(f"Critical failure initializing Neo4j: {e}", exc_info=True)
async def close_neo4j_driver(): # ... same ...
    if neo4j_connector_instance is not None: await neo4j_connector_instance.close_driver()

if __name__ == "__main__":
    # ... (Test block remains largely the same, ensure it uses non-APOC or that APOC is installed if testing APOC versions)
    # ... For non-APOC tests, the queries in get_subgraph_for_entities/find_shortest_paths are now standard Cypher.
    import asyncio
    async def test_neo4j_connection():
        print("--- Testing Neo4j Connector (Further Corrected Converters) ---")
        connector = await get_neo4j_connector()
        await connector.initialize_driver()

        if connector._driver is None:
            print("Failed to initialize Neo4j driver."); return

        await connector._ensure_constraints() # Assuming TEST_ENTITY is in schema or handled

        print("\n1. Basic query...")
        version_result = await connector.execute_query("CALL dbms.components() YIELD versions RETURN versions[0] AS version", {})
        if version_result: print(f"   Neo4j Version Info: {version_result[0]}")
        else: print("   Could not retrieve version info.")

        # Create some test data using a temporary label if TEST_ENTITY not in schema
        temp_label = "TEMP_TEST_NODE"
        entity_props1 = {"description": "Test entity 1", "aliases": ["T1"], "source_document_filename": "test.py"}
        entity_props2 = {"description": "Test entity 2", "aliases": ["T2"], "source_document_filename": "test.py"}
        await connector.merge_entity(temp_label, "TempNode1", entity_props1)
        await connector.merge_entity(temp_label, "TempNode2", entity_props2)
        await connector.merge_relationship(temp_label, "TempNode1", temp_label, "TempNode2", "TEMP_RELATED_TO", {"context": "test relation"})

        print("\n2. Testing Get Subgraph for 'TempNode1' (Standard Cypher)...")
        subgraph = await connector.get_subgraph_for_entities(["TempNode1"], hop_depth=1)
        if subgraph and not subgraph.is_empty():
            print(f"   Subgraph Nodes: {[node.id for node in subgraph.nodes]}")
            print(f"   Subgraph Edges Count: {len(subgraph.edges)}")
            for edge in subgraph.edges: print(f"     Edge: ({edge.source})-[:{edge.label}]->({edge.target})")
        else:
            print(f"   Subgraph for 'TempNode1' is empty or retrieval failed. Nodes found: {len(subgraph.nodes) if subgraph else 'None'}")

        print("\n3. Testing Find Shortest Paths between 'TempNode1' and 'TempNode2'...")
        paths_subgraph = await connector.find_shortest_paths("TempNode1", "TempNode2", max_hops=2)
        if paths_subgraph and not paths_subgraph.is_empty():
            print(f"   Paths Subgraph Nodes: {[node.id for node in paths_subgraph.nodes]}")
            print(f"   Paths Subgraph Edges Count: {len(paths_subgraph.edges)}")
            for edge in paths_subgraph.edges: print(f"     Path Edge: ({edge.source})-[:{edge.label}]->({edge.target})")
        else:
            print(f"   Paths subgraph for 'TempNode1' to 'TempNode2' is empty. Nodes found: {len(paths_subgraph.nodes) if paths_subgraph else 'None'}")

        # Clean up
        await connector.execute_query(f"MATCH (n:{temp_label}) DETACH DELETE n")
        print(f"Cleaned up {temp_label} nodes.")

        await connector.close_driver()
        print("\n--- Neo4j Connector Test Finished ---")

    if settings.NEO4J_URI: asyncio.run(test_neo4j_connection())
    else: print("Skipping Neo4j connector tests: NEO4J_URI not set.")