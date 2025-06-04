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
            # Handle tuple input
            if isinstance(rel_input, tuple):
                rel_input = rel_input[0]  # Extract relationship from tuple

            rel_type_val = ""
            rel_props = {}
            if isinstance(rel_input, Neo4jRelationshipObject):
                rel_obj = rel_input
                rel_type_val = rel_obj.type
                rel_props = dict(rel_obj)
            elif isinstance(rel_input, dict):
                rel_props = rel_input
                rel_type_val = rel_props.get("type", "RELATED_TO")
            else:
                logger.warning(f"Cannot convert to PydanticEdge, unexpected input type: {type(rel_input)}. Value: {str(rel_input)[:200]}")
                return None

            # Use element IDs from record context
            source_id = record_context.get("s_element_id")
            target_id = record_context.get("t_element_id")

            if source_id is None or target_id is None:
                logger.warning(f"Could not determine source/target ID for relationship. RecordCtx: {record_context}, RelInput: {str(rel_input)[:200]}")
                return None

            return PydanticEdge(
                source=str(source_id),
                target=str(target_id),
                label=rel_type_val,
                properties=rel_props
            )
        except Exception as e:
            logger.error(f"Error converting Neo4j val to PydanticEdge: {e}. Input: {str(rel_input)[:200]}", exc_info=True)
            return None

    async def get_subgraph_for_entities(self, canonical_names: List[str], hop_depth: int = 1) -> Subgraph:
        if not canonical_names: return Subgraph()
        logger.info(f"Getting subgraph for entities {canonical_names} (hop: {hop_depth}) using V7 standard Cypher.")

        # Revised query to capture all nodes and relationships
        query = (
            # Part 1: Get all distinct nodes from the N-hop paths
            f"MATCH path = (center)-[*0..{hop_depth}]-(neighbor) "
            f"WHERE center.canonical_name IN $names "
            f"WITH COLLECT(DISTINCT nodes(path)) AS allNodes "
            f"UNWIND allNodes AS nodeList "
            f"UNWIND nodeList AS n_item "
            f"RETURN DISTINCT n_item AS item, 'node' AS item_type, "
            f"       null AS s_element_id, null AS t_element_id, null AS rel_element_id "
    
            f"UNION ALL "
    
            # Part 2: Get all distinct relationships from the N-hop paths
            f"MATCH path = (center)-[*0..{hop_depth}]-(neighbor) "
            f"WHERE center.canonical_name IN $names "
            f"WITH COLLECT(DISTINCT relationships(path)) AS allRels "
            f"UNWIND allRels AS relList "
            f"UNWIND relList AS r_item "
            f"RETURN DISTINCT r_item AS item, 'relationship' AS item_type, "
            f"       elementId(startNode(r_item)) AS s_element_id, "
            f"       elementId(endNode(r_item)) AS t_element_id, "
            f"       elementId(r_item) AS rel_element_id"
        )

        params = {"names": canonical_names}
        results = await self.execute_query(query, params)
        return self._process_subgraph_results_revised_v3(results)


    async def find_shortest_paths(self, start_node_name: str, end_node_name: str, max_hops: int = 3) -> Subgraph:
        logger.info(f"Finding shortest paths between {start_node_name} and {end_node_name} (max_hops: {max_hops}) using V7 standard Cypher.")

        default_node_label = settings.SCHEMA.ENTITY_TYPES[0] if settings.SCHEMA.ENTITY_TYPES and settings.SCHEMA.ENTITY_TYPES[0] else 'Node'

        # Revised query to capture all nodes and relationships
        query_sp = (
            # Part 1: Nodes from all shortest paths
            f"MATCH path = allShortestPaths((startNode:{default_node_label} {{canonical_name: $start_name}})-[*1..{max_hops}]-(endNode:{default_node_label} {{canonical_name: $end_name}})) "
            f"WITH COLLECT(DISTINCT nodes(path)) AS allNodes "
            f"UNWIND allNodes AS nodeList "
            f"UNWIND nodeList AS node_item "
            f"RETURN DISTINCT node_item AS item, 'node' AS item_type, "
            f"       null AS s_element_id, null AS t_element_id, null AS rel_element_id "
    
            f"UNION ALL "
    
            # Part 2: Relationships from all shortest paths
            f"MATCH path = allShortestPaths((startNode:{default_node_label} {{canonical_name: $start_name}})-[*1..{max_hops}]-(endNode:{default_node_label} {{canonical_name: $end_name}})) "
            f"WITH COLLECT(DISTINCT relationships(path)) AS allRels "
            f"UNWIND allRels AS relList "
            f"UNWIND relList AS rel_item "
            f"RETURN DISTINCT rel_item AS item, 'relationship' AS item_type, "
            f"       elementId(startNode(rel_item)) AS s_element_id, "
            f"       elementId(endNode(rel_item)) AS t_element_id, "
            f"       elementId(rel_item) AS rel_element_id"
        )

        params = {"start_name": start_node_name, "end_name": end_node_name}
        results = await self.execute_query(query_sp, params)
        return self._process_subgraph_results_revised_v3(results)

    def _process_subgraph_results_revised_v3(self, results: List[Dict[str, Any]]) -> Subgraph:
        pydantic_nodes_map: Dict[str, PydanticNode] = {}
        pydantic_nodes_by_element_id: Dict[str, PydanticNode] = {}  # New: track nodes by element_id
        temp_edges: List[PydanticEdge] = []
        processed_rel_element_ids = set()

        logger.debug(f"Processing {len(results)} records with V3 subgraph processor.")

        # First pass: process all nodes
        for record in results:
            if record.get('item_type') != 'node' or not record.get('item'):
                continue

            node = record['item']
            pydantic_node = self._convert_neo4j_node_to_pydantic(node)
            if not pydantic_node:
                continue

            # Get element_id from Neo4j node object
            element_id = None
            if isinstance(node, Neo4jNodeObject):
                element_id = node.element_id
            elif isinstance(node, dict) and 'element_id' in node:
                element_id = node['element_id']

            if element_id:
                pydantic_nodes_by_element_id[element_id] = pydantic_node

            if pydantic_node.id not in pydantic_nodes_map:
                pydantic_nodes_map[pydantic_node.id] = pydantic_node

        # Second pass: process relationships
        for record in results:
            if record.get('item_type') != 'relationship' or not record.get('item'):
                continue

            item = record['item']
            current_rel_unique_id = record.get('rel_element_id')

            # Handle tuple input
            if isinstance(item, tuple):
                item = item[0]  # Extract relationship from tuple

            if not current_rel_unique_id:
                if isinstance(item, Neo4jRelationshipObject):
                    current_rel_unique_id = item.element_id
                elif isinstance(item, dict) and 'element_id' in item:
                    current_rel_unique_id = item['element_id']
                else:
                    logger.warning(f"Relationship missing element_id, skipping")
                    continue

            if current_rel_unique_id in processed_rel_element_ids:
                continue

            # Get source and target element IDs from record
            source_element_id = record.get('s_element_id')
            target_element_id = record.get('t_element_id')

            # Convert to PydanticEdge
            pydantic_edge = self._convert_neo4j_relationship_to_pydantic(item, {
                "s_element_id": source_element_id,
                "t_element_id": target_element_id
            })

            if not pydantic_edge:
                continue

            # Ensure nodes exist in our map
            if source_element_id and source_element_id in pydantic_nodes_by_element_id:
                source_node = pydantic_nodes_by_element_id[source_element_id]
                if source_node.id not in pydantic_nodes_map:
                    pydantic_nodes_map[source_node.id] = source_node

            if target_element_id and target_element_id in pydantic_nodes_by_element_id:
                target_node = pydantic_nodes_by_element_id[target_element_id]
                if target_node.id not in pydantic_nodes_map:
                    pydantic_nodes_map[target_node.id] = target_node

            temp_edges.append(pydantic_edge)
            processed_rel_element_ids.add(current_rel_unique_id)

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