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
        if self._driver is None:
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
            return []

    async def merge_entity(self, entity_type: str, canonical_name: str, properties: Dict[str, Any]) -> bool:
        """Merges a node into the graph, creating or updating it."""
        props = {"canonical_name": canonical_name, **properties}
        query = f"MERGE (n:{entity_type} {{canonical_name: $canonical_name}}) ON CREATE SET n = $props ON MATCH SET n += $props"
        try:
            await self.execute_query(query, {"canonical_name": canonical_name, "props": props})
            logger.info(f"Merged entity: {entity_type} - {canonical_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to merge entity {entity_type} - {canonical_name}: {e}")
            return False

    async def merge_relationship(self, s_type: str, s_name: str, t_type: str, t_name: str, r_type: str, props: Dict[str, Any]) -> bool:
        """Merges a relationship between two nodes."""
        query = f"MATCH (s:{s_type} {{canonical_name: $s_name}}), (t:{t_type} {{canonical_name: $t_name}}) MERGE (s)-[r:{r_type}]->(t) ON CREATE SET r = $props ON MATCH SET r += $props"
        params = {"s_name": s_name, "t_name": t_name, "props": props}
        try:
            await self.execute_query(query, params)
            logger.info(f"Merged relationship: ({s_name})-[{r_type}]->({t_name})")
            return True
        except Exception as e:
            logger.error(f"Failed to merge relationship ({s_name})-[{r_type}]->({t_name}): {e}")
            return False

    def _convert_neo4j_node_to_pydantic(self, props: dict) -> PydanticNode:
        """Converts a Neo4j Node object to a Pydantic Node model."""
        primary_type = "Unknown"
        node_labels = list(props['original_mentions'])
        for label in node_labels:
            if label in settings.SCHEMA.ENTITY_TYPES:
                primary_type = label
                break
        if primary_type == "Unknown" and node_labels:
            primary_type = node_labels[0]

        node_id = props.get("canonical_name", "No Id Found")
        return PydanticNode(id=node_id, label=props.get("canonical_name", node_id), type=primary_type, properties=props)

    def _convert_neo4j_relationship_to_pydantic(self, rel: tuple, nodes_map: Dict[str, PydanticNode]) -> Optional[PydanticEdge]:
        """Converts a Neo4j Relationship object to a Pydantic Edge model."""
        source_id_by_element = str(rel[0]['canonical_name'])
        target_id_by_element = str(rel[2]['canonical_name'])

        source_node_model = nodes_map.get(source_id_by_element)
        target_node_model = nodes_map.get(target_id_by_element)

        if not source_node_model or not target_node_model:
            logger.warning(f"Could not find start/end node in map for relationship {rel[1]}. Skipping edge.")
            return None

        return PydanticEdge(source=source_node_model.id, target=target_node_model.id, label=rel[1], properties=dict())

    def _process_subgraph_results(self, records: List[Dict[str, Any]]) -> Subgraph:
        """
        Processes raw query results into a Subgraph object.
        This is now robust enough to handle records containing lists of graph objects.
        """
        pydantic_nodes_map: Dict[str, PydanticNode] = {}  # Key: element_id
        pydantic_edges: List[PydanticEdge] = []

        all_items_to_process = []
        for record in records:
            for value in record.values():
                if isinstance(value, list):
                    all_items_to_process.extend(value)
                else:
                    all_items_to_process.append(value)

        # First pass: collect all unique nodes
        for item in all_items_to_process:
            if isinstance(item, dict):
                node_element_id = str(item['canonical_name'])
                if node_element_id not in pydantic_nodes_map:
                    pydantic_nodes_map[node_element_id] = self._convert_neo4j_node_to_pydantic(item)

        # Second pass: collect all relationships
        for item in all_items_to_process:
            if isinstance(item, tuple):
                edge = self._convert_neo4j_relationship_to_pydantic(item, pydantic_nodes_map)
                if edge:
                    pydantic_edges.append(edge)

        unique_edges = list({(e.source, e.target, e.label): e for e in pydantic_edges}.values())
        return Subgraph(nodes=list(pydantic_nodes_map.values()), edges=unique_edges)

    def _process_subgraph_results_full_graph_sample(self, records: List[Dict[str, Any]]) -> Subgraph:
        """
        Processes raw query results into a Subgraph object.
        This is now robust enough to handle records containing lists of graph objects.
        """
        pydantic_nodes_map: Dict[str, PydanticNode] = {}  # Key: element_id
        pydantic_edges: List[PydanticEdge] = []

        # First pass: collect all unique nodes
        for item in records:
            if isinstance(item, dict):
                node_element_id = str(item['canonical_name'])
                if node_element_id not in pydantic_nodes_map:
                    pydantic_nodes_map[node_element_id] = self._convert_neo4j_node_to_pydantic(item)

        # Second pass: collect all relationships
        for item in records:
            if isinstance(item, tuple):
                edge = self._convert_neo4j_relationship_to_pydantic(item, pydantic_nodes_map)
                if edge:
                    pydantic_edges.append(edge)

        unique_edges = list({(e.source, e.target, e.label): e for e in pydantic_edges}.values())
        return Subgraph(nodes=list(pydantic_nodes_map.values()), edges=unique_edges)


    async def get_subgraph_for_entities(self, canonical_names: List[str], hop_depth: int = 1) -> Subgraph:
        """Retrieves an N-hop subgraph for a list of seed entities."""
        if not canonical_names:
            return Subgraph()

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

        if not results:
            return Subgraph()

        # The results list contains records like [{'nodes': [...], 'rels': [...]}]
        # The corrected _process_subgraph_results can now handle this directly.
        return self._process_subgraph_results(results)

# --- Singleton Management ---
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
        await connector._ensure_constraints()
    except Exception as e:
        logger.critical(f"Critical failure during Neo4j initialization: {e}", exc_info=True)

async def close_neo4j_driver():
    connector = await get_neo4j_connector()
    if connector:
        await connector.close_driver()