import logging
from typing import List, Dict, Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.graph import Node as Neo4jNode, Relationship as Neo4jRelationship
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from app.core.config import settings
from app.models.common_models import Node as PydanticNode, Edge as PydanticEdge, Subgraph

logger = logging.getLogger(__name__)

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
        if self._driver:
            logger.info("Closing Neo4j driver.")
            await self._driver.close()
            self._driver = None

    async def _ensure_constraints(self):
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

    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Executes a Cypher query and returns the raw result records."""
        driver = await self._get_driver()
        try:
            async with driver.session() as session:
                logger.debug(f"Executing Cypher: {query} with params: {parameters}")
                result = await session.run(query, parameters)
                records_list = [record async for record in result]
                await result.consume()
                return records_list
        except Exception as e:
            logger.error(f"Error during Cypher query execution: {e}\nQuery: {query}\nParams: {parameters}", exc_info=True)
            return []

    async def merge_entity(self, entity_type: str, canonical_name: str, properties: Dict[str, Any]) -> bool:
        query = f"""
        MERGE (n:{entity_type} {{canonical_name: $canonical_name}})
        ON CREATE SET n = $props, n.source_document_filename = [$source_file]
        ON MATCH SET 
            n.contexts = [ctx IN coalesce(n.contexts, []) + $props.contexts WHERE ctx IS NOT NULL],
            n.original_mentions = [mention IN coalesce(n.original_mentions, []) + $props.original_mentions WHERE mention IS NOT NULL],
            n.source_document_filename = [file IN coalesce(n.source_document_filename, []) + [$source_file] WHERE file IS NOT NULL]
        """
        props_for_create = {"canonical_name": canonical_name, **properties}
        params = {
            "canonical_name": canonical_name,
            "props": props_for_create,
            "source_file": properties.get("source_document_filename")
        }
        try:
            await self.execute_query(query, params)
            logger.info(f"Merged entity: {entity_type} - {canonical_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to merge entity {entity_type} - {canonical_name}: {e}")
            return False

    async def merge_relationship(self, s_type: str, s_name: str, t_type: str, t_name: str, r_type: str, props: Dict[str, Any]) -> bool:
        query = f"""
        MATCH (s:{s_type} {{canonical_name: $s_name}}), (t:{t_type} {{canonical_name: $t_name}})
        MERGE (s)-[r:{r_type}]->(t)
        ON CREATE SET r = $props, r.source_document_filename = [$source_file]
        ON MATCH SET 
            r.contexts = [ctx IN coalesce(r.contexts, []) + $props.contexts WHERE ctx IS NOT NULL],
            r.source_document_filename = [file IN coalesce(r.source_document_filename, []) + [$source_file] WHERE file IS NOT NULL]
        """
        params = {"s_name": s_name, "t_name": t_name, "props": props, "source_file": props.get("source_document_filename")}
        try:
            await self.execute_query(query, params)
            logger.info(f"Merged relationship: ({s_name})-[{r_type}]->({t_name})")
            return True
        except Exception as e:
            logger.error(f"Failed to merge relationship ({s_name})-[{r_type}]->({t_name}): {e}")
            return False

    def _convert_node_to_pydantic(self, node: Neo4jNode) -> PydanticNode:
        """Converts a neo4j.graph.Node object to a Pydantic Node model."""
        props = dict(node)
        primary_type = "Unknown"
        if node.labels:
            primary_type = list(node.labels)[0]

        node_id = props.get("canonical_name", "Unknown ID")
        return PydanticNode(id=node_id, label=props.get("canonical_name", node_id), type=primary_type, properties=props)

    def _process_records_to_subgraph(self, records: List[Any]) -> Subgraph:
        """
        Processes raw Neo4j driver records into a Pydantic Subgraph object.
        This is the single, robust method for converting graph query results.
        """
        pydantic_nodes_map: Dict[str, PydanticNode] = {}
        pydantic_edges: List[PydanticEdge] = []

        for record in records:
            for value in record.values():
                items_to_process = value if isinstance(value, list) else [value]

                for item in items_to_process:
                    if isinstance(item, Neo4jNode):
                        if item.element_id not in pydantic_nodes_map:
                            pydantic_nodes_map[item.element_id] = self._convert_node_to_pydantic(item)

                    elif isinstance(item, Neo4jRelationship):
                        start_node_id = item.start_node.element_id
                        end_node_id = item.end_node.element_id

                        if start_node_id not in pydantic_nodes_map:
                            pydantic_nodes_map[start_node_id] = self._convert_node_to_pydantic(item.start_node)
                        if end_node_id not in pydantic_nodes_map:
                            pydantic_nodes_map[end_node_id] = self._convert_node_to_pydantic(item.end_node)

                        source_pydantic_node = pydantic_nodes_map[start_node_id]
                        target_pydantic_node = pydantic_nodes_map[end_node_id]

                        edge = PydanticEdge(
                            source=source_pydantic_node.id,
                            target=target_pydantic_node.id,
                            label=item.type,
                            properties=dict(item)
                        )
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
            RETURN nodes(path) AS nodes_in_path, relationships(path) as rels_in_path
        }}
        UNWIND nodes_in_path as n
        UNWIND rels_in_path as r
        RETURN collect(DISTINCT n) AS nodes, collect(DISTINCT r) AS rels
        """
        params = {"names": canonical_names}
        records = await self.execute_query(query, params)

        return self._process_records_to_subgraph(records)

    async def get_full_graph_sample(self, node_limit: int, edge_limit: int, filenames: Optional[List[str]] = None) -> Subgraph:
        """
        Retrieves a sample of the full graph using a robust query.
        All Cypher logic is now contained within the connector.
        """
        match_clause = "MATCH (n)"
        if filenames:
            match_clause = "MATCH (n) WHERE any(file IN n.source_document_filename WHERE file IN $filenames)"

        # This query first finds a set of edges, then returns those edges
        # along with their start and end nodes. This is a common sampling strategy.
        query = f"""
        {match_clause}
        WITH n LIMIT $node_limit
        MATCH (n)-[r]-(m)
        RETURN n, m, r
        LIMIT $edge_limit
        """
        params = {"node_limit": node_limit, "edge_limit": edge_limit, "filenames": filenames}
        records = await self.execute_query(query, params)

        # The connector is already responsible for converting records to a subgraph
        return self._process_records_to_subgraph(records)

    async def get_top_n_busiest_nodes_subgraph(self, top_n: int, hop_depth: int, filenames: Optional[List[str]] = None) -> Subgraph:
        """
        Finds the busiest nodes and returns their N-hop subgraph.
        """
        match_clause = "MATCH (n)"
        if filenames:
            match_clause = "MATCH (n) WHERE any(file IN n.source_document_filename WHERE file IN $filenames)"

        query_busiest_nodes = f"""
        {match_clause}
        WITH n
        WHERE n.canonical_name IS NOT NULL
        RETURN n.canonical_name AS canonical_name, COUNT{{(n)--()}} AS degree
        ORDER BY degree DESC
        LIMIT toInteger($top_n)
        """
        params = {"top_n": top_n, "filenames": filenames}
        busiest_nodes_results = await self.execute_query(query_busiest_nodes, params)

        if not busiest_nodes_results:
            return Subgraph()

        busiest_canonical_names = [record["canonical_name"] for record in busiest_nodes_results]

        # Reuse the existing subgraph fetching logic
        return await self.get_subgraph_for_entities(
            canonical_names=busiest_canonical_names,
            hop_depth=hop_depth
        )

    async def get_graph_schema(self) -> Dict[str, List[str]]:
        """
        Dynamically discovers the node labels and relationship types in the database.
        """
        # Query for all labels in use
        labels_query = "CALL db.labels() YIELD label RETURN collect(label) as labels"
        # Query for all relationship types in use
        rel_types_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as rel_types"

        labels_records = await self.execute_query(labels_query)
        rel_types_records = await self.execute_query(rel_types_query)

        labels = labels_records[0]['labels'] if labels_records and labels_records[0]['labels'] else []
        rel_types = rel_types_records[0]['rel_types'] if rel_types_records and rel_types_records[0]['rel_types'] else []

        return {"node_labels": labels, "relationship_types": rel_types}

    async def safely_remove_file_references(self, filename: str):
        """
        Finds all nodes and relationships referencing a specific file and safely
        removes them. If an element is only sourced from this file, it is deleted.
        Otherwise, only the filename is removed from its source list.
        This entire operation is contained within the connector.
        """
        logger.info(f"Connector: Initiating safe removal of references for file: '{filename}'.")

        # This is the raw Cypher query, now properly encapsulated in the data access layer.
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
        await self.execute_query(query, params)
        logger.info(f"Connector: Neo4j safe removal process completed for '{filename}'.")

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