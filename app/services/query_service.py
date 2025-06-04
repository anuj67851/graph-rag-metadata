import logging
from typing import List, Dict, Any, Optional, Tuple

from app.core.config import settings
from app.llm_integration.openai_connector import (
    extract_entities_from_query,
    classify_query_intent,
    generate_response_from_subgraph
)
from app.vector_store.faiss_connector import get_faiss_connector, FaissConnector
from app.graph_db.neo4j_connector import get_neo4j_connector, Neo4jConnector
from app.models.query_models import QueryRequest, QueryResponse, QueryIntent
from app.models.common_models import Subgraph, Node as PydanticNode, Edge as PydanticEdge

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

async def _link_query_entities_to_graph(
        query_entities_text: List[str],
        faiss_conn: FaissConnector
) -> List[Tuple[str, str, float]]:
    """
    Links extracted query entity strings to canonical entity names in the graph using FAISS.

    Args:
        query_entities_text: List of entity strings extracted from the user's query.
        faiss_conn: Instance of FaissConnector.

    Returns:
        A list of tuples: (original_query_entity_text, matched_canonical_name, match_score).
        Filters matches based on settings.VECTOR_MATCH_THRESHOLD (L2 distance, lower is better).
    """
    linked_graph_entities_info: List[Tuple[str, str, float]] = []
    if not query_entities_text:
        return linked_graph_entities_info

    for query_entity_str in query_entities_text:
        if not query_entity_str.strip():
            continue
        # Search FAISS for top_k (e.g., 1-3) similar entities
        # For simplicity, let's take top 1 if it meets threshold.
        # A more complex disambiguation might be needed if multiple good matches.
        search_results = await faiss_conn.search_similar_entities(query_entity_str, top_k=1)
        if search_results:
            matched_canonical_name, score = search_results[0]
            # Score is L2 distance. Lower is better. Threshold should be set accordingly.
            if score <= settings.VECTOR_MATCH_THRESHOLD: # e.g., threshold might be 0.7 for L2
                logger.info(f"Linked query entity '{query_entity_str}' to graph entity '{matched_canonical_name}' with score {score:.4f}")
                linked_graph_entities_info.append((query_entity_str, matched_canonical_name, score))
            else:
                logger.info(f"Query entity '{query_entity_str}' match '{matched_canonical_name}' (score: {score:.4f}) below threshold {settings.VECTOR_MATCH_THRESHOLD}.")
        else:
            logger.info(f"No graph entity found in FAISS for query entity: '{query_entity_str}'")

    return linked_graph_entities_info


def _format_subgraph_for_llm_context(subgraph: Subgraph) -> str:
    """
    Formats the retrieved subgraph (nodes and edges) into a textual representation
    suitable for providing as context to an LLM.
    """
    if subgraph.is_empty():
        return "No relevant information found in the knowledge graph for this query."

    context_parts = []
    context_parts.append("Knowledge Graph Context:")

    if subgraph.nodes:
        context_parts.append("\nEntities Found:")
        for node in subgraph.nodes:
            props_str_parts = [f"Name: {node.label}", f"Type: {node.type}"]
            if node.properties:
                aliases = node.properties.get('aliases') or node.properties.get('original_mentions') # from ingestion
                if aliases:
                    props_str_parts.append(f"Aliases: {', '.join(aliases[:3])}") # Show a few

                contexts = node.properties.get('contexts')
                if contexts:
                    # Show first context or a summary
                    props_str_parts.append(f"Context: {contexts[0][:150] + '...' if len(contexts[0]) > 150 else contexts[0]}")

                source_doc = node.properties.get('source_document_filename')
                if source_doc:
                    props_str_parts.append(f"Source Document: {source_doc}")
            context_parts.append(f"- Node: [{'; '.join(props_str_parts)}]")

    if subgraph.edges:
        context_parts.append("\nRelationships Found:")
        for edge in subgraph.edges:
            props_str_parts = [f"Type: {edge.label}"]
            if edge.properties:
                contexts = edge.properties.get('contexts')
                if contexts:
                    props_str_parts.append(f"Context: {contexts[0][:150] + '...' if len(contexts[0]) > 150 else contexts[0]}")
                source_doc = edge.properties.get('source_document_filename')
                if source_doc:
                    props_str_parts.append(f"Source Document: {source_doc}")
            context_parts.append(f"- Relationship: ({edge.source}) -[{', '.join(props_str_parts)}]-> ({edge.target})")

    return "\n".join(context_parts)


async def process_user_query(query_request: QueryRequest) -> QueryResponse:
    """
    Main service function to process a user's query using the RAG pipeline.
    """
    user_query = query_request.query
    logger.info(f"Processing user query: '{user_query}'")

    faiss_conn: FaissConnector = await get_faiss_connector()
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()

    # 1. Extract key entities from the query
    extracted_query_entities_text: Optional[List[str]] = await extract_entities_from_query(user_query)
    if extracted_query_entities_text is None: # Indicates an error in LLM call or parsing
        logger.error("Failed to extract entities from query due to LLM/parsing error.")
        return QueryResponse(
            llm_answer="I encountered an issue trying to understand the entities in your query. Please try rephrasing.",
            subgraph_context=Subgraph(),
            llm_context_input_text="Error: Could not extract entities from query."
        )
    logger.info(f"Extracted query entities: {extracted_query_entities_text}")

    # 2. Link query entities to graph entities via FAISS
    # Returns list of (original_query_entity, matched_canonical_name, score)
    linked_entities_info = await _link_query_entities_to_graph(extracted_query_entities_text, faiss_conn)

    # Get just the canonical names of successfully linked entities
    linked_canonical_names = [info[1] for info in linked_entities_info]
    if not linked_canonical_names:
        logger.info(f"No entities from query '{user_query}' could be confidently linked to the knowledge graph.")
        # Fallback: try to answer with a general LLM, or state no info. For RAG, we need linked entities.
        # For now, if no entities linked, we might not be able to proceed with graph retrieval effectively.
        # Let's try to classify intent even without linked entities, LLM might still give some direction.
        # Or, we could try a broader keyword search on graph if FAISS fails.
        pass # Continue to intent classification, it might handle queries without specific graph entities.


    # 3. Classify query intent
    intent_classification_result: Optional[Dict[str, Any]] = await classify_query_intent(user_query, linked_canonical_names)

    parsed_intent: Optional[QueryIntent] = None
    if intent_classification_result:
        try:
            parsed_intent = QueryIntent(**intent_classification_result)
            logger.info(f"Classified query intent: {parsed_intent.query_intent}, Target Entities: {parsed_intent.target_entities}")
        except Exception as e: # Pydantic ValidationError
            logger.error(f"Error parsing intent classification LLM output: {e}. Output: {intent_classification_result}")
            # Fallback intent or error

    if not parsed_intent:
        logger.warning("Could not classify query intent. Proceeding with default graph retrieval if entities are linked.")
        # Default behavior: if entities linked, get their N-hop. If not, result in empty subgraph.
        if linked_canonical_names:
            parsed_intent = QueryIntent(query_intent="entity_information", target_entities=linked_canonical_names, intent_description="Default fallback due to unclear intent.")
        else:
            return QueryResponse(
                llm_answer="I'm having trouble understanding the intent of your query or linking it to known information. Could you please rephrase or be more specific?",
                subgraph_context=Subgraph(),
                llm_context_input_text="Error: Could not determine query intent and no entities linked."
            )

    # 4. Retrieve subgraph from Neo4j based on intent and linked entities
    retrieved_subgraph = Subgraph() # Default to empty subgraph

    # Ensure target entities for retrieval are based on intent classification, fallback to linked_canonical_names if needed.
    entities_for_retrieval = parsed_intent.target_entities
    if not entities_for_retrieval and linked_canonical_names: # If intent didn't specify targets but we have links
        entities_for_retrieval = linked_canonical_names
        logger.info(f"Intent target entities empty, using linked entities for retrieval: {entities_for_retrieval}")


    if entities_for_retrieval: # Only attempt graph retrieval if we have some target entities
        intent_type = parsed_intent.query_intent.lower()
        if intent_type == "entity_information" or intent_type == "summarization_of_entity":
            logger.info(f"Retrieving {settings.ENTITY_INFO_HOP_DEPTH}-hop subgraph for entities: {entities_for_retrieval}")
            retrieved_subgraph = await neo4j_conn.get_subgraph_for_entities(
                canonical_names=entities_for_retrieval,
                hop_depth=settings.ENTITY_INFO_HOP_DEPTH
            )
        elif intent_type == "relationship_discovery" and len(entities_for_retrieval) >= 2:
            # Assuming relationship discovery is primarily between the first two target entities
            start_node = entities_for_retrieval[0]
            end_node = entities_for_retrieval[1]
            logger.info(f"Finding shortest paths between '{start_node}' and '{end_node}' (max hops: {settings.RELATIONSHIP_DISCOVERY_MAX_PATH_LENGTH})")
            retrieved_subgraph = await neo4j_conn.find_shortest_paths(
                start_node_name=start_node,
                end_node_name=end_node,
                max_hops=settings.RELATIONSHIP_DISCOVERY_MAX_PATH_LENGTH
            )
        elif intent_type == "complex_reasoning": # Or other types requiring broader context
            logger.info(f"Complex reasoning intent: Retrieving {settings.COMPLEX_QUERY_HOP_DEPTH}-hop subgraph for entities: {entities_for_retrieval}")
            retrieved_subgraph = await neo4j_conn.get_subgraph_for_entities(
                canonical_names=entities_for_retrieval,
                hop_depth=settings.COMPLEX_QUERY_HOP_DEPTH
            )
        # Add more intent handling as needed
        else:
            if entities_for_retrieval: # If there are entities but intent type is not specifically handled for graph traversal
                logger.info(f"Unhandled intent type '{intent_type}' with entities, defaulting to {settings.ENTITY_INFO_HOP_DEPTH}-hop subgraph for: {entities_for_retrieval}")
                retrieved_subgraph = await neo4j_conn.get_subgraph_for_entities(
                    canonical_names=entities_for_retrieval,
                    hop_depth=settings.ENTITY_INFO_HOP_DEPTH
                )
            else:
                logger.info(f"Intent type '{intent_type}' but no specific entities for graph retrieval.")

    if retrieved_subgraph.is_empty():
        logger.info(f"No subgraph data retrieved from Neo4j for query: '{user_query}' and linked entities: {entities_for_retrieval}")


    # 5. Format subgraph as LLM context
    llm_context_text = _format_subgraph_for_llm_context(retrieved_subgraph)
    logger.debug(f"Formatted LLM context:\n{llm_context_text}")

    # 6. Generate final answer using LLM
    final_answer_text: Optional[str] = await generate_response_from_subgraph(user_query, llm_context_text)

    if final_answer_text is None:
        logger.error("LLM failed to generate a final answer.")
        final_answer_text = "I encountered an issue while trying to formulate a response based on the available information."
        # If LLM fails, we still send the context we tried to use.

    return QueryResponse(
        llm_answer=final_answer_text,
        subgraph_context=retrieved_subgraph,
        llm_context_input_text=llm_context_text
    )


if __name__ == "__main__":
    import asyncio

    async def test_query_service():
        print("--- Testing Query Service ---")
        # This test requires:
        # 1. OpenAI API Key (for NER, intent, generation)
        # 2. Running Neo4j with some data (e.g., from ingesting sample_document.md)
        # 3. FAISS index populated with embeddings for that data
        # 4. All connectors (OpenAI, Neo4j, FAISS) to be functional.

        if not all([settings.OPENAI_API_KEY, settings.NEO4J_URI]):
            print("OPENAI_API_KEY or NEO4J_URI not set. Skipping query service test.")
            return

        # Ensure FAISS index and Neo4j have data from 'sample_document.md'
        # You might need to run ingestion_service test first, or have a pre-populated DB/Index.
        # For this test, we'll assume some data exists.

        # Query 1: Information about an entity
        # query1 = "Tell me about Alpha Corp and their projects."
        # Query 2: Relationship between entities
        query2 = "What is the role of Dr. Aris Thorne at Alpha Corp?"
        # Query 3: Technology used
        # query3 = "What AI technology does Project Nova use?"
        # Query 4: Query with an entity not in the sample doc (to test linking failure)
        # query4 = "What about Microsoft's latest AI?"


        test_queries = [
            "Tell me about Alpha Corp and its projects like Project Nova.",
            "What is the role of Dr. Aris Thorne at Alpha Corp?",
            "What AI technology does Project Nova use, specifically Helios Optimizer?",
            "Who is the CEO of Alpha Corp?",
            "Which companies did Alpha Corp partner with?" # Test multiple linked entities from question
        ]

        for i, test_query_str in enumerate(test_queries):
            print(f"\n--- Test Query {i+1}: '{test_query_str}' ---")
            req = QueryRequest(query=test_query_str)
            try:
                response = await process_user_query(req)
                print("\nLLM Answer:")
                print(response.llm_answer)
                print("\nSubgraph Context (Nodes):")
                for node in response.subgraph_context.nodes:
                    print(f"  - {node.id} ({node.type})")
                print("Subgraph Context (Edges):")
                for edge in response.subgraph_context.edges:
                    print(f"  - ({edge.source}) -[{edge.label}]-> ({edge.target})")

                # print("\nFull LLM Context Input Text (first 300 chars):")
                # print(response.llm_context_input_text[:300] + "...")
                print("-" * 30)

            except Exception as e:
                print(f"Error during query service test for query '{test_query_str}': {e}", exc_info=True)

        # Cleanup (close Neo4j driver, save FAISS - handled by main app lifecycle)
        # For standalone test, explicit close/save might be needed if not done in get_...
        try:
            neo4j_conn_main = await get_neo4j_connector()
            await neo4j_conn_main.close_driver()
            faiss_conn_main = await get_faiss_connector()
            faiss_conn_main.save_index()
        except Exception as e:
            print(f"Error during test cleanup: {e}")


    asyncio.run(test_query_service())