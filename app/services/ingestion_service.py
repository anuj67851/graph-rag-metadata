import logging
from typing import IO, Union, Dict, List, Any, Tuple

from app.utils.file_parser import extract_text_from_file, FileParsingError
from app.llm_integration.openai_connector import extract_entities_relationships_from_chunk
from app.graph_db.neo4j_connector import get_neo4j_connector, Neo4jConnector
from app.vector_store.faiss_connector import get_faiss_connector, FaissConnector
from app.models.ingestion_models import LLMExtractionOutput, ExtractedEntity, ExtractedRelationship, IngestionStatus

logger = logging.getLogger(__name__)
# Ensure logger is configured
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    if not text: return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len: break
        start += (chunk_size - chunk_overlap)
        if start >= end: start = end
    return chunks

def _create_entity_text_representation_for_embedding(
        canonical_name: str, entity_type: str, aggregated_contexts: List[str]
) -> str:
    contexts_str = " ".join(aggregated_contexts) if aggregated_contexts else "No context provided."
    representation = (
        f"Entity Name: {canonical_name}. "
        f"Type: {entity_type}. "
        f"Description: {contexts_str}"
    )
    return representation


async def process_document_for_ingestion(filename: str, file_content: Union[bytes, IO[bytes]]) -> IngestionStatus:
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()
    faiss_conn: FaissConnector = await get_faiss_connector()

    try:
        raw_text = extract_text_from_file(filename, file_content)
        if not raw_text or not raw_text.strip():
            return IngestionStatus(filename=filename, status="Failed", message="No text content found in file.")
    except (FileParsingError, ValueError) as e:
        return IngestionStatus(filename=filename, status="Failed", message=f"File processing error: {e}")

    text_chunks = chunk_text(raw_text)
    if not text_chunks:
        return IngestionStatus(filename=filename, status="Failed", message="No text chunks to process.")

    logger.info(f"Processing {filename}: {len(text_chunks)} chunks generated.")

    all_extracted_entities_from_llm: List[ExtractedEntity] = []
    all_extracted_relationships_from_llm: List[ExtractedRelationship] = []

    for i, chunk in enumerate(text_chunks):
        llm_raw_output = await extract_entities_relationships_from_chunk(chunk)
        if llm_raw_output:
            try:
                llm_data = LLMExtractionOutput(**llm_raw_output)
                all_extracted_entities_from_llm.extend(llm_data.entities)
                all_extracted_relationships_from_llm.extend(llm_data.relationships)
            except Exception as e:
                logger.error(f"Error parsing LLM output for chunk {i+1} of {filename}: {e}. Output: {str(llm_raw_output)[:500]}")
        else:
            logger.warning(f"LLM returned no output for chunk {i+1} of {filename}.")

    # --- Consolidation Logic Revised ---
    # Use dictionaries to hold consolidated data before creating final Pydantic models or Neo4j properties.
    # Key: canonical_name. Value: dictionary of properties.
    consolidated_entity_data_map: Dict[str, Dict[str, Any]] = {}

    for extr_entity in all_extracted_entities_from_llm:
        canonical_name = extr_entity.canonical_name
        if canonical_name not in consolidated_entity_data_map:
            consolidated_entity_data_map[canonical_name] = {
                "canonical_name": canonical_name,
                "entity_type": extr_entity.entity_type, # Type from first encounter
                "original_mentions": [extr_entity.original_mention],
                "contexts": list(set(extr_entity.contexts)) # Unique contexts
            }
        else:
            # Entity already seen, aggregate mentions and contexts
            data = consolidated_entity_data_map[canonical_name]
            data["original_mentions"].append(extr_entity.original_mention)
            # Make unique after adding all
            data["contexts"].extend(extr_entity.contexts)
            # Type remains from first encounter for simplicity

    # Post-process to ensure unique mentions and contexts
    for data in consolidated_entity_data_map.values():
        data["original_mentions"] = list(set(data["original_mentions"]))
        data["contexts"] = list(set(data["contexts"]))

    logger.info(f"Consolidated to {len(consolidated_entity_data_map)} unique entities for {filename}.")

    # Consolidate relationships (similar logic, using a temporary dictionary)
    # Key: (source_canonical, type, target_canonical). Value: dictionary of properties.
    consolidated_relationship_data_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for extr_rel in all_extracted_relationships_from_llm:
        rel_key = (extr_rel.source_canonical_name, extr_rel.relationship_type, extr_rel.target_canonical_name)
        if rel_key not in consolidated_relationship_data_map:
            consolidated_relationship_data_map[rel_key] = {
                "source_canonical_name": extr_rel.source_canonical_name,
                "relationship_type": extr_rel.relationship_type,
                "target_canonical_name": extr_rel.target_canonical_name,
                "contexts": list(set(extr_rel.contexts))
            }
        else:
            data = consolidated_relationship_data_map[rel_key]
            data["contexts"].extend(extr_rel.contexts)
            # Make unique after adding all

    for data in consolidated_relationship_data_map.values():
        data["contexts"] = list(set(data["contexts"]))

    logger.info(f"Consolidated to {len(consolidated_relationship_data_map)} unique relationships for {filename}.")

    entities_added_neo4j = 0
    rels_added_neo4j = 0

    # Store entities in Neo4j and add to FAISS
    for entity_data_dict in consolidated_entity_data_map.values():
        neo4j_node_properties = {
            "original_mentions": entity_data_dict["original_mentions"], # This is our alias list
            "contexts": entity_data_dict["contexts"],
            "source_document_filename": filename
        }
        neo4j_node_properties = {k: v for k, v in neo4j_node_properties.items() if v is not None}

        success = await neo4j_conn.merge_entity(
            entity_type=entity_data_dict["entity_type"],
            canonical_name=entity_data_dict["canonical_name"],
            properties=neo4j_node_properties
        )
        if success:
            entities_added_neo4j += 1
            entity_text_repr = _create_entity_text_representation_for_embedding(
                canonical_name=entity_data_dict["canonical_name"],
                entity_type=entity_data_dict["entity_type"],
                aggregated_contexts=entity_data_dict["contexts"] # Use the consolidated contexts
            )
            if entity_data_dict["canonical_name"] not in faiss_conn.entity_id_to_faiss_idx:
                await faiss_conn.add_or_update_entity_embedding(entity_data_dict["canonical_name"], entity_text_repr)

    # Store relationships in Neo4j
    for rel_data_dict in consolidated_relationship_data_map.values():
        neo4j_rel_properties = {
            "contexts": rel_data_dict["contexts"],
            "source_document_filename": filename
        }
        neo4j_rel_properties = {k: v for k, v in neo4j_rel_properties.items() if v is not None}

        source_entity_type = consolidated_entity_data_map.get(rel_data_dict["source_canonical_name"], {}).get("entity_type", "Unknown")
        target_entity_type = consolidated_entity_data_map.get(rel_data_dict["target_canonical_name"], {}).get("entity_type", "Unknown")

        if source_entity_type == "Unknown" or target_entity_type == "Unknown":
            logger.warning(f"Could not determine type for source/target of relationship: {rel_data_dict}. Skipping.")
            continue

        success = await neo4j_conn.merge_relationship(
            s_type=source_entity_type,
            s_name=rel_data_dict["source_canonical_name"],
            t_type=target_entity_type,
            t_name=rel_data_dict["target_canonical_name"],
            r_type=rel_data_dict["relationship_type"],
            props=neo4j_rel_properties
        )
        if success:
            rels_added_neo4j += 1

    logger.info(f"Ingestion for {filename} completed. Neo4j: {entities_added_neo4j} entities, {rels_added_neo4j} relationships potentially merged/created.")
    return IngestionStatus(
        filename=filename,
        status="Completed",
        message="Document processed successfully.",
        entities_added=entities_added_neo4j,
        relationships_added=rels_added_neo4j
    )

# ... (rest of the file, including if __name__ == "__main__": block)
# The __main__ block for testing would remain the same.
if __name__ == "__main__":
    import asyncio
    import os # Make sure os is imported if using it in main
    from app.core.config import settings

    async def test_ingestion_service():
        print("--- Testing Ingestion Service (Revised Consolidation) ---")
        if not settings.OPENAI_API_KEY or not settings.NEO4J_URI:
            print("OPENAI_API_KEY or NEO4J_URI not set. Skipping ingestion service test.")
            return
        sample_doc_path = "sample_document.md"
        if not os.path.exists(sample_doc_path):
            print(f"Sample document '{sample_doc_path}' not found. Creating a dummy one for basic flow.")
            with open(sample_doc_path, "w", encoding="utf-8") as f:
                f.write("# Test Header\nThis is a test document for ingestion by Alpha Corp, mentioning Project Omega. Alpha Corp also uses AI.")
        if os.path.exists(sample_doc_path):
            try:
                with open(sample_doc_path, "rb") as f_content:
                    status_report = await process_document_for_ingestion(sample_doc_path, f_content)
                    print("\n--- Ingestion Status Report ---")
                    print(status_report.model_dump_json(indent=2))
                    if status_report.status == "Completed":
                        print("\nTo verify:")
                        print("1. Check Neo4j for new nodes/relationships from 'sample_document.md'.")
                        faiss_conn_test = await get_faiss_connector()
                        print(f"2. FAISS index size: {faiss_conn_test.get_index_size()}")
            except Exception as e:
                print(f"Error during ingestion service test: {e}", exc_info=True)
            finally:
                neo4j_conn_main = await get_neo4j_connector()
                await neo4j_conn_main.close_driver()
                faiss_conn_main = await get_faiss_connector()
                faiss_conn_main.save_index()
        else:
            print(f"Sample document '{sample_doc_path}' still not found. Test cannot proceed.")
    asyncio.run(test_ingestion_service())