import logging
import os
from typing import IO, Union, Dict, List, Any, Tuple

from app.utils.file_parser import extract_text_from_file, FileParsingError
from app.llm_integration.openai_connector import extract_entities_relationships_from_chunk
from app.graph_db.neo4j_connector import get_neo4j_connector, Neo4jConnector
from app.vector_store.faiss_connector import get_faiss_connector, FaissConnector
from app.models.ingestion_models import LLMExtractionOutput, ExtractedEntity, ExtractedRelationship, IngestionStatus

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Basic text chunking (can be made more sophisticated)
# LLMs have context limits, so processing very large texts in one go might fail or be inefficient.
# GPT-4o-mini has a large context window (128k tokens), but smaller chunks might still be preferable for cost/latency/accuracy.
DEFAULT_CHUNK_SIZE = 2000  # Characters, approximate. Can be tuned.
DEFAULT_CHUNK_OVERLAP = 200 # Characters

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into chunks of a specified size with overlap.
    A simple implementation. More advanced methods (e.g., sentence splitting) could be used.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= text_len:
            break
        # Move start for the next chunk, considering overlap
        # Ensure overlap doesn't make start go backward if chunk_size is small
        start += (chunk_size - chunk_overlap)
        if start >= end : # if overlap is too large or chunk_size too small
            start = end # ensure progression

    return chunks


def _create_entity_text_representation_for_embedding(entity: ExtractedEntity) -> str:
    """
    Creates a rich text representation for an entity to be used for generating its embedding.
    (As defined in Feature 3 for FAISS indexing).
    """
    aliases_str = ", ".join(entity.aliases) if hasattr(entity, 'aliases') and entity.aliases else "N/A"
    contexts_str = " ".join(entity.contexts) if entity.contexts else "No context provided."

    # Keep it concise but informative
    representation = (
        f"Entity Name: {entity.canonical_name}. "
        f"Type: {entity.entity_type}. "
        # f"Aliases: {aliases_str}. " # Aliases can make it too long, consider if essential for matching
        f"Description: {contexts_str}"
    )
    return representation


async def process_document_for_ingestion(filename: str, file_content: Union[bytes, IO[bytes]]) -> IngestionStatus:
    """
    Main service function to process a single document for ingestion.
    1. Parses file to text.
    2. Chunks text.
    3. Extracts entities/relationships from chunks using LLM.
    4. Consolidates extracted data.
    5. Stores in Neo4j.
    6. Adds/updates embeddings in FAISS.
    """
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()
    faiss_conn: FaissConnector = await get_faiss_connector()

    # 1. Parse file to text
    try:
        raw_text = extract_text_from_file(filename, file_content)
        if not raw_text or not raw_text.strip():
            logger.warning(f"No text extracted from file: {filename}")
            return IngestionStatus(filename=filename, status="Failed", message="No text content found in file.")
    except FileParsingError as e:
        logger.error(f"File parsing error for {filename}: {e}")
        return IngestionStatus(filename=filename, status="Failed", message=f"File parsing error: {e}")
    except ValueError as e: # Unsupported file type
        logger.error(f"Unsupported file type for {filename}: {e}")
        return IngestionStatus(filename=filename, status="Failed", message=str(e))

    # 2. Chunk text
    # Using a simple chunk_text function here.
    # For production, consider more sophisticated chunking strategies (e.g., semantic chunking, Langchain's text_splitters).
    text_chunks = chunk_text(raw_text)
    if not text_chunks:
        logger.warning(f"Text from {filename} resulted in no processable chunks.")
        return IngestionStatus(filename=filename, status="Failed", message="No text chunks to process after parsing.")

    logger.info(f"Processing {filename}: {len(text_chunks)} chunks generated.")

    # 3. Extract entities/relationships from chunks
    all_extracted_entities: List[ExtractedEntity] = []
    all_extracted_relationships: List[ExtractedRelationship] = []

    for i, chunk in enumerate(text_chunks):
        logger.debug(f"Processing chunk {i+1}/{len(text_chunks)} for {filename}...")
        llm_raw_output = await extract_entities_relationships_from_chunk(chunk)

        if llm_raw_output:
            try:
                # Validate and parse using Pydantic model
                llm_data = LLMExtractionOutput(**llm_raw_output)
                all_extracted_entities.extend(llm_data.entities)
                all_extracted_relationships.extend(llm_data.relationships)
                logger.debug(f"Chunk {i+1}: Extracted {len(llm_data.entities)} entities, {len(llm_data.relationships)} relationships.")
            except Exception as e: # Pydantic ValidationError or other issues
                logger.error(f"Error parsing LLM output for chunk {i+1} of {filename}: {e}. Output: {str(llm_raw_output)[:500]}")
        else:
            logger.warning(f"LLM returned no output for chunk {i+1} of {filename}.")

    # 4. Consolidate extracted data (Intra-file deduplication and aggregation)
    # Entities: Group by canonical_name, merge aliases and contexts. Type taken from first encounter.
    consolidated_entities_dict: Dict[str, ExtractedEntity] = {}
    for extr_entity in all_extracted_entities:
        if extr_entity.canonical_name not in consolidated_entities_dict:
            # Create a new Pydantic model instance for the consolidated entity
            # to ensure 'aliases' exists if it's part of ExtractedEntity (it's not by default in current model)
            # Let's assume ExtractedEntity model handles its fields well.
            # We need to manage aliases and contexts accumulation.
            new_entity_data = extr_entity.model_dump()
            new_entity_data['original_mentions_collated'] = [extr_entity.original_mention] # Store all original mentions
            new_entity_data['contexts_collated'] = list(set(extr_entity.contexts)) # Unique contexts
            consolidated_entities_dict[extr_entity.canonical_name] = ExtractedEntity(**new_entity_data)

        else:
            existing_entity = consolidated_entities_dict[extr_entity.canonical_name]
            # Merge original_mentions (as aliases)
            if hasattr(existing_entity, 'original_mentions_collated'):
                existing_entity.original_mentions_collated.append(extr_entity.original_mention)
            else: # Initialize if not present (should be due to above logic)
                existing_entity.original_mentions_collated = [existing_entity.original_mention, extr_entity.original_mention]

            # Merge contexts (unique)
            if hasattr(existing_entity, 'contexts_collated'):
                existing_entity.contexts_collated.extend(extr_entity.contexts)
                existing_entity.contexts_collated = list(set(existing_entity.contexts_collated))
            else:
                existing_entity.contexts_collated = list(set(existing_entity.contexts + extr_entity.contexts))

            # Update main contexts field with the collated one for consistency if needed by ExtractedEntity model,
            # or use contexts_collated directly for storage.
            # For now, ExtractedEntity has 'contexts', so let's assume 'contexts_collated' is for internal use here.
            # The `properties` for Neo4j should get these aggregated lists.

    consolidated_entities: List[ExtractedEntity] = list(consolidated_entities_dict.values())
    logger.info(f"Consolidated to {len(consolidated_entities)} unique entities for {filename}.")

    # Relationships: Already based on canonical_names. We might want to deduplicate identical relationships
    # (same source, target, type) and merge their contexts if they arise from different chunks.
    consolidated_relationships_dict: Dict[Tuple[str, str, str], ExtractedRelationship] = {}
    for extr_rel in all_extracted_relationships:
        rel_key = (extr_rel.source_canonical_name, extr_rel.relationship_type, extr_rel.target_canonical_name)
        if rel_key not in consolidated_relationships_dict:
            new_rel_data = extr_rel.model_dump()
            new_rel_data['contexts_collated'] = list(set(extr_rel.contexts))
            consolidated_relationships_dict[rel_key] = ExtractedRelationship(**new_rel_data)
        else:
            existing_rel = consolidated_relationships_dict[rel_key]
            if hasattr(existing_rel, 'contexts_collated'):
                existing_rel.contexts_collated.extend(extr_rel.contexts)
                existing_rel.contexts_collated = list(set(existing_rel.contexts_collated))
            else:
                existing_rel.contexts_collated = list(set(existing_rel.contexts + extr_rel.contexts))


    consolidated_relationships: List[ExtractedRelationship] = list(consolidated_relationships_dict.values())
    logger.info(f"Consolidated to {len(consolidated_relationships)} unique relationships for {filename}.")


    # 5. Store in Neo4j
    entities_added_neo4j = 0
    rels_added_neo4j = 0

    for entity in consolidated_entities:
        # Prepare properties for Neo4j node
        node_properties = {
            "original_mentions": list(set(getattr(entity, 'original_mentions_collated', [entity.original_mention]))),
            "contexts": list(set(getattr(entity, 'contexts_collated', entity.contexts))),
            "source_document_filename": filename
            # Add other properties from entity if they are not part of canonical_name or entity_type directly
        }
        # Filter out any None values from properties before sending to Neo4j
        node_properties = {k: v for k, v in node_properties.items() if v is not None}

        success = await neo4j_conn.merge_entity(
            entity_type=entity.entity_type,
            canonical_name=entity.canonical_name,
            properties=node_properties
        )
        if success:
            entities_added_neo4j += 1

            # 6. Add/update embeddings in FAISS for this entity
            # Use the consolidated entity data for richer text representation
            entity_text_repr = _create_entity_text_representation_for_embedding(entity)
            # The faiss_connector's add_or_update currently skips if ID exists.
            # This means only truly new entities (by canonical_name) get added.
            # If an entity's context changes, its embedding won't update here without re-indexing.
            if entity.canonical_name not in faiss_conn.entity_id_to_faiss_idx: # Check before calling
                await faiss_conn.add_or_update_entity_embedding(entity.canonical_name, entity_text_repr)
            # else:
            # logger.debug(f"Entity {entity.canonical_name} already in FAISS, embedding not updated by this ingestion.")


    for rel in consolidated_relationships:
        rel_properties = {
            "contexts": list(set(getattr(rel, 'contexts_collated', rel.contexts))),
            "source_document_filename": filename
        }
        rel_properties = {k: v for k, v in rel_properties.items() if v is not None}

        # Need to ensure source/target entities exist or handle gracefully.
        # MERGE in Neo4j for relationships will only work if source/target nodes are found by MATCH.
        # We assume LLM provides valid entity types for source/target from its extraction,
        # or we need a lookup mechanism if types are not directly on relationship object.
        # For simplicity, we need to find the types of source/target entities from our consolidated list.
        source_entity_type = consolidated_entities_dict.get(rel.source_canonical_name, ExtractedEntity(original_mention="", entity_type="Unknown", canonical_name="", contexts=[])).entity_type
        target_entity_type = consolidated_entities_dict.get(rel.target_canonical_name, ExtractedEntity(original_mention="", entity_type="Unknown", canonical_name="", contexts=[])).entity_type

        if source_entity_type == "Unknown" or target_entity_type == "Unknown":
            logger.warning(f"Could not determine type for source/target of relationship: {rel.model_dump()}. Skipping.")
            continue

        success = await neo4j_conn.merge_relationship(
            source_entity_type=source_entity_type,
            source_canonical_name=rel.source_canonical_name,
            target_entity_type=target_entity_type,
            target_canonical_name=rel.target_canonical_name,
            relationship_type=rel.relationship_type,
            properties=rel_properties
        )
        if success:
            rels_added_neo4j += 1

    # FAISS index should be saved periodically or on shutdown, handled by faiss_connector lifecycle.

    logger.info(f"Ingestion for {filename} completed. Neo4j: {entities_added_neo4j} entities, {rels_added_neo4j} relationships potentially merged/created.")
    return IngestionStatus(
        filename=filename,
        status="Completed",
        message="Document processed successfully.",
        entities_added=entities_added_neo4j, # This is more like "entities processed for Neo4j"
        relationships_added=rels_added_neo4j
    )


if __name__ == "__main__":
    import asyncio
    from app.core.config import settings # Ensure settings are loaded

    async def test_ingestion_service():
        # Ensure all dependent modules (connectors, utils) are testable
        # and external services (OpenAI, Neo4j, FAISS writable directory) are available.
        # This is more of an integration test for the service.
        print("--- Testing Ingestion Service ---")

        if not settings.OPENAI_API_KEY or not settings.NEO4J_URI:
            print("OPENAI_API_KEY or NEO4J_URI not set. Skipping ingestion service test.")
            return

        # Use the sample_document.md from project root
        sample_doc_path = "sample_document.md"
        if not os.path.exists(sample_doc_path):
            print(f"Sample document '{sample_doc_path}' not found. Skipping test.")
            # Create a dummy one for basic test flow
            # with open(sample_doc_path, "w") as f:
            #     f.write("# Test Header\nThis is a test document for ingestion by Alpha Corp, mentioning Project Omega.")
            # print(f"Created dummy '{sample_doc_path}' for testing.")

        if os.path.exists(sample_doc_path):
            try:
                with open(sample_doc_path, "rb") as f_content: # Read as bytes
                    status_report = await process_document_for_ingestion(sample_doc_path, f_content)
                    print("\n--- Ingestion Status Report ---")
                    print(status_report.model_dump_json(indent=2))

                    # Verify data in Neo4j and FAISS (manually or with more test code)
                    if status_report.status == "Completed":
                        print("\nTo verify:")
                        print("1. Check Neo4j for new nodes/relationships from 'sample_document.md'.")
                        print("   Example Cypher: MATCH (n) WHERE n.source_document_filename = 'sample_document.md' RETURN n LIMIT 5")
                        faiss_conn_test = await get_faiss_connector()
                        print(f"2. FAISS index size: {faiss_conn_test.get_index_size()}")
                        # If you know some canonical names expected from sample_document.md:
                        # print(f"   Is 'Alpha Corp' in FAISS? {'Alpha Corp' in faiss_conn_test.entity_id_to_faiss_idx}")
                        # print(f"   Is 'Project Nova' in FAISS? {'Project Nova' in faiss_conn_test.entity_id_to_faiss_idx}")

            except Exception as e:
                print(f"Error during ingestion service test: {e}", exc_info=True)
            finally:
                # Clean up external services (Neo4j driver, FAISS save on shutdown is handled by main app lifecycle)
                neo4j_conn_main = await get_neo4j_connector()
                await neo4j_conn_main.close_driver()
                faiss_conn_main = await get_faiss_connector()
                faiss_conn_main.save_index() # Manual save for standalone test
        else:
            print(f"Sample document '{sample_doc_path}' still not found. Test cannot proceed.")


    # Ensure Neo4j is running, OpenAI API key is set.
    # The FAISS index files will be created in data/vector_store/
    asyncio.run(test_ingestion_service())