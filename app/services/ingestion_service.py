import logging
from typing import Dict, Any, Tuple

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.utils.file_parser import extract_text_from_file, FileParsingError
from app.llm_integration.openai_connector import extract_entities_relationships_from_chunk
from app.graph_db.neo4j_connector import get_neo4j_connector, Neo4jConnector
from app.vector_store.weaviate_connector import get_weaviate_connector, WeaviateConnector
from app.database.sqlite_connector import get_sqlite_connector, SQLiteConnector
from app.models.ingestion_models import LLMExtractionOutput, IngestionStatus

logger = logging.getLogger(__name__)

async def process_document_for_ingestion(filename: str, filepath: str) -> IngestionStatus:
    """
    Orchestrates the new ingestion pipeline for a single document.
    1.  Updates file status in SQLite to "Processing".
    2.  Parses text, chunks it semantically.
    3.  Extracts entities/relationships from each chunk via LLM.
    4.  Aggregates and consolidates graph data for the document.
    5.  Merges consolidated data into Neo4j.
    6.  Adds chunks to Weaviate for vector search.
    7.  Updates final status in SQLite (Completed or Failed).
    """
    # Get all necessary connectors from our singleton providers
    sqlite_conn: SQLiteConnector = get_sqlite_connector()
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()
    weaviate_conn: WeaviateConnector = get_weaviate_connector()

    try:
        # --- Step 1: Initial Status Update ---
        sqlite_conn.update_file_status(filename, "Processing")

        # --- Step 2: Text Extraction and Semantic Chunking ---
        with open(filepath, "rb") as file_content: # <-- This is the main fix
            raw_text = extract_text_from_file(filename, file_content)
        if not raw_text or not raw_text.strip():
            message = "No text content found in file."
            sqlite_conn.update_file_status(filename, "Failed", error_message=message)
            return IngestionStatus(filename=filename, status="Failed", message=message)

        embeddings_model = OpenAIEmbeddings(model=settings.LLM_EMBEDDING_MODEL_NAME)
        text_splitter = SemanticChunker(embeddings_model, breakpoint_threshold_type="percentile")
        text_chunks = text_splitter.split_text(raw_text)

        if not text_chunks:
            message = "No text chunks could be generated from the document."
            sqlite_conn.update_file_status(filename, "Failed", error_message=message)
            return IngestionStatus(filename=filename, status="Failed", message=message)
        logger.info(f"Processing '{filename}': {len(text_chunks)} semantic chunks generated.")

        # --- Step 3: Per-Chunk Graph Extraction ---
        chunks_with_extractions = []
        for i, chunk in enumerate(text_chunks):
            logger.debug(f"Extracting from chunk {i+1}/{len(text_chunks)} of '{filename}'...")
            llm_raw_output = await extract_entities_relationships_from_chunk(chunk)
            if llm_raw_output:
                try:
                    llm_data = LLMExtractionOutput(**llm_raw_output)
                    chunks_with_extractions.append({"chunk_text": chunk, "extracted_data": llm_data})
                except Exception as e:
                    logger.error(f"Error parsing LLM output for chunk {i+1}: {e}. Output: {str(llm_raw_output)[:300]}")
            else:
                logger.warning(f"LLM returned no output for chunk {i+1} of '{filename}'.")

        if not chunks_with_extractions:
            message = "LLM extraction failed for all chunks. No graph data to add."
            sqlite_conn.update_file_status(filename, "Failed", error_message=message)
            return IngestionStatus(filename=filename, status="Failed", message=message)

        # --- Step 4: Document-Level Aggregation for Neo4j ---
        all_entities = [entity for item in chunks_with_extractions for entity in item["extracted_data"].entities]
        all_relationships = [rel for item in chunks_with_extractions for rel in item["extracted_data"].relationships]

        consolidated_entity_data_map = _consolidate_entities(all_entities)
        logger.info(f"Consolidated to {len(consolidated_entity_data_map)} unique entities for '{filename}'.")

        consolidated_relationship_data_map = _consolidate_relationships(all_relationships)
        logger.info(f"Consolidated to {len(consolidated_relationship_data_map)} unique relationships for '{filename}'.")

        # --- Step 5: Store Graph Data in Neo4j ---
        entities_added_count = 0
        for entity_data in consolidated_entity_data_map.values():
            props = {
                "original_mentions": list(entity_data["original_mentions"]),
                "contexts": list(entity_data["contexts"]),
                "source_document_filename": filename # Add file source to node
            }
            if await neo4j_conn.merge_entity(entity_data["entity_type"], entity_data["canonical_name"], props):
                entities_added_count += 1

        rels_added_count = 0
        for rel_data in consolidated_relationship_data_map.values():
            # This logic ensures we only create relationships between entities we have processed
            source_type = consolidated_entity_data_map.get(rel_data["source_canonical_name"], {}).get("entity_type")
            target_type = consolidated_entity_data_map.get(rel_data["target_canonical_name"], {}).get("entity_type")

            if not source_type or not target_type:
                logger.warning(f"Skipping relationship due to missing entity: {rel_data}")
                continue

            props = {"contexts": list(rel_data["contexts"]), "source_document_filename": filename}
            if await neo4j_conn.merge_relationship(source_type, rel_data["source_canonical_name"], target_type, rel_data["target_canonical_name"], rel_data["relationship_type"], props):
                rels_added_count += 1
        logger.info(f"Neo4j merge completed for '{filename}': {entities_added_count} entities, {rels_added_count} relationships.")

        # --- Step 6: Store Chunks in Weaviate ---
        weaviate_batch_data = []
        for item in chunks_with_extractions:
            entity_ids_in_chunk = [entity.canonical_name for entity in item["extracted_data"].entities]
            weaviate_batch_data.append({
                "chunk_text": item["chunk_text"],
                "source_document": filename,
                "entity_ids": list(set(entity_ids_in_chunk))
            })
        await weaviate_conn.add_chunk_batch(weaviate_batch_data)
        logger.info(f"Submitted {len(weaviate_batch_data)} chunks to Weaviate for '{filename}'.")

        # --- Step 7: Final Status Update ---
        final_status = IngestionStatus(
            filename=filename,
            status="Completed",
            message="Document processed and ingested successfully.",
            entities_added=entities_added_count,
            relationships_added=rels_added_count
        )
        sqlite_conn.update_file_status(
            filename,
            status="Completed",
            chunk_count=len(weaviate_batch_data),
            entities_added=entities_added_count,
            relationships_added=rels_added_count,
            error_message=None
        )
        return final_status

    except (FileParsingError, ValueError) as e:
        logger.error(f"File parsing error for '{filename}': {e}", exc_info=True)
        sqlite_conn.update_file_status(filename, "Failed", error_message=str(e))
        return IngestionStatus(filename=filename, status="Failed", message=f"File processing error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during ingestion of '{filename}': {e}", exc_info=True)
        sqlite_conn.update_file_status(filename, "Failed", error_message=f"Unexpected error: {e}")
        return IngestionStatus(filename=filename, status="Failed", message=f"An unexpected error occurred: {e}")

def _consolidate_entities(entities: list) -> Dict[str, Dict[str, Any]]:
    """Helper function to aggregate entity data from all chunks."""
    consolidated = {}
    for entity in entities:
        if entity.canonical_name not in consolidated:
            consolidated[entity.canonical_name] = {
                "canonical_name": entity.canonical_name,
                "entity_type": entity.entity_type,
                "original_mentions": {entity.original_mention},
                "contexts": set(entity.contexts)
            }
        else:
            data = consolidated[entity.canonical_name]
            data["original_mentions"].add(entity.original_mention)
            data["contexts"].update(entity.contexts)
    return consolidated

def _consolidate_relationships(relationships: list) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Helper function to aggregate relationship data from all chunks."""
    consolidated = {}
    for rel in relationships:
        key = (rel.source_canonical_name, rel.relationship_type, rel.target_canonical_name)
        if key not in consolidated:
            consolidated[key] = {
                "source_canonical_name": rel.source_canonical_name,
                "relationship_type": rel.relationship_type,
                "target_canonical_name": rel.target_canonical_name,
                "contexts": set(rel.contexts)
            }
        else:
            consolidated[key]["contexts"].update(rel.contexts)
    return consolidated