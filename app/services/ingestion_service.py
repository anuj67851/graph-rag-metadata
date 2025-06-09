import logging
from typing import IO, Union, Dict, Any, Tuple

# LangChain components for semantic chunking
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.utils.file_parser import extract_text_from_file, FileParsingError
from app.llm_integration.openai_connector import extract_entities_relationships_from_chunk
from app.graph_db.neo4j_connector import get_neo4j_connector, Neo4jConnector
from app.vector_store.faiss_connector import get_faiss_connector, FaissConnector
from app.models.ingestion_models import LLMExtractionOutput, IngestionStatus

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


async def process_document_for_ingestion(filename: str, file_content: Union[bytes, IO[bytes]]) -> IngestionStatus:
    """
    Orchestrates the new ingestion pipeline for a single document.
    1. Parses the document text.
    2. Uses LangChain's SemanticChunker to create contextually coherent chunks.
    3. For each chunk, extracts entities and relationships using an LLM.
    4. Aggregates and consolidates all extracted graph data for the entire document.
    5. Merges the consolidated graph data into Neo4j.
    6. Creates embeddings for each semantic chunk and stores them in FAISS with metadata.
    """
    neo4j_conn: Neo4jConnector = await get_neo4j_connector()
    faiss_conn: FaissConnector = await get_faiss_connector()

    try:
        raw_text = extract_text_from_file(filename, file_content)
        if not raw_text or not raw_text.strip():
            return IngestionStatus(filename=filename, status="Failed", message="No text content found in file.")
    except (FileParsingError, ValueError) as e:
        return IngestionStatus(filename=filename, status="Failed", message=f"File processing error: {e}")

    # --- Step 1: Semantic Chunking ---
    # Instantiate the embedding model that the SemanticChunker will use
    embeddings = OpenAIEmbeddings(model=settings.LLM_EMBEDDING_MODEL_NAME)

    # Instantiate the SemanticChunker
    # "percentile" is a good data-driven way to find breakpoints.
    # We could also use "standard_deviation" or a fixed "distance" threshold.
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

    text_chunks = text_splitter.split_text(raw_text)

    if not text_chunks:
        return IngestionStatus(filename=filename, status="Failed", message="No text chunks could be generated.")

    logger.info(f"Processing '{filename}': {len(text_chunks)} semantic chunks generated.")

    # --- Step 2: Per-Chunk Extraction ---
    chunks_with_extractions = []
    for i, chunk in enumerate(text_chunks):
        logger.debug(f"Extracting from chunk {i+1}/{len(text_chunks)} of '{filename}'...")
        llm_raw_output = await extract_entities_relationships_from_chunk(chunk)
        if llm_raw_output:
            try:
                llm_data = LLMExtractionOutput(**llm_raw_output)
                chunks_with_extractions.append({
                    "chunk_text": chunk,
                    "extracted_data": llm_data
                })
            except Exception as e:
                logger.error(f"Error parsing LLM output for chunk {i+1} of '{filename}': {e}. Output: {str(llm_raw_output)[:500]}")
        else:
            logger.warning(f"LLM returned no output for chunk {i+1} of '{filename}'.")

    if not chunks_with_extractions:
        return IngestionStatus(filename=filename, status="Failed", message="LLM extraction failed for all chunks.")

    # --- Step 3: Document-Level Aggregation for Neo4j ---
    all_entities = [entity for item in chunks_with_extractions for entity in item["extracted_data"].entities]
    all_relationships = [rel for item in chunks_with_extractions for rel in item["extracted_data"].relationships]

    consolidated_entity_data_map: Dict[str, Dict[str, Any]] = {}
    for entity in all_entities:
        if entity.canonical_name not in consolidated_entity_data_map:
            consolidated_entity_data_map[entity.canonical_name] = {
                "canonical_name": entity.canonical_name,
                "entity_type": entity.entity_type,
                "original_mentions": {entity.original_mention},
                "contexts": set(entity.contexts)
            }
        else:
            data = consolidated_entity_data_map[entity.canonical_name]
            data["original_mentions"].add(entity.original_mention)
            data["contexts"].update(entity.contexts)

    logger.info(f"Consolidated to {len(consolidated_entity_data_map)} unique entities for '{filename}'.")

    consolidated_relationship_data_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for rel in all_relationships:
        key = (rel.source_canonical_name, rel.relationship_type, rel.target_canonical_name)
        if key not in consolidated_relationship_data_map:
            consolidated_relationship_data_map[key] = {
                "source_canonical_name": rel.source_canonical_name,
                "relationship_type": rel.relationship_type,
                "target_canonical_name": rel.target_canonical_name,
                "contexts": set(rel.contexts)
            }
        else:
            consolidated_relationship_data_map[key]["contexts"].update(rel.contexts)

    logger.info(f"Consolidated to {len(consolidated_relationship_data_map)} unique relationships for '{filename}'.")

    # --- Step 4: Store Graph Data in Neo4j ---
    entities_added_neo4j = 0
    for entity_data in consolidated_entity_data_map.values():
        props = {
            "original_mentions": list(entity_data["original_mentions"]),
            "contexts": list(entity_data["contexts"]),
            "source_document_filename": filename
        }
        success = await neo4j_conn.merge_entity(
            entity_type=entity_data["entity_type"],
            canonical_name=entity_data["canonical_name"],
            properties=props
        )
        if success:
            entities_added_neo4j += 1

    rels_added_neo4j = 0
    for rel_data in consolidated_relationship_data_map.values():
        source_type = consolidated_entity_data_map.get(rel_data["source_canonical_name"], {}).get("entity_type", "Unknown")
        target_type = consolidated_entity_data_map.get(rel_data["target_canonical_name"], {}).get("entity_type", "Unknown")
        if source_type == "Unknown" or target_type == "Unknown":
            logger.warning(f"Skipping relationship due to unknown entity type: {rel_data}")
            continue

        props = {"contexts": list(rel_data["contexts"]), "source_document_filename": filename}
        success = await neo4j_conn.merge_relationship(
            s_type=source_type, s_name=rel_data["source_canonical_name"],
            t_type=target_type, t_name=rel_data["target_canonical_name"],
            r_type=rel_data["relationship_type"], props=props
        )
        if success:
            rels_added_neo4j += 1

    logger.info(f"Neo4j merge completed for '{filename}': {entities_added_neo4j} entities, {rels_added_neo4j} relationships.")

    # --- Step 5: Store Chunks in FAISS Vector DB ---
    faiss_chunks_batch = []
    for item in chunks_with_extractions:
        entity_ids_in_chunk = [entity.canonical_name for entity in item["extracted_data"].entities]
        faiss_chunks_batch.append({
            "chunk_text": item["chunk_text"],
            "source_document": filename,
            "entity_ids": list(set(entity_ids_in_chunk))
        })

    await faiss_conn.add_chunk_embeddings_batch(faiss_chunks_batch)
    logger.info(f"Submitted {len(faiss_chunks_batch)} chunks to FAISS for '{filename}'.")

    return IngestionStatus(
        filename=filename,
        status="Completed",
        message="Document processed successfully.",
        entities_added=entities_added_neo4j,
        relationships_added=rels_added_neo4j
    )

if __name__ == "__main__":
    import asyncio
    import os

    # Note: For this test to run, you will need to install langchain:
    # pip install langchain langchain-openai

    async def test_ingestion_service():
        print("--- Testing Ingestion Service (with Semantic Chunker) ---")
        if not settings.OPENAI_API_KEY or not settings.NEO4J_URI:
            print("OPENAI_API_KEY or NEO4J_URI not set. Skipping test.")
            return

        sample_doc_path = "sample_document.md"
        if not os.path.exists(sample_doc_path):
            print(f"Sample document '{sample_doc_path}' not found. Creating a dummy one.")
            with open(sample_doc_path, "w", encoding="utf-8") as f:
                f.write("# Project Report\n\nThis is a test document about Alpha Corp. Alpha Corp is working on Project Omega with their AI team. The AI team is led by Dr. Evelyn Reed. \n\nIn a separate development, Beta Systems announced a new partnership with Gamma Tech for cloud services.")

        try:
            with open(sample_doc_path, "rb") as f_content:
                status_report = await process_document_for_ingestion(sample_doc_path, f_content)
                print("\n--- Ingestion Status Report ---")
                print(status_report.model_dump_json(indent=2))

                if status_report.status == "Completed":
                    print("\nVerification:")
                    print("1. Check Neo4j for nodes and relationships.")
                    faiss_conn_test = await get_faiss_connector()
                    print(f"2. FAISS index should contain new chunks. Current size: {faiss_conn_test.get_index_size()}")
        except Exception as e:
            print(f"Error during ingestion service test: {e}", exc_info=True)
        finally:
            neo4j_conn_main = await get_neo4j_connector()
            await neo4j_conn_main.close_driver()
            faiss_conn_main = await get_faiss_connector()
            faiss_conn_main.save()

    asyncio.run(test_ingestion_service())