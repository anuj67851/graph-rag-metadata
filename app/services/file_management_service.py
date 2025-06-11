import logging
import os
import zipfile
from io import BytesIO
from typing import List, Any, Coroutine

from app.database.sqlite_connector import get_sqlite_connector
from app.vector_store.weaviate_connector import get_weaviate_connector
from app.services.graph_service import safely_remove_file_references

logger = logging.getLogger(__name__)


async def list_all_documents() -> List[dict]:
    """Retrieves all file records from the SQLite database."""
    sqlite_conn = get_sqlite_connector()
    return sqlite_conn.list_all_files()


async def get_document_record(filename: str) -> dict[str, Any] | None:
    """Retrieves a single file record and checks for physical file existence."""
    sqlite_conn = get_sqlite_connector()
    record = sqlite_conn.get_file_record(filename)
    if not record or not os.path.exists(record['filepath']):
        return None
    return record


async def delete_document_and_all_data(filename: str):
    """
    Orchestrates the complete deletion of a document and its associated data
    from all connected systems (SQLite, Weaviate, Neo4j, and file storage).
    """
    sqlite_conn = get_sqlite_connector()
    weaviate_conn = get_weaviate_connector()

    record = sqlite_conn.get_file_record(filename)
    if not record:
        raise FileNotFoundError(f"File record for '{filename}' not found in database.")

    # 1. Delete chunks from Weaviate
    num_deleted = await weaviate_conn.delete_chunks_by_filename(filename)
    logger.info(f"Deleted {num_deleted} chunks from Weaviate for file '{filename}'.")

    # 2. Safely remove references from Neo4j
    await safely_remove_file_references(filename)
    logger.info(f"Completed safe removal of Neo4j references for '{filename}'.")

    # 3. Delete the physical file from storage
    if os.path.exists(record['filepath']):
        os.remove(record['filepath'])
        logger.info(f"Deleted physical file: {record['filepath']}")

    # 4. Delete the record from SQLite
    sqlite_conn.delete_file_record(filename)
    logger.info(f"Deleted file record for '{filename}' from SQLite.")


async def reprocess_document_from_storage(filename: str) -> str:
    """
    Orchestrates the re-processing of a document. It first deletes all existing
    data associated with the file and then triggers the ingestion pipeline again.
    """
    sqlite_conn = get_sqlite_connector()
    record = sqlite_conn.get_file_record(filename)
    if not record or not os.path.exists(record['filepath']):
        raise FileNotFoundError(f"File '{filename}' not found on disk, cannot reprocess.")

    logger.info(f"Clearing old data for '{filename}' before re-processing.")
    await get_weaviate_connector().delete_chunks_by_filename(filename)
    await safely_remove_file_references(filename)

    sqlite_conn.update_file_status(filename, "Reprocessing")

    # Return the filepath to be used for scheduling the background task
    return record['filepath']


async def prepare_batch_download(filenames: list) -> bytes:
    """Creates a ZIP archive of requested files in-memory."""
    sqlite_conn = get_sqlite_connector()
    zip_io = BytesIO()
    with zipfile.ZipFile(zip_io, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in filenames:
            record = sqlite_conn.get_file_record(filename)
            if record and os.path.exists(record['filepath']):
                zipf.write(record['filepath'], arcname=filename)
            else:
                logger.warning(f"File '{filename}' not found for zipping. Skipping.")
    zip_io.seek(0)
    return zip_io.getvalue()