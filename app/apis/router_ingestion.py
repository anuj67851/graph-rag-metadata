import logging
import os
import shutil
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse

from app.core.config import settings
from app.services.ingestion_service import process_document_for_ingestion
from app.database.sqlite_connector import get_sqlite_connector
from app.vector_store.weaviate_connector import get_weaviate_connector
from app.services.graph_service import safely_remove_file_references

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion & File Management"],
)

SUPPORTED_FILE_EXTENSIONS = [".txt", ".pdf", ".docx", ".md"]

@router.post("/upload_files/", status_code=status.HTTP_202_ACCEPTED)
async def upload_files_for_ingestion(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(..., description="A list of files to be ingested.")
):
    """
    Accepts files, saves them, adds a record to the DB, and schedules them for background ingestion.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")

    logger.info(f"Received {len(files)} files for ingestion.")

    # Ensure the storage directory exists
    storage_path = settings.FILE_STORAGE_PATH
    os.makedirs(storage_path, exist_ok=True)

    sqlite_conn = get_sqlite_connector()
    accepted_files = []

    for file in files:
        filename = file.filename
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension not in SUPPORTED_FILE_EXTENSIONS:
            logger.warning(f"Skipping unsupported file: {filename}")
            continue

        filepath = os.path.join(storage_path, filename)

        # Save the uploaded file to the storage directory
        try:
            with open(filepath, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            filesize = os.path.getsize(filepath)
        finally:
            file.file.close()

        # Add a record to SQLite and schedule for background processing
        if sqlite_conn.add_file_record(filename=filename, filepath=filepath, filesize=filesize, status="Accepted"):
            background_tasks.add_task(process_document_for_ingestion, filename, filepath)
            accepted_files.append(filename)
            logger.info(f"File '{filename}' accepted and scheduled for background ingestion.")
        else:
            logger.error(f"Failed to add record to SQLite for file '{filename}'.")

    if not accepted_files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="None of the provided files were of a supported type.")

    return {
        "message": "Files accepted for background processing.",
        "accepted_files": accepted_files
    }

@router.get("/documents/", response_model=List[dict])
async def list_ingested_documents():
    """Returns a list of all managed files and their metadata from the SQLite DB."""
    sqlite_conn = get_sqlite_connector()
    files = sqlite_conn.list_all_files()
    return files

@router.get("/documents/{filename}/download")
async def download_ingested_file(filename: str):
    """Allows downloading a copy of a previously ingested file."""
    sqlite_conn = get_sqlite_connector()
    record = sqlite_conn.get_file_record(filename)
    if not record or not os.path.exists(record['filepath']):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    return FileResponse(path=record['filepath'], filename=filename, media_type='application/octet-stream')

@router.delete("/documents/{filename}", status_code=status.HTTP_200_OK)
async def delete_ingested_file(filename: str, background_tasks: BackgroundTasks):
    """
    Deletes a file and all its associated data from all connected systems.
    """
    logger.info(f"Deletion requested for file: {filename}")
    sqlite_conn = get_sqlite_connector()
    weaviate_conn = get_weaviate_connector()

    record = sqlite_conn.get_file_record(filename)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File record not found in database.")

    # 1. Delete chunks from Weaviate
    num_deleted_chunks = await weaviate_conn.delete_chunks_by_filename(filename)
    logger.info(f"Deleted {num_deleted_chunks} chunks from Weaviate for file '{filename}'.")

    # 2. Schedule "safe remove" from Neo4j to run in the background
    background_tasks.add_task(safely_remove_file_references, filename)
    logger.info(f"Scheduled safe removal of Neo4j references for '{filename}'.")

    # 3. Delete the physical file from storage
    filepath = record['filepath']
    if os.path.exists(filepath):
        os.remove(filepath)
        logger.info(f"Deleted physical file: {filepath}")

    # 4. Delete the record from SQLite
    sqlite_conn.delete_file_record(filename)

    return {"detail": f"Successfully initiated deletion of file '{filename}' and all its associated data."}