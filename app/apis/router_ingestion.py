import logging
import os
import shutil
import time
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.core.config import settings
from app.services.ingestion_service import process_document_for_ingestion
from app.services.file_management_service import (
    list_all_documents,
    get_document_record,
    delete_document_and_all_data,
    reprocess_document_from_storage,
    prepare_batch_download,
)
from app.database.sqlite_connector import get_sqlite_connector

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion & File Management"],
    responses={
        404: {"description": "Resource not found."},
        500: {"description": "Internal Server Error."}
    }
)

SUPPORTED_FILE_EXTENSIONS = [".txt", ".pdf", ".docx", ".md"]

class FileDownloadRequest(BaseModel):
    filenames: List[str]

@router.post(
    "/upload_files/",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload one or more documents for ingestion.",
    response_description="A confirmation message indicating which files were accepted for background processing."
)
async def upload_files_for_ingestion(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(..., description="A list of files (.txt, .pdf, .docx, .md) to be ingested.")
):
    """
    Accepts one or more files, performs the following actions:
    1.  Validates the file extension.
    2.  Saves the file to the configured storage path.
    3.  Creates an initial metadata record in the database with 'Accepted' status.
    4.  Schedules a background task to process each valid file through the full ingestion pipeline.

    The API returns immediately while processing happens in the background.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")

    storage_path = settings.FILE_STORAGE_PATH
    os.makedirs(storage_path, exist_ok=True)
    # This is the only direct connector call left in a router, for speed on upload
    sqlite_conn = get_sqlite_connector()
    accepted_files, skipped_files = [], []

    for file in files:
        if os.path.splitext(file.filename)[1].lower() not in SUPPORTED_FILE_EXTENSIONS:
            logger.warning(f"Skipping unsupported file: {file.filename}")
            skipped_files.append(file.filename)
            continue

        filepath = os.path.join(storage_path, file.filename)
        try:
            with open(filepath, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            filesize = os.path.getsize(filepath)
        finally:
            file.file.close()

        if sqlite_conn.add_file_record(filename=file.filename, filepath=filepath, filesize=filesize, status="Accepted"):
            background_tasks.add_task(process_document_for_ingestion, file.filename, filepath)
            accepted_files.append(file.filename)
        else:
            logger.error(f"Failed to add record to SQLite for file '{file.filename}'.")
            skipped_files.append(file.filename)

    if not accepted_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"None of the provided files were of a supported type. Supported: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
        )

    return {
        "message": f"{len(accepted_files)} of {len(files)} files were accepted for background processing.",
        "accepted_files": accepted_files,
        "skipped_files": skipped_files
    }

@router.get("/documents/", response_model=List[dict], summary="List all managed documents.")
async def list_ingested_documents():
    """Returns a list of all documents currently tracked by the system and their metadata from the database."""
    return await list_all_documents()

@router.get("/documents/{filename}/status", response_model=dict, summary="Get the status of a specific document.")
async def get_document_status(filename: str):
    """Returns the current ingestion status and detailed metadata for a single file."""
    record = await get_document_record(filename)
    if not record:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    return record

@router.get("/documents/{filename}/download", summary="Download an original document.")
async def download_ingested_file(filename: str):
    """Allows downloading a copy of a previously ingested file from the system's storage."""
    record = await get_document_record(filename)
    if not record:
        raise HTTPException(status_code=404, detail="File not found in database or on disk.")
    return FileResponse(path=record['filepath'], filename=filename, media_type='application/octet-stream')

@router.delete("/documents/{filename}", status_code=status.HTTP_200_OK, summary="Delete a document and all its data.")
async def delete_ingested_file(filename: str):
    """
    **WARNING: This is a destructive operation.**

    Deletes a file and all its associated data from all connected systems:
    - Removes all vector chunks from Weaviate.
    - Safely removes all graph nodes and relationships from Neo4j.
    - Deletes the original file from storage.
    - Deletes the metadata record from the database.
    """
    try:
        await delete_document_and_all_data(filename)
        return {"detail": f"Successfully initiated deletion of file '{filename}' and all associated data."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error during deletion of '{filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during deletion: {e}")

@router.post("/documents/download/batch", summary="Download multiple documents as a ZIP file.")
async def download_files_as_zip(request: FileDownloadRequest):
    """Accepts a list of filenames, creates a ZIP archive of them in memory, and streams it for download."""
    if not request.filenames:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filenames provided.")

    zip_content = await prepare_batch_download(request.filenames)
    zip_filename = f"GraphRAG_Documents_{int(time.time())}.zip"

    return StreamingResponse(
        iter([zip_content]),
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )

@router.post("/documents/{filename}/reprocess", status_code=status.HTTP_202_ACCEPTED, summary="Re-process a document.")
async def reprocess_document(filename: str, background_tasks: BackgroundTasks):
    """
    Triggers a full re-processing of an already uploaded document. This is useful
    after updating ingestion logic or LLM prompts.

    This performs a delete-and-re-ingest operation:
    1.  Deletes all existing data for the file (vectors, graph entities).
    2.  Schedules a new background task to ingest the file from scratch using the same file on disk.
    """
    try:
        filepath = await reprocess_document_from_storage(filename)
        background_tasks.add_task(process_document_for_ingestion, filename, filepath)
        return {"detail": f"Successfully scheduled '{filename}' for re-processing."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error during reprocessing of '{filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during reprocessing: {e}")