import logging
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.services.ingestion_service import process_document_for_ingestion
from app.models.ingestion_models import IngestionStatus

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Create an APIRouter instance for ingestion-related endpoints
# The prefix and tags will be used when including this router in the main FastAPI app.
router = APIRouter(
    # prefix=f"{settings.API_V1_STR}/ingest", # Using API_V1_STR from settings
    prefix="/ingest", # Simpler prefix for now, can adjust with settings
    tags=["Ingestion"],
    responses={404: {"description": "Not found"}},
)

# In-memory store for tracking status of background ingestion tasks (simple example)
# For a production system, use a more robust task queue like Celery with a message broker (Redis/RabbitMQ)
# or FastAPI's built-in BackgroundTasks for simpler, non-critical tasks.
# BackgroundTasks are executed in the same process after the response is sent.
# They are not suitable for long-running, CPU-intensive, or critical tasks that need guarantees.
# Given our LLM calls and DB interactions, ingestion can be time-consuming.
# For now, we will run it directly and the client will wait.
# BackgroundTasks example is commented out for now.

# task_status_db: Dict[str, IngestionStatus] = {}

# async def background_ingest_task(filename: str, file_content: bytes):
#     """ Helper for background processing. """
#     logger.info(f"Background task started for: {filename}")
#     status_report = await process_document_for_ingestion(filename, file_content)
#     task_status_db[filename] = status_report
#     logger.info(f"Background task finished for {filename}. Status: {status_report.status}")


@router.post(
    "/upload_file/",
    response_model=IngestionStatus,
    summary="Upload a single file for ingestion into the knowledge graph.",
    description=(
            "Accepts a single file (PDF, TXT, DOCX, MD). "
            "The file is processed to extract text, identify entities and relationships, "
            "and store them in the Neo4j graph and FAISS vector store. "
            "This operation can be time-consuming depending on file size and LLM processing."
    )
)
async def upload_file_for_ingestion(
        # background_tasks: BackgroundTasks, # If using FastAPI BackgroundTasks
        file: UploadFile = File(..., description="The file to be ingested.")
):
    """
    Endpoint to upload and process a single document.
    The processing happens synchronously in this version.
    """
    filename = file.filename
    logger.info(f"Received file for ingestion: {filename}")

    if not filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename cannot be empty.")

    # Supported file types based on file_parser
    supported_extensions = [".txt", ".pdf", ".docx", ".md"]
    file_extension = "." + filename.split(".")[-1].lower() if "." in filename else ""

    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}. Supported types are: {', '.join(supported_extensions)}"
        )

    try:
        file_content: bytes = await file.read() # Read file content into memory
        logger.info(f"File '{filename}' read into memory, size: {len(file_content)} bytes.")

        # --- Synchronous Processing (client waits) ---
        status_report: IngestionStatus = await process_document_for_ingestion(filename, file_content)

        if status_report.status == "Failed":
            # Determine appropriate HTTP status code based on failure reason
            # For now, let's use 500 if processing failed internally, 400 if it was a bad file.
            # FileParsingError or ValueError from service might indicate a 400 or 422.
            # Other errors (LLM, DB) might be 500.
            # The IngestionStatus message should have details.
            http_status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            if "parsing error" in status_report.message.lower() or \
                    "unsupported file type" in status_report.message.lower() or \
                    "no text content found" in status_report.message.lower():
                http_status_code = status.HTTP_422_UNPROCESSABLE_ENTITY # Or 400

            raise HTTPException(status_code=http_status_code, detail=status_report.message)

        logger.info(f"Ingestion successful for {filename}. Entities: {status_report.entities_added}, Rels: {status_report.relationships_added}")
        return status_report

        # --- Asynchronous Processing with BackgroundTasks (example) ---
        # # This is suitable for tasks that don't need immediate feedback beyond "accepted".
        # # Client gets an immediate response, and task runs in background.
        # # Status polling or websockets would be needed for client to get final status.
        #
        # initial_status = IngestionStatus(
        #     filename=filename,
        #     status="Accepted",
        #     message="File accepted for background processing.",
        #     entities_added=0,
        #     relationships_added=0
        # )
        # task_status_db[filename] = initial_status # Store initial status
        #
        # background_tasks.add_task(background_ingest_task, filename, file_content)
        # logger.info(f"File '{filename}' submitted for background ingestion.")
        # return initial_status # Return accepted status immediately

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during file upload for {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing the file: {str(e)}"
        )
    finally:
        await file.close() # Ensure file is closed


@router.post(
    "/upload_files/",
    response_model=List[IngestionStatus],
    summary="Upload multiple files for ingestion into the knowledge graph.",
    description=(
            "Accepts a list of files (PDF, TXT, DOCX, MD). Each file is processed sequentially. "
            "This can be a very long-running operation if many or large files are uploaded."
    )
)
async def upload_multiple_files_for_ingestion(
        files: List[UploadFile] = File(..., description="A list of files to be ingested.")
):
    """
    Endpoint to upload and process multiple documents.
    Processes files sequentially and synchronously in this version.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")

    logger.info(f"Received {len(files)} files for ingestion.")
    status_reports: List[IngestionStatus] = []

    for file in files:
        filename = file.filename
        logger.info(f"Processing file from batch: {filename}")

        if not filename:
            status_reports.append(IngestionStatus(filename="Unknown_File", status="Failed", message="Filename cannot be empty."))
            continue

        supported_extensions = [".txt", ".pdf", ".docx", ".md"]
        file_extension = "." + filename.split(".")[-1].lower() if "." in filename else ""

        if file_extension not in supported_extensions:
            status_reports.append(IngestionStatus(
                filename=filename,
                status="Skipped",
                message=f"Unsupported file type: {file_extension}. Supported: {', '.join(supported_extensions)}"
            ))
            await file.close()
            continue

        current_report: Optional[IngestionStatus] = None
        try:
            file_content: bytes = await file.read()
            logger.info(f"File '{filename}' (from batch) read, size: {len(file_content)} bytes.")

            current_report = await process_document_for_ingestion(filename, file_content)
            status_reports.append(current_report)

        except Exception as e:
            logger.error(f"Error processing file '{filename}' in batch: {e}", exc_info=True)
            # Ensure a status report is added even on unexpected error during file read/process call
            if current_report is None: # Error happened before process_document_for_ingestion returned
                status_reports.append(IngestionStatus(
                    filename=filename,
                    status="Failed",
                    message=f"Unexpected error: {str(e)}"
                ))
        finally:
            await file.close()

    return status_reports


# Example for checking status if using background tasks (not implemented for sync version)
# @router.get("/upload_status/{filename}", response_model=IngestionStatus)
# async def get_ingestion_status(filename: str):
#     status = task_status_db.get(filename)
#     if not status:
#         raise HTTPException(status_code=404, detail="Ingestion task not found or not started for this filename.")
#     return status


if __name__ == "__main__":
    # This router is meant to be run by Uvicorn as part of a FastAPI app.
    # To test this individually, you'd typically run the main FastAPI app.
    # However, you can do some ad-hoc testing with an HTTP client like 'requests'
    # if you temporarily spin up this router in a minimal FastAPI app.

    # Example of how to run this router for quick manual testing:
    # from fastapi import FastAPI
    # temp_app = FastAPI()
    # temp_app.include_router(router)
    # import uvicorn
    # print("Starting temporary FastAPI server for ingestion router on http://127.0.0.1:8001")
    # print("Try POSTing a file to http://127.0.0.1:8001/ingest/upload_file/")
    # uvicorn.run(temp_app, host="127.0.0.1", port=8001)

    # To test properly, run the main 'app/main.py' once it's created and use an HTTP client
    # like Insomnia, Postman, or curl, or the Streamlit UI once it's built.
    # Example curl command (ensure FastAPI server is running):
    # curl -X POST -F "file=@/path/to/your/sample_document.md" http://127.0.0.1:8000/api/v1/ingest/upload_file/
    # (Adjust URL based on your main app's prefix and port)
    print("Ingestion router defined. Run with a FastAPI application (e.g., app/main.py) to test endpoints.")