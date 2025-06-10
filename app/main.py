import logging
import os
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.apis import router_ingestion, router_query, router_graph

# Import lifecycle event handlers from all connectors
from app.graph_db.neo4j_connector import init_neo4j_driver, close_neo4j_driver
from app.vector_store.weaviate_connector import init_vector_store, save_vector_store_on_shutdown
from app.database.sqlite_connector import get_sqlite_connector

# --- Advanced Logging Configuration ---
# Create logs directory if it doesn't exist
log_dir = os.path.dirname(settings.LOG_FILE_PATH)
os.makedirs(log_dir, exist_ok=True)

# Define the log format
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO) # Set the minimum level for the root logger

# Configure Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# Configure Timed Rotating File Handler
# This will create a new log file at midnight and keep the last 7 files
file_handler = TimedRotatingFileHandler(
    filename=settings.LOG_FILE_PATH,
    when='midnight',
    backupCount=settings.LOG_RETENTION_DAYS,
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# Get a logger for this specific module
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.APP_NAME,
    description="API for an advanced Retrieval Augmented Generation system using a Knowledge Graph.",
    version="1.0.0",
)

# --- CORS Middleware ---
# Allows the Streamlit UI (on a different port) to communicate with this backend.
origins = [
    "http://localhost",
    "http://localhost:8501", # Default Streamlit port
    "http://127.0.0.1",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Application Lifecycle Events (Startup & Shutdown) ---
@app.on_event("startup")
async def startup_event():
    """
    Actions to perform when the application starts up.
    - Initialize Neo4j driver and ensure constraints.
    - Initialize Weaviate vector store and ensure schema.
    - Initialize SQLite database and ensure schema.
    """
    logger.info("Application startup sequence initiated...")
    try:
        # Initialize SQLite and its schema
        sqlite_conn = get_sqlite_connector()
        sqlite_conn.initialize_schema()
        logger.info("SQLite connector initialized successfully.")
    except Exception as e:
        logger.error(f"Error during SQLite initialization on startup: {e}", exc_info=True)
        # We might want to exit if a critical DB fails

    try:
        await init_neo4j_driver()
        logger.info("Neo4j driver initialization process completed.")
    except Exception as e:
        logger.error(f"Error during Neo4j driver initialization on startup: {e}", exc_info=True)

    try:
        await init_vector_store()
        logger.info("Weaviate vector store initialization process completed.")
    except Exception as e:
        logger.error(f"Error during Weaviate vector store initialization on startup: {e}", exc_info=True)

    logger.info(f"'{settings.APP_NAME}' application started successfully.")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Actions to perform when the application shuts down.
    - Close Neo4j driver.
    - Close SQLite connection.
    - Weaviate client does not need an explicit close.
    """
    logger.info("Application shutdown sequence initiated...")
    try:
        await close_neo4j_driver()
        logger.info("Neo4j driver closed.")
    except Exception as e:
        logger.error(f"Error closing Neo4j driver on shutdown: {e}", exc_info=True)

    try:
        await save_vector_store_on_shutdown() # Logs a message for Weaviate
        logger.info("Weaviate vector store shutdown process completed.")
    except Exception as e:
        logger.error(f"Error during Weaviate shutdown: {e}", exc_info=True)

    try:
        sqlite_conn = get_sqlite_connector()
        sqlite_conn.close_connection()
        logger.info("SQLite connection closed.")
    except Exception as e:
        logger.error(f"Error closing SQLite connection on shutdown: {e}", exc_info=True)

    logger.info(f"'{settings.APP_NAME}' application shut down gracefully.")

# --- Include API Routers ---
# Using the common API prefix from our settings file.
common_api_prefix = settings.API_V1_STR

app.include_router(router_ingestion.router, prefix=common_api_prefix)
app.include_router(router_query.router, prefix=common_api_prefix)
app.include_router(router_graph.router, prefix=common_api_prefix)

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {
        "message": f"Welcome to the {settings.APP_NAME}. API documentation is available at /docs or /redoc."
    }