import logging
import os
from contextlib import asynccontextmanager
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
log_dir = os.path.dirname(settings.LOG_FILE_PATH)
os.makedirs(log_dir, exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)
file_handler = TimedRotatingFileHandler(
    filename=settings.LOG_FILE_PATH,
    when='midnight',
    backupCount=settings.LOG_RETENTION_DAYS,
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)
logger = logging.getLogger(__name__)


# --- THE FIX: Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    """
    # --- Startup Logic ---
    logger.info("Application startup sequence initiated...")
    try:
        sqlite_conn = get_sqlite_connector()
        sqlite_conn.initialize_schema()
        logger.info("SQLite connector initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Error during SQLite initialization: {e}", exc_info=True)
        # In a real app, you might want a more graceful way to handle a failed DB connection
        # For now, we log it as critical.

    try:
        await init_neo4j_driver()
        logger.info("Neo4j driver initialization process completed.")
    except Exception as e:
        logger.critical(f"FATAL: Error during Neo4j driver initialization: {e}", exc_info=True)

    try:
        await init_vector_store()
        logger.info("Weaviate vector store initialization process completed.")
    except Exception as e:
        logger.critical(f"FATAL: Error during Weaviate vector store initialization: {e}", exc_info=True)

    logger.info(f"'{settings.APP_NAME}' application has started.")

    yield # The application runs here

    # --- Shutdown Logic ---
    logger.info("Application shutdown sequence initiated...")
    try:
        await close_neo4j_driver()
        logger.info("Neo4j driver closed.")
    except Exception as e:
        logger.error(f"Error closing Neo4j driver on shutdown: {e}", exc_info=True)

    try:
        await save_vector_store_on_shutdown()
        logger.info("Weaviate vector store shutdown process completed.")
    except Exception as e:
        logger.error(f"Error during Weaviate shutdown: {e}", exc_info=True)

    try:
        sqlite_conn = get_sqlite_connector()
        sqlite_conn.close_connection()
        logger.info("SQLite connection closed.")
    except Exception as e:
        logger.error(f"Error closing SQLite connection on shutdown: {e}", exc_info=True)

    logger.info(f"'{settings.APP_NAME}' application has shut down gracefully.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.APP_NAME,
    description="API for an advanced Retrieval Augmented Generation system using a Knowledge Graph.",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:8501",
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

# --- Include API Routers ---
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