import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings # For app name, API prefix, etc.
from app.apis import router_ingestion, router_query, router_graph

# Import lifecycle event handlers from connectors
from app.graph_db.neo4j_connector import init_neo4j_driver, close_neo4j_driver
from app.vector_store.faiss_connector import init_vector_store, save_vector_store_on_shutdown
# (OpenAI client is typically initialized on import or lazily, no explicit init/shutdown needed here unless specific resource management)

# Configure logging for the main application
# You might want to use a more sophisticated logging setup for production (e.g., structured logging, external log management)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
# The title and description will appear in the auto-generated API docs (e.g., /docs)
app = FastAPI(
    title=settings.APP_NAME or "Graph RAG API",
    description="API for a Retrieval Augmented Generation system using a Knowledge Graph.",
    version="0.1.0",
    # openapi_url=f"{settings.API_V1_STR}/openapi.json" # If using an API version prefix for docs
)

# --- CORS Middleware ---
# This is important to allow your Streamlit UI (running on a different port)
# to make requests to this FastAPI backend.
# For production, you should restrict origins to the specific domain of your Streamlit app.
origins = [
    "http://localhost",        # Common base for local dev
    "http://localhost:8501",   # Default Streamlit port
    "http://127.0.0.1",
    "http://127.0.0.1:8501",
    # Add any other origins if needed (e.g., your deployed Streamlit app's URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Allows specific origins
    # allow_origins=["*"],       # Allows all origins (less secure, use for broad testing only)
    allow_credentials=True,      # Allows cookies to be included in requests
    allow_methods=["*"],         # Allows all standard HTTP methods
    allow_headers=["*"],         # Allows all headers
)


# --- Application Lifecycle Events (Startup & Shutdown) ---
@app.on_event("startup")
async def startup_event():
    """
    Actions to perform when the application starts up.
    - Initialize Neo4j driver and ensure constraints.
    - Initialize/load FAISS vector store.
    """
    logger.info("Application startup sequence initiated...")
    try:
        await init_neo4j_driver()
        logger.info("Neo4j driver initialization process completed.")
    except Exception as e:
        logger.error(f"Error during Neo4j driver initialization on startup: {e}", exc_info=True)
        # Depending on severity, you might want to prevent app startup or run in a degraded mode.

    try:
        await init_vector_store()
        logger.info("FAISS vector store initialization process completed.")
    except Exception as e:
        logger.error(f"Error during FAISS vector store initialization on startup: {e}", exc_info=True)

    logger.info(f"'{settings.APP_NAME}' application started successfully.")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Actions to perform when the application shuts down.
    - Close Neo4j driver.
    - Save FAISS vector store.
    """
    logger.info("Application shutdown sequence initiated...")
    try:
        await close_neo4j_driver()
        logger.info("Neo4j driver closed.")
    except Exception as e:
        logger.error(f"Error closing Neo4j driver on shutdown: {e}", exc_info=True)

    try:
        await save_vector_store_on_shutdown() # Ensures FAISS index is saved
        logger.info("FAISS vector store save process on shutdown completed.")
    except Exception as e:
        logger.error(f"Error saving FAISS vector store on shutdown: {e}", exc_info=True)

    logger.info(f"'{settings.APP_NAME}' application shut down gracefully.")


# --- Include API Routers ---
# Routers defined in app/apis/ are included here.
# You can use a common prefix for all API endpoints if desired, e.g., settings.API_V1_STR.
# For simplicity, the prefixes are currently defined within each router file.
# If you want a global /api/v1 prefix, you would add it here when including the router.
# e.g., app.include_router(router_ingestion.router, prefix=settings.API_V1_STR)
# and then remove the prefix from the router files themselves, or make them relative.

common_api_prefix = settings.API_V1_STR if hasattr(settings, 'API_V1_STR') and settings.API_V1_STR else "/api/v1"

app.include_router(router_ingestion.router, prefix=f"{common_api_prefix}{router_ingestion.router.prefix}")
app.include_router(router_query.router, prefix=f"{common_api_prefix}{router_query.router.prefix}")
app.include_router(router_graph.router, prefix=f"{common_api_prefix}{router_graph.router.prefix}")

# --- Root Endpoint (Optional) ---
@app.get("/", tags=["Root"])
async def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {
        "message": f"Welcome to the {settings.APP_NAME or 'Graph RAG API'}. API documentation is available at /docs or /redoc."
    }

# To run this application (from the 'graph_rag_app' root directory):
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
#
# Then you can access:
# - API root: http://localhost:8000/
# - Interactive API docs (Swagger UI): http://localhost:8000/docs
# - Alternative API docs (ReDoc): http://localhost:8000/redoc
# - Ingestion endpoint example: http://localhost:8000/api/v1/ingest/upload_file/ (POST)
# - Query endpoint example: http://localhost:8000/api/v1/query/ (POST)
# - Graph endpoint example: http://localhost:8000/api/v1/graph/full_sample (GET)