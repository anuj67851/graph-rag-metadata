# 🧠 Graph RAG: Advanced Knowledge Management System

This project is a sophisticated, production-ready Retrieval Augmented Generation (RAG) system designed for advanced knowledge management. It leverages a hybrid approach, combining the power of a **Vector Database** for semantic search and a **Graph Database** for understanding complex relationships within unstructured documents.

The system is built with a modular, highly configurable, and scalable architecture using FastAPI for the backend and Streamlit for the user interface. It is not just a simple RAG pipeline; it's a complete platform for building, managing, querying, and exploring a knowledge base derived from your documents.

[![API Docs Swagger](https://img.shields.io/badge/API-Documentation-blue?style=for-the-badge&logo=fastapi)](http://localhost:8000/docs)
[![Neo4j Browser](https://img.shields.io/badge/Neo4j-Browser-008cc1?style=for-the-badge&logo=neo4j)](http://localhost:7474)

---

## ✨ Core Features

*   **Hybrid Retrieval Pipeline**:
    *   **Context-Aware Query Expansion**: Uses an LLM to rewrite user queries based on initially retrieved context, overcoming ambiguity and improving search intent.
    *   **Vector Search**: Utilizes **Weaviate** to perform fast, semantic search over document chunks.
    *   **Graph Context Augmentation**: Enriches the context by fetching related entities and relationships from a **Neo4j** knowledge graph.
    *   **Cross-Encoder Re-ranking**: Employs a powerful re-ranker model to score and re-order search results for maximum relevance before passing them to the final LLM.

*   **Intelligent Ingestion**:
    *   **Multi-Format Support**: Ingests `.pdf`, `.docx`, `.txt`, and `.md` files.
    *   **Semantic Chunking**: Breaks documents down into contextually coherent chunks using language models, rather than fixed-size splits.
    *   **Automated Knowledge Graph Creation**: Extracts entities (e.g., people, organizations) and their relationships from text and populates a Neo4j graph automatically.

*   **Robust & Scalable Backend**:
    *   **FastAPI**: A modern, high-performance web framework for the API, with automatic interactive documentation.
    *   **Dockerized Services**: The entire application stack (API, Weaviate, Neo4j, Redis) is containerized with Docker for easy setup and deployment.
    *   **Background Tasks**: Ingestion is handled as a background task for a responsive, non-blocking user experience.
    *   **Performance Caching**: **Redis** is used to cache query results, reducing latency and LLM costs.

*   **Comprehensive User & Developer Interface**:
    *   **Multi-Page Streamlit App**: A clean, organized UI for all system interactions.
    *   **File Management**: Upload, monitor status, download, and delete documents from the knowledge base.
    *   **Interactive Chat**: A familiar chat interface for querying documents, with support for filtering by source file.
    *   **Knowledge Graph Explorer**: A visual interface powered by Pyvis to explore the full graph, find the busiest nodes, or inspect the neighborhood of any specific entity.
    *   **Dynamic API Explorer**: An interactive page within the UI to test *any* backend endpoint directly, with dynamically generated forms for parameters and request bodies.

*   **Highly Configurable & Maintainable**:
    *   **YAML & `.env` Configuration**: Easily configure everything from database connections and LLM model names to retrieval pipeline parameters.
    *   **Modular Architecture**: Each component (database connectors, caching, services) is designed as a replaceable module with clear separation of concerns, making the system easy to extend and maintain.

---

## 🛠️ Tech Stack

*   **Backend**: FastAPI, Python 3.11
*   **Frontend**: Streamlit
*   **Vector Database**: Weaviate
*   **Graph Database**: Neo4j
*   **Cache**: Redis
*   **Containerization**: Docker, Docker Compose
*   **NLP/LLM Orchestration**: LangChain, OpenAI
*   **Core LLMs**: OpenAI models (configurable in `config.yaml`)
*   **Embedding Model**: Sentence-Transformers (configurable in `config.yaml` and `.env`)

---

## 🚀 Getting Started

### Prerequisites

*   **Docker** and **Docker Compose**: Make sure they are installed on your system. [Install Docker](https://docs.docker.com/get-docker/)
*   **Git**: For cloning the repository.
*   **An OpenAI API Key**.

### 1. Clone the Repository

```bash
git clone https://github.com/anuj67851/graph-rag-metadata.git
cd graph-rag-metadata
```

### 2. Configure Environment Variables

Create a file named `.env` in the root of the project directory by copying the example file.

```bash
cp .env.example .env
```

Now, open the `.env` file and fill in your details:

```dotenv
# .env

# Your OpenAI API Key
OPENAI_API_KEY="sk-..."

# Neo4j Credentials (choose a strong password)
NEO4J_URI="bolt://neo4j:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your_very_strong_password"

# Weaviate Connection (uses service name from docker-compose)
WEAVIATE_HOST="weaviate"
WEAVIATE_PORT="8080"

# Redis Connection (uses service name from docker-compose)
REDIS_HOST="redis"
REDIS_PORT="6379"

# Model used by the Weaviate inference container (must match config.yaml)
EMBEDDING_MODEL_REPO="sentence-transformers/multi-qa-mpnet-base-cos-v1"
```

### 3. Run the Entire System

The simplest way to run the entire backend stack (API, Weaviate, Neo4j, Redis) is with a single Docker Compose command.

Open a terminal in the project root and run:
```bash
docker-compose up --build
```
This will:
1.  Pull all necessary Docker images (Weaviate, Neo4j, Redis, etc.).
2.  Build the `rag-api` container, installing all Python dependencies from `requirements.txt`.
3.  Start all services and connect them on an internal Docker network.

You will see logs from all services in your terminal. Wait until you see messages indicating that all services have started successfully.

### 4. Run the Streamlit Frontend

For the best development experience, run the Streamlit UI in a separate terminal.

First, ensure you have the Python dependencies installed locally for Streamlit to use:
```bash
pip install -r requirements.txt
```

Then, run the Streamlit app:
```bash
streamlit run streamlit_ui/app.py
```

Your web browser should open a new tab to **`http://localhost:8501`**.

### 5. Accessing Services

Once everything is running, you can access the various parts of the system:

*   **Streamlit UI**: `http://localhost:8501`
*   **FastAPI Backend**: `http://localhost:8000`
*   **Interactive API Docs (Swagger)**: `http://localhost:8000/docs`
*   **Alternative API Docs (ReDoc)**: `http://localhost:8000/redoc`
*   **Neo4j Browser**: `http://localhost:7474` (Use the username/password from your `.env` file to log in).

---

## 📖 API Endpoints Overview

The backend exposes a rich set of endpoints for interacting with the system. You can test these live using the **API Explorer** page in the Streamlit UI or at `http://localhost:8000/docs`.

### Ingestion & File Management (`/api/v1/ingest`)

*   `POST /upload_files/`: Upload one or more documents for background ingestion.
*   `GET /documents/`: List all managed documents and their metadata.
*   `GET /documents/{filename}/status`: Get the live ingestion status for a single file.
*   `GET /documents/{filename}/download`: Download the original copy of an ingested file.
*   `POST /documents/download/batch`: Download multiple documents as a single ZIP archive.
*   `POST /documents/{filename}/reprocess`: Re-process an existing file (e.g., after updating prompts).
*   `DELETE /documents/{filename}`: **(Destructive)** Delete a file and all its associated data from all systems.

### Querying (`/api/v1/query`)

*   `POST /`: The main endpoint to ask a question. Processes the query through the full hybrid RAG pipeline.
*   `POST /vector_search`: A raw vector search endpoint. Bypasses the RAG pipeline to return the top `k` similar text chunks directly from Weaviate. Useful for debugging.

### Graph Exploration (`/api/v1/graph`)

*   `GET /full_sample`: Get a random sample of the entire knowledge graph.
*   `GET /busiest_nodes`: Find the most highly-connected entities in the graph.
*   `GET /node/{node_id}`: Explore the graph starting from a specific entity.
*   `GET /schema`: Dynamically discover all entity and relationship types currently in the graph.

---

## ⚙️ Development Workflow

### Resetting the Databases

During development, you will often want to clear all data and start fresh. A helper script is provided for this.

**Important**: This script will permanently delete all data in your Weaviate, Neo4j, and Redis volumes.

First, make the script executable (you only need to do this once):
*   On Linux/macOS or Git Bash on Windows:
    ```bash
    chmod +x reset_and_rebuild.sh
    ```

To perform a full reset, run the script:
```bash
./reset_and_rebuild.sh
```
This script will:
1.  Stop the containers.
2.  Delete the database volumes.
3.  Rebuild the API container (to pick up any new dependencies).
4.  Start all services again.

### Stopping the Application

To stop all running services, press `Ctrl + C` in the `docker-compose` terminal, or run:
```bash
docker-compose down
```
To stop and remove the data volumes (a clean shutdown), run:
```bash
docker-compose down -v
```

---

## 🔧 Configuration & Architecture

The application is highly configurable through two main files:

*   **`.env`**: For secrets and connection details for external services.
*   **`config.yaml`**: For application logic, model names, retrieval pipeline parameters, and file paths. You can easily switch LLMs, embedding models, or tune the RAG pipeline by editing this file.

The backend follows a clean, three-layer architecture to ensure separation of concerns:

1.  **API Layer (`/apis`)**: Contains the FastAPI routers. This layer is responsible for handling HTTP requests and responses, performing validation, and calling the service layer. It contains no business logic.
2.  **Service Layer (`/services`)**: The core business logic of the application resides here. It orchestrates operations, such as calling multiple connectors to fulfill a request (e.g., deleting data from Weaviate and Neo4j).
3.  **Data Access Layer (`/database`, `/graph_db`, `/vector_store`)**: Contains the connectors. Each connector is responsible for all communication with a specific database. It encapsulates all platform-specific code (e.g., Cypher queries for Neo4j, `hweaviate` client calls).