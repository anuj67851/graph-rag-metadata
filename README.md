# 🧠 Graph RAG: Advanced Knowledge Management System

This project is a sophisticated, production-ready Retrieval Augmented Generation (RAG) system designed for advanced knowledge management. It leverages a hybrid approach, combining the power of a Vector Database for semantic search and a Graph Database for understanding complex relationships within unstructured documents.

The system is built with a modular, highly configurable, and scalable architecture using FastAPI for the backend and Streamlit for the user interface.

---

## ✨ Core Features

*   **Hybrid Retrieval Pipeline**:
    *   **Context-Aware Query Expansion**: Uses an LLM to rewrite user queries based on initially retrieved context, overcoming ambiguity and improving search intent.
    *   **Vector Search**: Utilizes **Weaviate** to perform fast, semantic search over document chunks.
    *   **Graph Context Augmentation**: Enriches the context by fetching related entities and relationships from a **Neo4j** knowledge graph.
    *   **Cross-Encoder Re-ranking**: Employs a powerful re-ranker model to score and re-order search results for maximum relevance before passing them to the final LLM.
*   **Intelligent Ingestion**:
    *   **Semantic Chunking**: Breaks documents down into contextually coherent chunks using language models.
    *   **Automated Knowledge Graph Creation**: Extracts entities and relationships from text and populates a Neo4j graph automatically.
*   **Robust & Scalable Backend**:
    *   **FastAPI**: A modern, high-performance web framework for the API.
    *   **Dockerized Services**: The entire application stack (API, Weaviate, Neo4j, Redis) is containerized with Docker for easy setup and deployment.
    *   **Background Tasks**: Ingestion is handled as a background task for a responsive, non-blocking user experience.
    *   **Performance Caching**: **Redis** is used to cache query results, reducing latency and LLM costs.
*   **Intuitive User Interface**:
    *   **Multi-Page Streamlit App**: A clean, organized UI for all system interactions.
    *   **File Management**: Upload, monitor status, and delete documents from the knowledge base.
    *   **Interactive Chat**: A familiar chat interface for querying documents, with support for filtering by source file.
    *   **Knowledge Graph Explorer**: A visual interface powered by Pyvis to explore the extracted knowledge graph.
*   **Highly Configurable**:
    *   **YAML & `.env` Configuration**: Easily configure everything from database connections and LLM model names to retrieval pipeline parameters.
    *   **Modular Design**: Each component (database connectors, caching, retrieval) is designed as a replaceable module.

---

## 🛠️ Tech Stack

*   **Backend**: FastAPI, Python 3.11
*   **Frontend**: Streamlit
*   **Vector Database**: Weaviate
*   **Graph Database**: Neo4j
*   **Cache**: Redis
*   **Containerization**: Docker, Docker Compose
*   **NLP/LLM Orchestration**: LangChain
*   **Primary LLMs**: OpenAI (configurable)

---

## 🚀 Getting Started

### Prerequisites

*   **Docker** and **Docker Compose**: Make sure they are installed on your system. [Install Docker](https://docs.docker.com/get-docker/)
*   **Git**: For cloning the repository.
*   **Python 3.11+**: For running the Streamlit frontend.
*   **An OpenAI API Key**.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
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
1.  Pull all necessary Docker images (Weaviate, Neo4j, Redis).
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

Your web browser should open a new tab to **`http://localhost:8501`**. You can now start using the application!

*   The FastAPI backend is accessible at `http://localhost:8000`.
*   The interactive API documentation (Swagger UI) is at `http://localhost:8000/docs`.
*   The Neo4j Browser is at `http://localhost:7474`.

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
To stop and remove the data volumes, run:
```bash
docker-compose down -v
```

---

## 🔧 Configuration

The application is highly configurable through two main files:

*   **`.env`**: For secrets and connection details for external services.
*   **`config.yaml`**: For application logic, model names, retrieval pipeline parameters, and file paths. You can easily switch LLMs, embedding models, or tune the RAG pipeline by editing this file.