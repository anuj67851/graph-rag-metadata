import os
import yaml
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
load_dotenv()

class PromptsConfig:
    """Loads and provides access to LLM prompts from a YAML file."""
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prompts configuration file not found at '{filepath}'")
        with open(filepath, 'r') as f:
            self._config = yaml.safe_load(f)
        self.SYSTEM_MESSAGES: Dict[str, str] = self._config.get("system_messages", {})
        self.USER_PROMPTS: Dict[str, str] = self._config.get("user_prompts", {})
    def get_system_message(self, key: str, default: str = "") -> str: return self.SYSTEM_MESSAGES.get(key, default)
    def get_user_prompt(self, key: str, default: str = "") -> str: return self.USER_PROMPTS.get(key, default)

class SchemaConfig:
    """Loads and provides access to knowledge graph schema hints from a YAML file."""
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Schema configuration file not found at '{filepath}'")
        with open(filepath, 'r') as f:
            self._config = yaml.safe_load(f)
        self.ENTITY_TYPES: List[str] = self._config.get("entity_types", [])
        self.RELATIONSHIP_TYPES: List[str] = self._config.get("relationship_types", [])
        self.ALLOW_DYNAMIC_ENTITY_TYPES: bool = self._config.get("allow_dynamic_entity_types", True)
        self.ALLOW_DYNAMIC_RELATIONSHIP_TYPES: bool = self._config.get("allow_dynamic_relationship_types", True)

class Settings:
    """Aggregates all application settings from environment variables and config files."""
    def __init__(self, config_file_path: str = "config.yaml"):
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Main configuration file not found at '{config_file_path}'")
        with open(config_file_path, 'r') as f:
            app_config = yaml.safe_load(f)

        # --- Environment Variables ---
        self.OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

        # --- Database & Service Connections (from .env) ---
        self.NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
        self.NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
        self.WEAVIATE_HOST: str = os.getenv("WEAVIATE_HOST", "localhost")
        self.WEAVIATE_PORT: str = os.getenv("WEAVIATE_PORT", "8080")
        self.REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))

        # --- YAML Configuration ---
        self.APP_NAME: str = app_config.get("app_name", "Graph RAG Application")
        self.API_V1_STR: str = app_config.get("api_v1_str", "/api/v1")

        # --- Core LLM Models ---
        self.LLM_INGESTION_MODEL_NAME: str = app_config.get("llm_ingestion_model_name", "gpt-4o")
        self.LLM_QUERY_RESPONSE_MODEL_NAME: str = app_config.get("llm_query_response_model_name", "gpt-4o-mini")

        # --- Embedding Model ---
        self.EMBEDDING_MODEL_REPO: str = app_config.get("embedding_model_repo", "sentence-transformers/multi-qa-mpnet-base-cos-v1")
        self.EMBEDDING_DIMENSION: int = app_config.get("embedding_dimension", 768)

        # --- File Paths and Storage ---
        self.SCHEMA_FILE_PATH: str = app_config.get("schema_file_path", "schema.yaml")
        self.PROMPTS_FILE_PATH: str = app_config.get("prompts_file_path", "prompts.yaml")
        self.FILE_STORAGE_PATH: str = app_config.get("file_storage_path", "data/uploaded_files")
        self.SQLITE_DB_PATH: str = app_config.get("sqlite_db_path", "data/file_metadata.db")
        self.LOG_FILE_PATH: str = app_config.get("log_file_path", "logs/graph_rag_app.log")
        self.LOG_RETENTION_DAYS: int = app_config.get("log_retention_days", 7)

        # --- Weaviate Configuration ---
        self.WEAVIATE_CLASS_NAME: str = app_config.get("weaviate_class_name", "TextChunk")

        # --- Retrieval Pipeline Configuration ---
        self.RETRIEVAL_PIPELINE: Dict[str, Any] = app_config.get("retrieval_pipeline", {})

        # --- Graph Context & Exploration ---
        self.ENTITY_INFO_HOP_DEPTH: int = int(app_config.get("entity_info_hop_depth", 1))
        self.DEFAULT_FULL_GRAPH_NODE_LIMIT: int = int(app_config.get("default_full_graph_node_limit", 100))
        self.DEFAULT_FULL_GRAPH_EDGE_LIMIT: int = int(app_config.get("default_full_graph_edge_limit", 150))

        # Loaded Configurations from other files
        self.PROMPTS = PromptsConfig(self.PROMPTS_FILE_PATH)
        self.SCHEMA = SchemaConfig(self.SCHEMA_FILE_PATH)

        if not self.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY is not set in the environment. LLM calls will fail.")

# Create a single, globally accessible settings object
settings = Settings()