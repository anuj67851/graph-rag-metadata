import os
import yaml
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv()

class PromptsConfig:
    """Loads and provides access to LLM prompts from a YAML file."""
    def __init__(self, filepath: str = "prompts.yaml"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prompts configuration file not found at '{filepath}'")
        with open(filepath, 'r') as f:
            self._config = yaml.safe_load(f)

        self.SYSTEM_MESSAGES: Dict[str, str] = self._config.get("system_messages", {})
        self.USER_PROMPTS: Dict[str, str] = self._config.get("user_prompts", {})

    def get_system_message(self, key: str, default: Optional[str] = "") -> str:
        return self.SYSTEM_MESSAGES.get(key, default)

    def get_user_prompt(self, key: str, default: Optional[str] = "") -> str:
        return self.USER_PROMPTS.get(key, default)

class SchemaConfig:
    """Loads and provides access to knowledge graph schema hints from a YAML file."""
    def __init__(self, filepath: str = "schema.yaml"):
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
        self.NEO4J_URI: Optional[str] = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.NEO4J_USERNAME: Optional[str] = os.getenv("NEO4J_USERNAME", "neo4j")
        self.NEO4J_PASSWORD: Optional[str] = os.getenv("NEO4J_PASSWORD", "password")

        # --- YAML Configuration ---
        self.APP_NAME: str = app_config.get("app_name", "Graph RAG Application")
        self.API_V1_STR: str = app_config.get("api_v1_str", "/api/v1")

        # Core LLM Models
        self.LLM_INGESTION_MODEL_NAME: str = app_config.get("llm_ingestion_model_name", "gpt-4o-mini")
        self.LLM_QUERY_RESPONSE_MODEL_NAME: str = app_config.get("llm_query_response_model_name", "gpt-4o")
        self.LLM_EMBEDDING_MODEL_NAME: str = app_config.get("llm_embedding_model_name", "text-embedding-3-small")
        self.EMBEDDING_DIMENSION: int

        # File Paths
        self.SCHEMA_FILE_PATH: str = app_config.get("schema_file_path", "schema.yaml")
        self.PROMPTS_FILE_PATH: str = app_config.get("prompts_file_path", "prompts.yaml")
        self.FAISS_INDEX_PATH: str = app_config.get("faiss_index_path", "data/vector_store/graph_rag.index")

        # Query & Retrieval Parameters
        self.ENTITY_INFO_HOP_DEPTH: int = int(app_config.get("entity_info_hop_depth", 1))
        self.SEMANTIC_SEARCH_TOP_K: int = int(app_config.get("semantic_search_top_k", 3))

        # Graph Exploration Endpoint Defaults
        self.DEFAULT_FULL_GRAPH_NODE_LIMIT: int = int(app_config.get("default_full_graph_node_limit", 100))
        self.DEFAULT_FULL_GRAPH_EDGE_LIMIT: int = int(app_config.get("default_full_graph_edge_limit", 150))

        # Derive EMBEDDING_DIMENSION from model name
        model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        self.EMBEDDING_DIMENSION = model_dims.get(self.LLM_EMBEDDING_MODEL_NAME)
        if self.EMBEDDING_DIMENSION is None:
            # Fallback for unknown models
            default_dim = 1536
            self.EMBEDDING_DIMENSION = int(app_config.get("embedding_dimension", default_dim))
            print(f"Warning: Unknown embedding dimension for model '{self.LLM_EMBEDDING_MODEL_NAME}'. Defaulting to {self.EMBEDDING_DIMENSION}.")

        # Loaded Configurations
        self.PROMPTS = PromptsConfig(self.PROMPTS_FILE_PATH)
        self.SCHEMA = SchemaConfig(self.SCHEMA_FILE_PATH)

        if not self.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY is not set in the environment. LLM calls will fail.")

settings = Settings()