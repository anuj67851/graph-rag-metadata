import os
import yaml
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load .env file variables into environment variables
# Call this at the module level so environment variables are set when Settings is initialized.
load_dotenv()

class PromptsConfig:
    """
    Loads and provides access to LLM prompts from a YAML file.
    """
    def __init__(self, filepath: str = "prompts.yaml"):
        if not os.path.exists(filepath):
            # Create a dummy file if it doesn't exist to prevent startup errors
            # In a real scenario, this file should be part of the deployment
            print(f"Warning: Prompts file '{filepath}' not found. Creating a dummy file.")
            dummy_prompts = {
                "system_messages": {"default": "You are a helpful assistant."},
                "user_prompts": {"default_task": "Perform the default task with this input: {input}"}
            }
            with open(filepath, 'w') as f:
                yaml.dump(dummy_prompts, f)
            self._config = dummy_prompts
        else:
            with open(filepath, 'r') as f:
                self._config = yaml.safe_load(f)

        self.SYSTEM_MESSAGES: Dict[str, str] = self._config.get("system_messages", {})
        self.USER_PROMPTS: Dict[str, str] = self._config.get("user_prompts", {})

    def get_system_message(self, key: str, default: Optional[str] = "") -> str:
        return self.SYSTEM_MESSAGES.get(key, default)

    def get_user_prompt(self, key: str, default: Optional[str] = "") -> str:
        return self.USER_PROMPTS.get(key, default)

class SchemaConfig:
    """
    Loads and provides access to knowledge graph schema hints from a YAML file.
    """
    def __init__(self, filepath: str = "schema.yaml"):
        if not os.path.exists(filepath):
            print(f"Warning: Schema file '{filepath}' not found. Creating a dummy file.")
            dummy_schema = {
                "entity_types": ["PERSON", "ORGANIZATION"],
                "relationship_types": ["WORKS_FOR"],
                "allow_dynamic_entity_types": True,
                "allow_dynamic_relationship_types": True
            }
            with open(filepath, 'w') as f:
                yaml.dump(dummy_schema, f)
            self._config = dummy_schema
        else:
            with open(filepath, 'r') as f:
                self._config = yaml.safe_load(f)

        self.ENTITY_TYPES: List[str] = self._config.get("entity_types", [])
        self.RELATIONSHIP_TYPES: List[str] = self._config.get("relationship_types", [])
        self.ALLOW_DYNAMIC_ENTITY_TYPES: bool = self._config.get("allow_dynamic_entity_types", True)
        self.ALLOW_DYNAMIC_RELATIONSHIP_TYPES: bool = self._config.get("allow_dynamic_relationship_types", True)

    def get_schema_details(self) -> Dict[str, Any]:
        return {
            "entity_types": self.ENTITY_TYPES,
            "relationship_types": self.RELATIONSHIP_TYPES,
            "allow_dynamic_entity_types": self.ALLOW_DYNAMIC_ENTITY_TYPES,
            "allow_dynamic_relationship_types": self.ALLOW_DYNAMIC_RELATIONSHIP_TYPES,
        }

class Settings:
    """
    Aggregates all application settings from environment variables and YAML files.
    """
    # --- Environment Variables ---
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    NEO4J_URI: Optional[str] = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: Optional[str] = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: Optional[str] = os.getenv("NEO4J_PASSWORD", "password")

    # --- YAML Configuration (config.yaml) ---
    APP_NAME: str = "Graph RAG Application"
    API_V1_STR: str = "/api/v1"

    LLM_INGESTION_MODEL_NAME: str
    LLM_QUERY_NER_MODEL_NAME: str
    LLM_QUERY_INTENT_MODEL_NAME: str
    LLM_QUERY_RESPONSE_MODEL_NAME: str
    LLM_EMBEDDING_MODEL_NAME: str
    EMBEDDING_DIMENSION: int

    SCHEMA_FILE_PATH: str
    PROMPTS_FILE_PATH: str

    FAISS_INDEX_PATH: str
    VECTOR_MATCH_THRESHOLD: float
    ENTITY_INFO_HOP_DEPTH: int
    RELATIONSHIP_DISCOVERY_MAX_PATH_LENGTH: int
    COMPLEX_QUERY_HOP_DEPTH: int

    # --- Loaded Configurations from dedicated files ---
    PROMPTS: PromptsConfig
    SCHEMA: SchemaConfig


    def __init__(self, config_file_path: str = "config.yaml"):
        if not os.path.exists(config_file_path):
            # Create a dummy config.yaml if it doesn't exist
            print(f"Warning: Main config file '{config_file_path}' not found. Creating a dummy file.")
            dummy_config = {
                "app_name": self.APP_NAME,
                "api_v1_str": self.API_V1_STR,
                "llm_ingestion_model_name": "gpt-4o-mini",
                "llm_query_ner_model_name": "gpt-4o-mini",
                "llm_query_intent_model_name": "gpt-4o-mini",
                "llm_query_response_model_name": "gpt-4o",
                "llm_embedding_model_name": "text-embedding-3-small",
                "schema_file_path": "schema.yaml",
                "prompts_file_path": "prompts.yaml",
                "faiss_index_path": "data/vector_store/graph_entities.index",
                "vector_match_threshold": 0.7,
                "entity_info_hop_depth": 1,
                "relationship_discovery_max_path_length": 3,
                "complex_query_hop_depth": 2
            }
            with open(config_file_path, 'w') as f:
                yaml.dump(dummy_config, f)
            app_config = dummy_config
        else:
            with open(config_file_path, 'r') as f:
                app_config = yaml.safe_load(f)

        # Overwrite defaults with values from config.yaml if present
        self.APP_NAME = app_config.get("app_name", self.APP_NAME)
        self.API_V1_STR = app_config.get("api_v1_str", self.API_V1_STR)

        self.LLM_INGESTION_MODEL_NAME = app_config.get("llm_ingestion_model_name", "gpt-4o-mini")
        self.LLM_QUERY_NER_MODEL_NAME = app_config.get("llm_query_ner_model_name", "gpt-4o-mini")
        self.LLM_QUERY_INTENT_MODEL_NAME = app_config.get("llm_query_intent_model_name", "gpt-4o-mini")
        self.LLM_QUERY_RESPONSE_MODEL_NAME = app_config.get("llm_query_response_model_name", "gpt-4o")
        self.LLM_EMBEDDING_MODEL_NAME = app_config.get("llm_embedding_model_name", "text-embedding-3-small")

        self.SCHEMA_FILE_PATH = app_config.get("schema_file_path", "schema.yaml")
        self.PROMPTS_FILE_PATH = app_config.get("prompts_file_path", "prompts.yaml")

        self.FAISS_INDEX_PATH = app_config.get("faiss_index_path", "data/vector_store/graph_entities.index")
        self.VECTOR_MATCH_THRESHOLD = float(app_config.get("vector_match_threshold", 0.7))
        self.ENTITY_INFO_HOP_DEPTH = int(app_config.get("entity_info_hop_depth", 1))
        self.RELATIONSHIP_DISCOVERY_MAX_PATH_LENGTH = int(app_config.get("relationship_discovery_max_path_length", 3))
        self.COMPLEX_QUERY_HOP_DEPTH = int(app_config.get("complex_query_hop_depth", 2))

        # Derive EMBEDDING_DIMENSION based on the chosen model
        # This is a simplified lookup; a more robust solution might involve querying OpenAI API
        # or having a more comprehensive mapping.
        if self.LLM_EMBEDDING_MODEL_NAME == "text-embedding-3-small":
            self.EMBEDDING_DIMENSION = 1536
        elif self.LLM_EMBEDDING_MODEL_NAME == "text-embedding-3-large":
            self.EMBEDDING_DIMENSION = 3072
        elif self.LLM_EMBEDDING_MODEL_NAME == "text-embedding-ada-002": # Older model
            self.EMBEDDING_DIMENSION = 1536
        else:
            # Fallback or raise error if dimension is unknown and not specified
            default_dim = 1536
            print(
                f"Warning: Embedding dimension for model '{self.LLM_EMBEDDING_MODEL_NAME}' is not explicitly known. "
                f"Defaulting to {default_dim}. Please verify or set 'embedding_dimension' in config.yaml."
            )
            self.EMBEDDING_DIMENSION = int(app_config.get("embedding_dimension", default_dim))


        # Load prompts and schema configurations
        self.PROMPTS = PromptsConfig(self.PROMPTS_FILE_PATH)
        self.SCHEMA = SchemaConfig(self.SCHEMA_FILE_PATH)

        # Validate essential configurations
        if not self.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY is not set in the environment.")
        # Add other critical validations as needed


# Create a single global instance of the Settings class
# This instance will be imported by other modules.
settings = Settings()

if __name__ == "__main__":
    # Example usage and test print
    print("--- Settings Loaded ---")
    print(f"App Name: {settings.APP_NAME}")
    print(f"OpenAI API Key Loaded: {'Yes' if settings.OPENAI_API_KEY else 'No'}")
    print(f"Neo4j URI: {settings.NEO4J_URI}")
    print(f"Ingestion LLM: {settings.LLM_INGESTION_MODEL_NAME}")
    print(f"Embedding Model: {settings.LLM_EMBEDDING_MODEL_NAME}")
    print(f"Embedding Dimension: {settings.EMBEDDING_DIMENSION}")
    print(f"FAISS Index Path: {settings.FAISS_INDEX_PATH}")

    print("\n--- Schema Config ---")
    print(f"Entity Types: {settings.SCHEMA.ENTITY_TYPES}")
    print(f"Allow Dynamic Entities: {settings.SCHEMA.ALLOW_DYNAMIC_ENTITY_TYPES}")

    print("\n--- Prompts Config (Example) ---")
    print(f"System (json_expert): {settings.PROMPTS.get_system_message('json_expert', 'Not Found')}") # Assuming 'json_expert' exists in prompts.yaml
    print(f"User (extract_entities_relationships): {settings.PROMPTS.get_user_prompt('extract_entities_relationships', 'Not Found')[:50]}...") # Print first 50 chars