import json
import logging
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI, OpenAIError # Use AsyncOpenAI for FastAPI
from app.core.config import settings # Provides API key, model names, and prompts

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Configure as needed

# Initialize the AsyncOpenAI client
# Ensure OPENAI_API_KEY is set in your .env file
if not settings.OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment settings. OpenAI connector will not function.")
    # Depending on desired behavior, you might raise an exception here or allow lazy init.
    # For now, we'll let it proceed but calls will fail.
    async_client = None
else:
    async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, max_retries=2) # Configure retries as needed

async def _call_openai_api(model_name: str, messages: List[Dict[str, str]], is_json_mode: bool = False) -> Optional[str]:
    """
    Helper function to make calls to the OpenAI Chat Completions API.
    Handles common API call logic and error handling.
    """
    if not async_client:
        logger.error("OpenAI async client not initialized. Cannot make API call.")
        return None
    try:
        logger.debug(f"Calling OpenAI API. Model: {model_name}, JSON Mode: {is_json_mode}, Messages: {messages}")
        completion_params = {
            "model": model_name,
            "messages": messages,
        }
        if is_json_mode:
            completion_params["response_format"] = {"type": "json_object"}

        response = await async_client.chat.completions.create(**completion_params)

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            content = response.choices[0].message.content.strip()
            logger.debug(f"OpenAI API response content: {content[:200]}...") # Log snippet
            return content
        else:
            logger.warning("OpenAI API call succeeded but returned no content or unexpected structure.")
            return None
    except OpenAIError as e:
        logger.error(f"OpenAI API error for model {model_name}: {e}")
        # Consider re-raising specific errors or returning None based on strategy
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during OpenAI API call for model {model_name}: {e}")
        return None

async def extract_entities_relationships_from_chunk(text_chunk: str) -> Optional[Dict[str, Any]]:
    """
    Uses LLM for extracting entities and relationships from a text chunk.
    """
    schema_details = settings.SCHEMA.get_schema_details()
    system_message = settings.PROMPTS.get_system_message("json_expert", "Output valid JSON.")
    user_prompt_template = settings.PROMPTS.get_user_prompt("extract_entities_relationships")

    if not user_prompt_template:
        logger.error("extract_entities_relationships prompt not found.")
        return None

    schema_hint = (
        f"Schema Hint:\n"
        f"Preferred Entity Types: {schema_details.get('entity_types', [])}\n"
        f"Preferred Relationship Types: {schema_details.get('relationship_types', [])}\n"
        f"Allow dynamic entity types: {schema_details.get('allow_dynamic_entity_types', True)}\n"
        f"Allow dynamic relationship types: {schema_details.get('allow_dynamic_relationship_types', True)}"
    )

    formatted_user_prompt = user_prompt_template.format(
        text_chunk=text_chunk,
        schema_hint=schema_hint
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": formatted_user_prompt}
    ]

    response_content = await _call_openai_api(
        model_name=settings.LLM_INGESTION_MODEL_NAME,
        messages=messages,
        is_json_mode=True
    )

    if response_content:
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM for entity extraction: {e}\nResponse: {response_content}")
            return None
    return None

async def extract_entities_from_query(user_query: str) -> Optional[List[str]]:
    """
    Uses LLM to extract key entity names from a user query. Returns a list of entity names.
    """
    system_message = settings.PROMPTS.get_system_message("json_expert", "Output valid JSON containing a list of strings under 'entities' key.")
    user_prompt_template = settings.PROMPTS.get_user_prompt("extract_entities_from_query")

    if not user_prompt_template:
        logger.error("extract_entities_from_query prompt not found.")
        return None

    formatted_user_prompt = user_prompt_template.format(user_query=user_query)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": formatted_user_prompt}
    ]

    response_content = await _call_openai_api(
        model_name=settings.LLM_QUERY_NER_MODEL_NAME,
        messages=messages,
        is_json_mode=True
    )

    if response_content:
        try:
            data = json.loads(response_content)
            # Expects format like {"entities": ["Entity1", "Entity2"]}
            entities = data.get("entities", [])
            if isinstance(entities, list) and all(isinstance(e, str) for e in entities):
                return entities
            else:
                logger.warning(f"LLM query NER output format unexpected: {data}. Expected list of strings under 'entities'.")
                return [] # Return empty list on format mismatch
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM for query NER: {e}\nResponse: {response_content}")
            return None # Indicate error rather than empty list for JSON errors
    return None


async def classify_query_intent(user_query: str, linked_entities: List[str]) -> Optional[Dict[str, Any]]:
    """
    Uses LLM to classify query intent and identify target entities.
    """
    system_message = settings.PROMPTS.get_system_message("query_understanding_expert", "Output valid JSON.")
    user_prompt_template = settings.PROMPTS.get_user_prompt("classify_query_intent")

    if not user_prompt_template:
        logger.error("classify_query_intent prompt not found.")
        return None

    formatted_user_prompt = user_prompt_template.format(user_query=user_query, linked_entities=linked_entities)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": formatted_user_prompt}
    ]

    response_content = await _call_openai_api(
        model_name=settings.LLM_QUERY_INTENT_MODEL_NAME,
        messages=messages,
        is_json_mode=True
    )

    if response_content:
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM for intent classification: {e}\nResponse: {response_content}")
            return None
    return None

async def generate_response_from_subgraph(user_query: str, subgraph_text: str) -> Optional[str]:
    """
    Uses LLM to generate a response based on the query and subgraph context.
    """
    system_message = settings.PROMPTS.get_system_message("graph_rag_assistant", "You are a helpful AI.")
    user_prompt_template = settings.PROMPTS.get_user_prompt("generate_response_from_subgraph")

    if not user_prompt_template:
        logger.error("generate_response_from_subgraph prompt not found.")
        return None

    formatted_user_prompt = user_prompt_template.format(user_query=user_query, subgraph_text=subgraph_text)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": formatted_user_prompt}
    ]

    return await _call_openai_api(
        model_name=settings.LLM_QUERY_RESPONSE_MODEL_NAME,
        messages=messages,
        is_json_mode=False # This response is natural language text
    )

async def get_text_embedding(text: str, model_name: Optional[str] = None) -> Optional[List[float]]:
    """
    Generates embeddings for a given text.
    """
    if not async_client:
        logger.error("OpenAI async client not initialized. Cannot get text embedding.")
        return None

    active_model_name = model_name or settings.LLM_EMBEDDING_MODEL_NAME
    try:
        text_to_embed = text.replace("\n", " ") # OpenAI recommends replacing newlines
        if not text_to_embed.strip(): # Handle empty or whitespace-only strings
            logger.warning("Attempted to get embedding for empty or whitespace-only text. Returning None.")
            return None

        logger.debug(f"Requesting embedding for text (first 50 chars): '{text_to_embed[:50]}...' with model {active_model_name}")
        response = await async_client.embeddings.create(
            input=[text_to_embed], # API expects a list of strings
            model=active_model_name
        )
        if response.data and response.data[0].embedding:
            return response.data[0].embedding
        else:
            logger.warning(f"OpenAI embedding call for model {active_model_name} returned no embedding data.")
            return None
    except OpenAIError as e:
        logger.error(f"OpenAI API error during embedding generation with model {active_model_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during embedding generation with model {active_model_name}: {e}")
        return None


if __name__ == "__main__":
    import asyncio

    async def main_test():
        # Ensure your .env, config.yaml, prompts.yaml, schema.yaml are set up correctly
        # and OPENAI_API_KEY is valid.

        print(f"Using Ingestion Model: {settings.LLM_INGESTION_MODEL_NAME}")
        print(f"Using Embedding Model: {settings.LLM_EMBEDDING_MODEL_NAME}")

        # Test 1: Entity and Relationship Extraction
        print("\n--- Test: Entity and Relationship Extraction ---")
        sample_text_chunk = (
            "Dr. Evelyn Reed, the CTO of Alpha Corp since 2023, "
            "is leading the cloud platform initiative. Alpha Corp uses Azure and AWS."
            "Project Nova is a key focus."
        )
        extracted_data = await extract_entities_relationships_from_chunk(sample_text_chunk)
        if extracted_data:
            print("Extraction Output:")
            print(json.dumps(extracted_data, indent=2))
        else:
            print("Extraction failed or returned no data.")

        # Test 2: Query Entity Extraction
        print("\n--- Test: Query Entity Extraction ---")
        sample_query = "Tell me about Alpha Corp's Project Nova and its CTO Evelyn Reed."
        query_entities = await extract_entities_from_query(sample_query)
        if query_entities is not None: # Check for None to distinguish from empty list
            print(f"Entities from query '{sample_query}': {query_entities}")
        else:
            print("Query entity extraction failed.")


        # Test 3: Query Intent Classification
        print("\n--- Test: Query Intent Classification ---")
        linked_graph_entities = ["Alpha Corp", "Project Nova", "Evelyn Reed"]
        intent_data = await classify_query_intent(sample_query, linked_graph_entities)
        if intent_data:
            print("Intent Classification Output:")
            print(json.dumps(intent_data, indent=2))
        else:
            print("Intent classification failed.")

        # Test 4: Text Embedding
        print("\n--- Test: Text Embedding ---")
        embedding_text = "Alpha Corporation focuses on renewable energy."
        embedding_vector = await get_text_embedding(embedding_text)
        if embedding_vector:
            print(f"Embedding for '{embedding_text}': First 5 dims: {embedding_vector[:5]}... (Length: {len(embedding_vector)})")
            # Check against expected dimension
            if len(embedding_vector) != settings.EMBEDDING_DIMENSION:
                print(f"WARNING: Embedding dimension mismatch! Got {len(embedding_vector)}, expected {settings.EMBEDDING_DIMENSION}")
        else:
            print(f"Failed to get embedding for '{embedding_text}'.")

        # Test 5: Response Generation (requires dummy subgraph context)
        print("\n--- Test: Response Generation ---")
        dummy_subgraph_context = (
            "Node: Alpha Corp (Type: ORGANIZATION, Description: A tech company). "
            "Node: Project Nova (Type: PROJECT, Description: An innovative project by Alpha Corp). "
            "Node: Evelyn Reed (Type: PERSON, Role: CTO of Alpha Corp). "
            "Relationship: Evelyn Reed -[WORKS_FOR]-> Alpha Corp. "
            "Relationship: Alpha Corp -[DEVELOPS]-> Project Nova."
        )
        final_answer = await generate_response_from_subgraph(sample_query, dummy_subgraph_context)
        if final_answer:
            print(f"Generated Answer for query '{sample_query}':\n{final_answer}")
        else:
            print("Response generation failed.")

        # Test embedding with empty string
        print("\n--- Test: Text Embedding with empty string ---")
        empty_embedding_vector = await get_text_embedding("   ") # Test with whitespace only
        if empty_embedding_vector:
            print(f"Embedding for empty string: {empty_embedding_vector}")
        else:
            print("Correctly failed to get embedding for empty string or returned None.")


    if settings.OPENAI_API_KEY and async_client: # Only run tests if API key is present
        asyncio.run(main_test())
    else:
        print("Skipping OpenAI connector tests as OPENAI_API_KEY is not set or client failed to initialize.")