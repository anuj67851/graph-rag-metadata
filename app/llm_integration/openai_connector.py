import json
import logging
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI, OpenAIError
from app.core.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if not settings.OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment settings. OpenAI connector will not function.")
    async_client = None
else:
    async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, max_retries=2)

async def _call_openai_api(model_name: str, messages: List[Dict[str, str]], is_json_mode: bool = False) -> Optional[str]:
    """Helper function to make calls to the OpenAI Chat Completions API."""
    if not async_client:
        logger.error("OpenAI async client not initialized. Cannot make API call.")
        return None
    try:
        logger.debug(f"Calling OpenAI API. Model: {model_name}, JSON Mode: {is_json_mode}")
        completion_params = {
            "model": model_name,
            "messages": messages,
        }
        if is_json_mode:
            completion_params["response_format"] = {"type": "json_object"}

        response = await async_client.chat.completions.create(**completion_params)

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            logger.warning("OpenAI API call succeeded but returned no content.")
            return None
    except OpenAIError as e:
        logger.error(f"OpenAI API error for model {model_name}: {e}")
        # Pass more specific error info up if needed
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during OpenAI API call for model {model_name}: {e}")
        raise e

async def extract_entities_relationships_from_chunk(text_chunk: str) -> Optional[Dict[str, Any]]:
    """Uses LLM for extracting entities and relationships from a text chunk (for ingestion)."""
    # Note: I've removed the manual schema_details building from here.
    # The prompt itself is now responsible for its content.
    system_message = settings.PROMPTS.get_system_message("json_expert", "Output valid JSON.")
    user_prompt_template = settings.PROMPTS.get_user_prompt("extract_entities_relationships")

    if not user_prompt_template:
        logger.error("extract_entities_relationships prompt not found.")
        return None

    # Let's create a more detailed schema hint from the config
    schema_hint = (
        f"Schema Hint:\n"
        f"Preferred Entity Types: {settings.SCHEMA.ENTITY_TYPES}\n"
        f"Preferred Relationship Types: {settings.SCHEMA.RELATIONSHIP_TYPES}\n"
        f"Allow dynamic entity types: {settings.SCHEMA.ALLOW_DYNAMIC_ENTITY_TYPES}\n"
        f"Allow dynamic relationship types: {settings.SCHEMA.ALLOW_DYNAMIC_RELATIONSHIP_TYPES}"
    )

    formatted_user_prompt = user_prompt_template.format(
        text_chunk=text_chunk,
        schema_hint=schema_hint
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": formatted_user_prompt}
    ]

    try:
        response_content = await _call_openai_api(
            model_name=settings.LLM_INGESTION_MODEL_NAME,
            messages=messages,
            is_json_mode=True
        )
        if response_content:
            return json.loads(response_content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from LLM for entity extraction: {e}")
    except Exception as e:
        logger.error(f"An error occurred during entity extraction LLM call: {e}")

    return None


async def generate_response_from_context(user_query: str, combined_context: str) -> Optional[str]:
    """Uses LLM to generate a response based on the query and a combined text/graph context."""
    system_message = settings.PROMPTS.get_system_message("graph_rag_assistant")
    user_prompt_template = settings.PROMPTS.get_user_prompt("generate_response_from_context")

    if not user_prompt_template:
        logger.error("generate_response_from_context prompt not found.")
        return None

    formatted_user_prompt = user_prompt_template.format(user_query=user_query, combined_context=combined_context)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": formatted_user_prompt}
    ]

    try:
        return await _call_openai_api(
            model_name=settings.LLM_QUERY_RESPONSE_MODEL_NAME,
            messages=messages,
            is_json_mode=False
        )
    except Exception as e:
        logger.error(f"An error occurred during final response generation LLM call: {e}")
        return None


async def get_text_embedding(text: str, model_name: Optional[str] = None) -> Optional[List[float]]:
    """Generates embeddings for a given text."""
    if not async_client:
        logger.error("OpenAI async client not initialized. Cannot get text embedding.")
        return None

    active_model_name = model_name or settings.LLM_EMBEDDING_MODEL_NAME
    try:
        text_to_embed = text.replace("\n", " ").strip()
        if not text_to_embed:
            logger.warning("Attempted to get embedding for empty text. Returning None.")
            return None

        response = await async_client.embeddings.create(
            input=[text_to_embed],
            model=active_model_name
        )
        if response.data and response.data[0].embedding:
            return response.data[0].embedding
        else:
            logger.warning(f"OpenAI embedding call returned no embedding data for model {active_model_name}.")
            return None
    except Exception as e:
        logger.error(f"An unexpected error during embedding generation with model {active_model_name}: {e}")
        # Don't re-raise here, just return None as per function signature
        return None