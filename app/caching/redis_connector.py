import redis.asyncio as redis
import logging
import json
from typing import Optional

from app.core.config import settings
from app.models.query_models import QueryResponse
from redis import exceptions

logger = logging.getLogger(__name__)

class RedisConnector:
    """
    An ASYNCHRONOUS connector for managing a Redis cache.
    """
    _client: Optional[redis.Redis] = None
    _CACHE_EXPIRATION_SECONDS = 3600 # 1 hour

    async def _get_client(self) -> redis.Redis:
        """Establishes and returns the async Redis client."""
        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=0,
                    decode_responses=True
                )
                await self._client.ping()
                logger.info("Async Redis client connected successfully.")
            except exceptions.ConnectionError as e:
                logger.error(f"Failed to connect to async Redis: {e}", exc_info=True)
                self._client = None
        return self._client

    async def get_query_cache(self, cache_key: str) -> Optional[QueryResponse]:
        """Asynchronously retrieves a cached query response."""
        client = await self._get_client()
        if not client:
            return None

        try:
            cached_result = await client.get(cache_key)
            if cached_result:
                logger.info(f"Cache HIT for query key: {cache_key}")
                data = json.loads(cached_result)
                return QueryResponse(**data)
            return None
        except Exception as e:
            logger.error(f"Error getting async query cache for key '{cache_key}': {e}", exc_info=True)
            return None

    async def set_query_cache(self, cache_key: str, response: QueryResponse):
        """Asynchronously caches a query response."""
        client = await self._get_client()
        if not client:
            return

        try:
            json_response = response.model_dump_json()
            await client.set(cache_key, json_response, ex=self._CACHE_EXPIRATION_SECONDS)
            logger.info(f"Cached response for async query key: {cache_key}")
        except Exception as e:
            logger.error(f"Error setting async query cache for key '{cache_key}': {e}", exc_info=True)

# --- Singleton Management ---
_redis_connector_instance: Optional[RedisConnector] = None

def get_redis_connector() -> RedisConnector:
    """Provides a singleton instance of the RedisConnector."""
    global _redis_connector_instance
    if _redis_connector_instance is None:
        _redis_connector_instance = RedisConnector()
    return _redis_connector_instance