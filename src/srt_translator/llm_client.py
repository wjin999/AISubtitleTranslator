"""LLM API client utilities."""

from __future__ import annotations

import asyncio
import logging
from typing import List, Dict, Optional

from openai import AsyncOpenAI, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)


async def call_llm_async(
    client: AsyncOpenAI, 
    model: str, 
    messages: List[Dict[str, str]], 
    temperature: float = 0.3, 
    max_retries: int = 3,
    json_mode: bool = False
) -> str:
    """
    Make async call to LLM API with retry logic.
    
    Args:
        client: AsyncOpenAI client instance
        model: Model name to use
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_retries: Maximum retry attempts
        json_mode: Whether to request JSON response format
        
    Returns:
        Response content as string, empty string on failure
    """
    attempt = 0
    
    while attempt < max_retries:
        try:
            params: Dict = {
                "model": model, 
                "messages": messages, 
                "temperature": temperature
            }
            if json_mode:
                params["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**params)
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
        except RateLimitError as e:
            delay = 2 * (attempt + 1)
            logger.warning(
                f"Rate limit hit: {e}. Retrying in {delay}s... ({attempt+1}/{max_retries})"
            )
            await asyncio.sleep(delay)
            
        except APIConnectionError as e:
            delay = 2 * (attempt + 1)
            logger.warning(
                f"Connection error: {e}. Retrying in {delay}s... ({attempt+1}/{max_retries})"
            )
            await asyncio.sleep(delay)
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            await asyncio.sleep(2)
            
        attempt += 1
    
    return ""


def create_client(
    api_key: str,
    base_url: str = "https://api.deepseek.com"
) -> AsyncOpenAI:
    """
    Create an AsyncOpenAI client.
    
    Args:
        api_key: API key for authentication
        base_url: API base URL
        
    Returns:
        Configured AsyncOpenAI client
    """
    return AsyncOpenAI(api_key=api_key, base_url=base_url)
