"""LLM API client utilities."""

from __future__ import annotations

import asyncio
import logging
from typing import List, Dict, Optional
from enum import Enum

from openai import (
    AsyncOpenAI, 
    APIConnectionError, 
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    APIStatusError,
)

logger = logging.getLogger(__name__)


class APIErrorType(Enum):
    """API 错误类型分类。"""
    RATE_LIMIT = "rate_limit"      # 429 - 可重试
    CONNECTION = "connection"       # 网络问题 - 可重试
    AUTH = "auth"                   # 401 - 不可重试
    BAD_REQUEST = "bad_request"     # 400 - 不可重试
    SERVER = "server"               # 500+ - 可重试
    UNKNOWN = "unknown"


def classify_error(error: Exception) -> tuple[APIErrorType, bool]:
    """
    分类 API 错误并判断是否可重试。
    
    Returns:
        (错误类型, 是否可重试)
    """
    if isinstance(error, RateLimitError):
        return APIErrorType.RATE_LIMIT, True
    elif isinstance(error, APIConnectionError):
        return APIErrorType.CONNECTION, True
    elif isinstance(error, AuthenticationError):
        return APIErrorType.AUTH, False
    elif isinstance(error, BadRequestError):
        return APIErrorType.BAD_REQUEST, False
    elif isinstance(error, APIStatusError):
        # 5xx 错误可重试
        if hasattr(error, 'status_code') and error.status_code >= 500:
            return APIErrorType.SERVER, True
        return APIErrorType.UNKNOWN, False
    else:
        return APIErrorType.UNKNOWN, False


async def call_llm_async(
    client: AsyncOpenAI, 
    model: str, 
    messages: List[Dict[str, str]], 
    temperature: float = 0.3, 
    max_retries: int = 3,
    json_mode: bool = False,
    timeout: float = 60.0
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
        timeout: Request timeout in seconds
        
    Returns:
        Response content as string, empty string on failure
    """
    last_error: Optional[Exception] = None
    
    for attempt in range(max_retries):
        try:
            params: Dict = {
                "model": model, 
                "messages": messages, 
                "temperature": temperature,
            }
            if json_mode:
                params["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**params)
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
        except Exception as e:
            last_error = e
            error_type, retryable = classify_error(e)
            
            if not retryable:
                logger.error(f"Non-retryable error ({error_type.value}): {e}")
                break
            
            # 计算退避时间
            if error_type == APIErrorType.RATE_LIMIT:
                # Rate limit 使用更长的退避时间
                delay = min(2 ** (attempt + 2), 60)  # 4, 8, 16... max 60
            else:
                delay = 2 ** (attempt + 1)  # 2, 4, 8
            
            logger.warning(
                f"Retryable error ({error_type.value}): {e}. "
                f"Retry {attempt + 1}/{max_retries} in {delay}s..."
            )
            await asyncio.sleep(delay)
    
    if last_error:
        logger.error(f"All {max_retries} retries failed. Last error: {last_error}")
    
    return ""


async def call_llm_with_fallback(
    client: AsyncOpenAI,
    models: List[str],
    messages: List[Dict[str, str]],
    **kwargs
) -> tuple[str, str]:
    """
    Try multiple models in sequence until one succeeds.
    
    Args:
        client: AsyncOpenAI client
        models: List of model names to try
        messages: Messages to send
        **kwargs: Additional arguments for call_llm_async
        
    Returns:
        Tuple of (response, model_used)
    """
    for model in models:
        result = await call_llm_async(client, model, messages, **kwargs)
        if result:
            return result, model
    
    return "", ""


def create_client(
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    timeout: float = 60.0
) -> AsyncOpenAI:
    """
    Create an AsyncOpenAI client.
    
    Args:
        api_key: API key for authentication
        base_url: API base URL
        timeout: Default timeout for requests
        
    Returns:
        Configured AsyncOpenAI client
    """
    return AsyncOpenAI(
        api_key=api_key, 
        base_url=base_url,
        timeout=timeout,
    )
