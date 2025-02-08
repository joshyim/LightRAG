"""
Gemini API implementation for LightRAG
"""
import google.generativeai as genai
from typing import Optional, Dict, Any, List, Union
import asyncio
from google.api_core import exceptions
import logging

logger = logging.getLogger(__name__)

async def retry_with_exponential_backoff(func, *args, max_retries=5, initial_delay=1, **kwargs):
    """Retry a function with exponential backoff when rate limited."""
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except exceptions.ResourceExhausted as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise
            
            wait_time = delay * (2 ** attempt)  # Exponential backoff
            logger.warning(f"Rate limited by Gemini API. Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
    
    raise last_exception


async def gemini_complete(
    messages: Union[str, List[Dict[str, str]]],
    model: str = "gemini-2.0-flash-001",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """
    Complete text using Google's Gemini API
    Args:
        messages: Either a string prompt or a list of message dictionaries
        model: The model to use (default: gemini-2.0-flash-001)
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional arguments for generation config
    Returns:
        str: The generated response
    """
    # Configure the model
    model_instance = genai.GenerativeModel(model)
    
    # Handle different input types
    if isinstance(messages, str):
        content = messages
    else:
        # Convert message list to Gemini format
        content = [{
            'role': msg['role'],
            'parts': [msg['content']]
        } for msg in messages]
    
    # Handle LightRAG's hashing_kv
    kwargs.pop('hashing_kv', None)
    
    # Filter out LightRAG-specific parameters
    gemini_params = {}
    valid_params = {'candidate_count', 'stop_sequences', 'top_p', 'top_k'}
    ignored_params = {'history_messages', 'history_turns', 'only_need_context', 'only_need_prompt', 'response_type', 'stream'}
    for key, value in kwargs.items():
        if key in valid_params:
            gemini_params[key] = value
        elif key not in ignored_params:
            print(f"Warning: Unknown parameter {key} passed to gemini_complete")
    
    # Generate response with retry logic
    async def _generate():
        response = model_instance.generate_content(
            content,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens if max_tokens else None,
                **gemini_params
            )
        )
        return response
    
    response = await retry_with_exponential_backoff(_generate)
    return response.text


async def gemini_embed(
    text: str,
    model: str = "models/text-embedding-004",
    **kwargs: Any,
) -> List[float]:
    """
    Get embeddings using Google's Text Embedding API
    Args:
        text: The text to embed
        model: The model to use for embedding (default: text-embedding-004)
        **kwargs: Additional arguments to pass to the API
    Returns:
        List[float]: The embedding vector
    """
    # Handle LightRAG's hashing_kv
    kwargs.pop('hashing_kv', None)
    
    # Filter out LightRAG-specific parameters
    gemini_params = {}
    valid_params = {'task_type', 'title', 'output_dimensionality'}
    ignored_params = {'history_messages', 'history_turns', 'only_need_context', 'only_need_prompt', 'response_type', 'stream'}
    for key, value in kwargs.items():
        if key in valid_params:
            gemini_params[key] = value
        elif key not in ignored_params:
            print(f"Warning: Unknown parameter {key} passed to gemini_embed")
    
    # Call the embedding API with retry logic
    async def _embed():
        response = genai.embed_content(
            model=model,
            content=text,
            task_type="RETRIEVAL_DOCUMENT",  # Use uppercase for task type
            **gemini_params
        )
        return response
    
    response = await retry_with_exponential_backoff(_embed)
    
    # The response should be a dictionary containing the embedding values
    if not isinstance(response, dict):
        raise ValueError(f"Expected dict response, got {type(response)}")
        
    # Get the embedding values
    if 'embedding' not in response:
        raise ValueError(f"No 'embedding' key in response: {response}")
        
    embedding = response['embedding']
    if not embedding or not isinstance(embedding, list):
        raise ValueError(f"Invalid embedding format: {embedding}")
        
    return embedding
