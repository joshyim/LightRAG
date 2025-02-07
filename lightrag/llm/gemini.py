"""
Gemini API implementation for LightRAG
"""
import google.generativeai as genai
from typing import Optional, Dict, Any, List


async def gemini_complete(
    messages: List[Dict[str, str]],
    model: str = "gemini-2.0-flash-001",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """
    Complete text using Google's Gemini API
    """
    # Convert messages to Gemini format
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"{content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"

    # Configure the model
    model = genai.GenerativeModel(model)
    
    # Generate response
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens if max_tokens else None,
            **kwargs
        )
    )
    
    return response.text


async def gemini_embed(
    text: str,
    model: str = "models/embedding-001",
    **kwargs: Any,
) -> List[float]:
    """
    Get embeddings using Google's Gemini API
    """
    model = genai.GenerativeModel(model)
    result = model.embed_content(text)
    return result.embedding
