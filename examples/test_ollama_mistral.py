import asyncio
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./olm_mistral_dickens"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="mistral-nemo:latest",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

# with open("./olm_mistral_dickens/book.txt", "r", encoding="utf-8") as f:
#     rag.insert(f.read())

# # Perform naive search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
# )

# # Perform local search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
# )

# # Perform global search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
# )

# #Perform hybrid search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
# )

# # stream response
# resp = rag.query(
#     "What are the top themes in this story?",
#     param=QueryParam(mode="hybrid", stream=True),
# )


# async def print_stream(stream):
#     async for chunk in stream:
#         print(chunk, end="", flush=True)


# if inspect.isasyncgen(resp):
#     asyncio.run(print_stream(resp))
# else:
#     print(resp)


# Perform naive search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="naive")))

# Perform local search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))

# Perform global search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

# Perform hybrid search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid")))

# Perform mix search (Knowledge Graph + Vector Retrieval)
# Mix mode combines knowledge graph and vector search:
# - Uses both structured (KG) and unstructured (vector) information
# - Provides comprehensive answers by analyzing relationships and context
# - Supports image content through HTML img tags
# - Allows control over retrieval depth via top_k parameter
print(rag.query("What are the top themes in this story?", param=QueryParam(
    mode="mix")))
