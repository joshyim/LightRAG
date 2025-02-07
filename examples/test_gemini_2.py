import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_complete, gemini_embed
import google.generativeai as genai
from lightrag.utils import EmbeddingFunc
#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./gemini_dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Configure Gemini API
genai.configure(api_key='YOUR_GEMINI_API_KEY')  # Replace with your API key

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=lambda *args, **kwargs: gemini_complete(*args, model="gemini-2.0-flash-001", **kwargs),  # Use Gemini 2.0 Flash model
    embedding_func=EmbeddingFunc(
        embedding_dim=768,  # Gemini's embedding dimension
        max_token_size=8192,
        func=gemini_embed,
    )
)

with open("./oai_dickens/book.txt") as f:
    rag.insert(f.read())

# Perform naive search
# print(rag.query("What are the top themes in this story?", param=QueryParam(mode="naive")))

# # Perform local search
# print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))

# # Perform global search
# print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

# # Perform hybrid search
# print(rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid")))

# Perform mix search (Knowledge Graph + Vector Retrieval)
# Mix mode combines knowledge graph and vector search:
# - Uses both structured (KG) and unstructured (vector) information
# - Provides comprehensive answers by analyzing relationships and context
# - Supports image content through HTML img tags
# - Allows control over retrieval depth via top_k parameter
# print(rag.query("What are the top themes in this story?", param=QueryParam(
#     mode="mix")))