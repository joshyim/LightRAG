import asyncio
import os
import logging
import inspect
import time
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv
from lightrag.kg.postgres_impl import PostgreSQLDB

# neo4j
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "y*gqZqq76f_bNzkWfqKpMbYv"

#postgres
postgres_db = PostgreSQLDB(
    config={
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "6FTQzK3XJk-kQEjEqQt-h4fA",
        "database": "rag",
    }
)

WORKING_DIR = "./dickens"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def main():
    await postgres_db.initdb()
    # Check if PostgreSQL DB tables exist, if not, tables will be created
    await postgres_db.check_tables()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="gemma2:9Bmod",
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
        graph_storage="Neo4JStorage",
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        vector_storage="PGVectorStorage",
    )

    rag.doc_status.db = postgres_db
    rag.full_docs.db = postgres_db
    rag.text_chunks.db = postgres_db
    rag.llm_response_cache.db = postgres_db
    rag.key_string_value_json_storage_cls.db = postgres_db
    rag.chunks_vdb.db = postgres_db
    rag.relationships_vdb.db = postgres_db
    rag.entities_vdb.db = postgres_db


    with open("./dickens/book.txt", "r", encoding="utf-8") as f:
        await rag.ainsert(f.read())

    # # Query time
    # print("==== Trying to test the rag queries ====")
    # start_time = time.time()

    # # Perform hybrid search
    # print("**** Start Hybrid Query ****")
    # print(
    #     await rag.aquery(
    #         "who is scrooge?", param=QueryParam(mode="naive", stream=True)
    #     )
    # )
    # print(f"Hybrid Query Time: {time.time() - start_time} seconds")

    # stream response
    # resp = await rag.aquery(
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


if __name__ == "__main__":
    asyncio.run(main())


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

# # Perform hybrid search
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
