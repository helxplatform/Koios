from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client import AsyncQdrantClient, models
import numpy as np,os,json
import csv,glob
ollama_emb = OllamaEmbeddings(model="llama3", base_url="http://localhost:11434")
client = QdrantClient(url="http://localhost:54713/")

async def search(user_query):
    question_embedding = ollama_emb.embed_query(user_query)
    print(len(question_embedding))
    results = await client.search(
        collection_name="test_collection",
        query_vector=question_embedding,
        limit=20,
    )
    return results


user_query = "Heart and Vascular ?"
#question_embedding = ollama_emb.embed_query(user_query)
query_vector = ollama_emb.embed_query(user_query)
hits = client.search(
    collection_name="test_collection3",
    query_vector=query_vector,
    limit=10  # Return 5 closest points
)
for i in hits:
    print ("\n",i)
#print(hits)