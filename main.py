from embeddings import EmbeddingModel
from indexer import FaissIndexer
from search import SemanticSearch
from flask import render_template
from utils import load_text_file, chunk_text
import random
import time
import math


text = load_text_file("sample.txt")

documents = chunk_text(text,chunk_size=80, overlap=20)

print("Total chunks created:", len(documents))

# Step 1: Initialize embedding model
embedding_model = EmbeddingModel()

# Step 2: Get embedding dimension
dimension = embedding_model.dimension

# Step 3: Initialize FAISS index

num_docs = len(documents)
nlist = int(math.sqrt(num_docs))
indexer = FaissIndexer(dimension, nlist=nlist)

# Step 4: Initialize search engine
search_engine = SemanticSearch(embedding_model, indexer)

# Step 5: Add documents
start_time = time.time()
search_engine.add_documents(documents)
end_time = time.time()

print("Indexing time:", end_time - start_time, "seconds")
print("Total vectors indexed:", indexer.total_vectors())

# Step 6: Query
start_time = time.time()
results = search_engine.query("Tell me about pets", top_k=2)
end_time = time.time()

# print(f"Query time: {end_time - start_time:.6f} seconds")

# print("\nTop Results:")
# for result in results:
#     print(result)

query = "What is quantum mechanics?"

context = search_engine.query_with_context(query, top_k=3)

print("\nRetrieved Context:\n")
print(context)