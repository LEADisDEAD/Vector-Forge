from embeddings import EmbeddingModel
from indexer import FaissIndexer
from search import SemanticSearch
from utils import load_text_file,load_pdf_file, chunk_text
import math
import os

# Load and index the documents


documents = []

for filename in os.listdir("uploads"):
    if filename.endswith(".pdf"):
        text = load_pdf_file(os.path.join("uploads", filename))
        chunks = chunk_text(text, chunk_size=150, overlap=40)
        documents.extend(chunks)

print("Total chunks:", len(documents))

print("Total chunks:", len(documents))

embedding_model = EmbeddingModel()
dimension = embedding_model.dimension
nlist = int(math.sqrt(len(documents)))
indexer = FaissIndexer(dimension, nlist=nlist)

search_engine = SemanticSearch(embedding_model, indexer)
search_engine.add_documents(documents)


# Example Evaluation Queries

evaluation_queries = [
    {"query": "How does general relativity describe gravity?", "expected_phrase": "spacetime"},
    {"query": "What are black holes according to general relativity?", "expected_phrase": "black hole"},
    {"query": "What is a qubit in quantum computing?", "expected_phrase": "qubit"},
    {"query": "How does electromagnetism describe electric forces?", "expected_phrase": "electric charge"},
    {"query": "What are Maxwell's equations?", "expected_phrase": "Maxwell"},
]

k = 3

precision_total = 0
mrr_total = 0

for item in evaluation_queries:
    query = item["query"]
    expected = item["expected_phrase"].lower()

    results = search_engine.query(query, top_k=k)

    hit = False
    reciprocal_rank = 0

    for rank, result in enumerate(results, start=1):
        if expected in result["text"].lower():
            hit = True
            reciprocal_rank = 1 / rank
            break

    precision_total += 1 if hit else 0
    mrr_total += reciprocal_rank

    print(f"\nQuery: {query}")
    print(f"Hit: {hit}, Reciprocal Rank: {reciprocal_rank}")

precision_at_k = precision_total / len(evaluation_queries)
mrr = mrr_total / len(evaluation_queries)

print("\n--- FINAL METRICS ---")
print(f"Precision@{k}: {precision_at_k:.2f}")
print(f"MRR: {mrr:.2f}")