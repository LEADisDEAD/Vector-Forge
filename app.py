from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from embeddings import EmbeddingModel
from indexer import FaissIndexer
from search import SemanticSearch
from utils import load_text_file, chunk_text,load_pdf_file
import math
import time
import os


# -------------------------------
# Initialize Flask App
# -------------------------------
app = Flask(__name__)
app.secret_key = "vectorforge_secret_key"

@app.route("/", methods=["GET"])
def landing():
    return render_template("landing.html")

@app.route("/app", methods=["GET"])
def home():
    return render_template(
        "index.html",
        chat_history=session.get("chat_history", []),
        uploaded_files=search_engine.uploaded_files,
        total_chunks=len(search_engine.documents),
        total_files=len(search_engine.uploaded_files),
        search_engine=search_engine
    )

@app.route("/web_query", methods=["POST"])
def web_query():

    question = request.form.get("question")

    if not question:
        return redirect(url_for("home"))

    q = question.lower()

    # ---- Intent Detection ----
    summary_keywords = ["summarize", "summary", "overview", "brief"]
    is_summary = any(word in q for word in summary_keywords)

    is_document_level = (
        "document" in q or
        ("explain" in q and "this" in q) or
        ("describe" in q and "this" in q) or
        ("about" in q and "document" in q)
    )

    # ---- Chat Session Init ----
    if "chat_history" not in session:
        session["chat_history"] = []

    # 🚫 No Documents Guard
    if not search_engine.documents:
        session["chat_history"].append({
            "role": "user",
            "content": question
        })

        session["chat_history"].append({
            "role": "assistant",
            "content": "No documents uploaded. Please upload a file first.",
            "similarity": 0,
            "sources": [],
            "retrieval_time": 0,
            "llm_time": 0,
            "total_time": 0
        })

        session.modified = True
        return redirect(url_for("home"))

    total_start = time.perf_counter()

    # ---- 1️⃣ Retrieval ----
    retrieval_start = time.perf_counter()

    if is_summary or is_document_level:
        results = search_engine.query(question, top_k=10)
    else:
        results = search_engine.query(question, top_k=3)

    retrieval_end = time.perf_counter()

    # ---- 2️⃣ LLM ----
    llm_start = time.perf_counter()
    answer = search_engine.generate_answer_with_llm(question, results)
    llm_end = time.perf_counter()

    total_end = time.perf_counter()

    # ---- Metrics ----
    retrieval_time = round(retrieval_end - retrieval_start, 4)
    llm_time = round(llm_end - llm_start, 4)
    total_time = round(total_end - total_start, 4)

    top_similarity = results[0]["final_score"] if results else 0

    # ---- Hallucination Guardrail ----
    if not (is_summary or is_document_level) and top_similarity < 0.25:
        answer = "Answer not found in documents."

    # ---- Save Conversation ----
    session["chat_history"].append({
        "role": "user",
        "content": question
    })

    session["chat_history"].append({
        "role": "assistant",
        "content": answer,
        "similarity": top_similarity,
        "sources": [
            {
                "text": r["text"][:500],
                "source": r["source"],
                "citation_id": r["citation_id"]
            }
            for r in results
        ],
        "retrieval_time": retrieval_time,
        "llm_time": llm_time,
        "total_time": total_time
    })

    session.modified = True

    return redirect(url_for("home"))

# -------------------------------
# Build Empty Search Engine at Startup
# -------------------------------

print("Initializing empty search engine...")

embedding_model = EmbeddingModel()
dimension = embedding_model.dimension

# Start with empty index
nlist = 1
indexer = FaissIndexer(dimension, nlist=nlist)

search_engine = SemanticSearch(embedding_model, indexer)

print("System ready. No documents indexed.")

# -------------------------------
# Health Check Endpoint
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})

# -------------------------------
# Query Endpoint
# -------------------------------
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"]

    start_time = time.perf_counter()
    results = search_engine.query(question, top_k=3)
    end_time = time.perf_counter()

    latency = end_time - start_time
    
    return jsonify({
        "question": question,
        "num_chunks_indexed": len(search_engine.documents),
        "top_k": 3,
        "latency_seconds": latency,
        "results": results  
    })

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]

    if file.filename == "":
        return "Empty filename", 400

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    print(search_engine.uploaded_files)
    
    # Determine file type
    if file.filename.endswith(".txt"):
        text = load_text_file(filepath)
    elif file.filename.endswith(".pdf"):
        text = load_pdf_file(filepath)
    else:
        return "Unsupported file type", 400

    chunks = chunk_text(text, chunk_size=80, overlap=20)

    # Add to index dynamically
    search_engine.add_documents(chunks, source_name=file.filename)
    
    # clearing cache
    search_engine.answer_cache = {}
    
    return render_template(
        "index.html",
        results=None,
        latency=None,
        uploaded_files=search_engine.uploaded_files,
        total_chunks=len(search_engine.documents),
        total_files=len(search_engine.uploaded_files)
    )

@app.route("/clear", methods=["POST"])
def clear():
    search_engine.documents = []
    search_engine.doc_metadata = []
    search_engine.uploaded_files = {}

    # Reset FAISS
    dimension = embedding_model.dimension
    nlist = max(1, int(math.sqrt(len(search_engine.documents) or 1)))
    search_engine.indexer = FaissIndexer(dimension, nlist=nlist)

    return render_template(
        "index.html",
        uploaded_files=search_engine.uploaded_files,
        total_chunks=len(search_engine.documents),
        total_files=len(search_engine.uploaded_files)
    )

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return redirect(url_for("home"))

@app.route("/delete_file", methods=["POST"])
def delete_file():
    filename = request.form.get("filename")

    if filename not in search_engine.uploaded_files:
        return redirect(url_for("home"))

    # Remove file entry
    del search_engine.uploaded_files[filename]

    # Remove associated documents + metadata
    remaining_docs = []
    remaining_meta = []

    for doc, meta in zip(search_engine.documents, search_engine.doc_metadata):
        if meta["source"] != filename:
            remaining_docs.append(doc)
            remaining_meta.append(meta)

    search_engine.documents = remaining_docs
    search_engine.doc_metadata = remaining_meta

    # Rebuild FAISS index
    dimension = embedding_model.dimension
    nlist = max(1, int(math.sqrt(len(search_engine.documents) or 1)))
    search_engine.indexer = FaissIndexer(dimension, nlist=nlist)

    if search_engine.documents:
        search_engine.indexer.add(search_engine.embedding_model.encode(search_engine.documents))

    return redirect(url_for("home"))

# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)