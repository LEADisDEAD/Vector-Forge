# VectorForge

VectorForge is a production-style Retrieval-Augmented Generation (RAG) system designed for intelligent document understanding, grounded response generation, and explainable semantic search — built entirely with local infrastructure.

It is not a chatbot demo.

It is a modular AI retrieval engine that demonstrates real-world RAG architecture, multi-stage retrieval logic, hallucination control, dynamic indexing, and full-stack deployment readiness.

---

## Core Capabilities

- Multi-document upload & dynamic indexing
- Dense semantic search using Sentence-Transformers
- FAISS Approximate Nearest Neighbor (ANN) indexing
- Intent-aware retrieval (fact-level vs document-level queries)
- Hallucination guardrails via similarity thresholds
- Context-aware summarization
- Session-based conversation memory
- File deletion with full index rebuild
- Latency instrumentation (retrieval, LLM, total)
- Clean SaaS-style UI

---

## 🧠 Architecture Overview

VectorForge follows a structured multi-layer pipeline:
User Query
↓
Intent Detection Layer
↓
Embedding Model (all-MiniLM-L6-v2)
↓
FAISS Vector Search (Cosine Similarity)
↓
Top-K Context Retrieval
↓
Similarity Guardrail
↓
Local LLM (Llama3 via Ollama)
↓
Structured Response


The system dynamically adapts retrieval depth based on query intent (fact lookup vs document summarization).

---

## 🏗 System Design Principles

VectorForge is built with:

- Separation of concerns (embeddings, indexing, retrieval, API)
- Modular architecture
- Explainable retrieval outputs
- Measurable latency tracking
- Safe fallback for empty or low-confidence states
- Fully local inference (no cloud APIs required)

---

## 🛠 Tech Stack

**Backend**
- Python
- Flask
- FAISS (ANN search)
- Sentence-Transformers
- Ollama (Llama3 local inference)

**Frontend**
- HTML
- Custom CSS (minimal SaaS design)
- Lightweight JavaScript

---

## 📊 Performance Characteristics

- Retrieval latency: ~1–5 ms
- LLM latency: ~2–4 seconds (local inference)
- Zero external API dependency
- Dynamic indexing without server restart

---

## 🧩 Engineering Highlights

- Cosine similarity via normalized embeddings
- Hybrid retrieval logic
- Similarity-based hallucination guardrails
- Session-managed conversation state
- Dynamic FAISS index rebuild on file deletion
- Persistent upload handling
- Clean Git-based version control

---

## 📦 Running Locally

### 1️⃣ Create virtual environment
```bash
python -m venv venv
source venv/Scripts/activate
