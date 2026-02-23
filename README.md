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

## Architecture Overview

VectorForge follows a modular, layered architecture designed to simulate production-grade Retrieval-Augmented Generation systems.

The system is divided into four primary layers:

### 1️. Interface Layer

- Handles user interaction through a Flask-based web application.
- Manages file uploads, chat state, and session memory.

### 2️. Retrieval Layer

- Responsible for document processing and semantic search.
- Documents are chunked and embedded using all-MiniLM-L6-v2.
- Embeddings are normalized and stored in a FAISS index.
- Cosine similarity is used for approximate nearest neighbor search.
- Retrieval depth adapts based on query intent (fact-level vs document-level queries).

### 3️. Control & Safety Layer

- Implements guardrails and logic before LLM invocation.
- Intent detection (summary, explanation, fact lookup)
- Similarity-based hallucination prevention
- Dynamic top-K adjustment
- Empty-index handling

### 4️. Generation Layer

Uses a local LLM (Llama3 via Ollama) to generate grounded responses strictly from retrieved context.

The LLM receives:

- Retrieved document chunks
- Structured system instructions
- User query
- Output formatting constraints
---

##  System Design Principles

VectorForge is built with:

- Separation of concerns (embeddings, indexing, retrieval, API)
- Modular architecture
- Explainable retrieval outputs
- Measurable latency tracking
- Safe fallback for empty or low-confidence states
- Fully local inference (no cloud APIs required)

---

##  Tech Stack

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

##  Performance Characteristics

- Retrieval latency: ~1–5 ms
- LLM latency: ~2–4 seconds (local inference)
- Zero external API dependency
- Dynamic indexing without server restart

---

##  Engineering Highlights

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
