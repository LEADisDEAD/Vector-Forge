# VectorForge

VectorForge is a production-style Retrieval-Augmented Generation (RAG) system designed for intelligent document understanding, grounded response generation, and explainable semantic search — built entirely with local infrastructure.

It is not a chatbot demo.

It is a modular AI retrieval engine that demonstrates real-world RAG architecture, multi-stage retrieval logic, hallucination control, dynamic indexing, and full-stack deployment readiness.

---

## 🚀 Core Capabilities

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
