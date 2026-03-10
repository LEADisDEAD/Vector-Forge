# VectorForge

VectorForge is a production-style Retrieval-Augmented Generation (RAG) system designed for intelligent document understanding, grounded response generation, and explainable semantic search — built entirely with local infrastructure.

It is not a chatbot demo.

It is a modular AI retrieval engine that demonstrates real-world RAG architecture, multi-stage retrieval logic, hallucination control, dynamic indexing, and full-stack deployment readiness.

---

## Application Preview

### 1. Landing Page
<img width="1918" height="867" alt="image" src="https://github.com/user-attachments/assets/2af4ed17-f972-4bc1-95bf-19858844d9e5" />


The landing page introduces VectorForge as a production-style Retrieval-Augmented Generation system.
It highlights:

- Semantic search capabilities
- Hybrid retrieval architecture
- Local LLM integration
- Performance instrumentation

This screen communicates the system’s purpose, architectural maturity, and production positioning before users enter the application.

### 2. Home Interface - Document Workspace

<img width="1919" height="863" alt="Screenshot 2026-02-23 140819" src="https://github.com/user-attachments/assets/8072227c-09c8-4558-b4b4-af1d6e37e53e" />

The primary workspace enables:

- Document upload and dynamic indexing
- Chunk tracking per file
- Conversation management
- Structured query interaction

The interface is intentionally minimal and dark-themed to maintain focus on retrieval and response generation.

### 3. Semantic Retrieval with Explainability

<img width="1918" height="867" alt="image" src="https://github.com/user-attachments/assets/306efda3-12c0-4fd3-bcd1-88caaa234f3a" />


This example demonstrates VectorForge’s full two-stage Retrieval-Augmented Generation pipeline.

The system performs:

- Hybrid retrieval (Dense + BM25)
- Cross-encoder second-stage reranking
- Measured ranking stability (MRR evaluation)
- Inline citation-grounded generation
- Structured evidence panel (always visible)
- Similarity-based confidence scoring
- Latency instrumentation (retrieval, rerank, LLM, total)

Retrieved document chunks are first ranked for recall.
They are then reranked using a cross-encoder for precision.

The final answer includes inline citations like [1], directly mapped to the displayed evidence blocks below the response.

This ensures transparency, traceability, and reduced hallucination risk.






## Core Capabilities

- Multi-document upload & dynamic indexing
- Hybrid retrieval (Dense embeddings + BM25 sparse search)
- Cross-encoder reranking (second-stage precision layer)
- Citation-grounded answer generation
- Structured always-visible evidence panel
- Measured retrieval evaluation (Precision@K, MRR)
- Similarity-based hallucination guardrails
- Intent-aware retrieval depth
- Session-safe source handling
- Latency instrumentation (retrieval, rerank, LLM, total)
- Clean SaaS-style UI

---

## Architecture Overview

VectorForge follows a modular, layered architecture designed to simulate production-grade Retrieval-Augmented Generation systems.

The system is divided into four primary layers:

### 1️. Interface Layer

- Handles user interaction through a Flask-based web application.
- Manages file uploads, chat state, and session memory.

### 2️. Retrieval Layer

Implements a two-stage retrieval pipeline:

Stage 1 — Recall:
- Dense embeddings using all-MiniLM-L6-v2
- Sparse lexical retrieval using BM25
- Weighted hybrid score fusion

Stage 2 — Precision:
- Cross-encoder reranking using ms-marco-MiniLM-L-6-v2
- Joint query–chunk relevance scoring
- Improved rank stability under semantic overlap

This layered retrieval design mirrors production search systems.

### 3️. Control & Safety Layer

- Implements guardrails and logic before LLM invocation.
- Intent detection (summary, explanation, fact lookup)
- Similarity-based hallucination prevention
- Dynamic top-K adjustment
- Empty-index handling

### 4️. Generation Layer

Uses a local LLM (Llama3 via Ollama) to generate grounded responses.

The LLM receives:
- Numbered retrieved chunks
- Structured instructions
- Query intent classification
- Strict citation formatting rules

Responses include inline citations like [1], directly mapped to structured evidence blocks displayed below the answer.

##  System Design Principles

VectorForge is built with:

- Separation of concerns (embeddings, indexing, retrieval, API)
- Modular architecture
- Explainable retrieval outputs
- Measurable latency tracking
- Safe fallback for empty or low-confidence states
- Fully local inference (no cloud APIs required)

## Retrieval Evaluation

VectorForge includes a retrieval evaluation harness to measure ranking performance under multi-document conditions.

Metrics used:
- Precision@K
- Mean Reciprocal Rank (MRR)

Under a 450+ chunk multi-document corpus:

- Hybrid-only retrieval: MRR ≈ 0.87
- Hybrid + Cross-Encoder reranking: MRR ≈ 0.90

This demonstrates measurable ranking improvement through second-stage precision refinement.

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

## Folder Structure

```
Vector-Forge/
│
├── app.py                    # Flask application entry point
├── search.py                 # Retrieval pipeline + LLM integration
├── indexer.py                # FAISS indexing logic
├── embeddings.py             # Embedding model wrapper
├── utils.py                  # Utility functions (chunking, helpers)
│
├── templates/                # HTML templates
│   ├── index.html
│   └── landing.html
│
├── static/                   # Frontend assets
│   └── style.css
│
├── uploads/                  # Uploaded documents (runtime storage)
│
├── requirements.txt
├── .gitignore
└── README.md
```
##  Running Locally

### 1️. Create virtual environment
```bash
python -m venv venv
```
Activate it(Windows Git Bash):

```bash
source venv/Scripts/activate
```
If using powershell:

```bash
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Ollama (Llama3):
Make sure Ollama is running and Llama3 is installed:
```bash
ollama run llama3
```
### 4. Run the Application

```bash
python app.py
```
### 5. Open in Browser

http://127.0.0.1:5000
 
---
## Example Use cases:


### 1. Academic Research Assistant


Upload Research Papers (PDF/TXT) and:
- Ask concept-level questions
- Generate document summaries
- Extract methodology or findings
- Compare sections across multiple papers

Ideal for students, researchers, and thesis work.

### 2. Internal Knowledge Base Search

Use VectorForge as a lightweight internal documentation engine:

- Upload technical docs
- Query system architecture
- Retrieve configuration details
- Summarize large documentation sets

Works well for startups or small teams without full enterprise search tooling.

### 3️. Technical Document Summarization

Instead of reading long whitepapers or reports:

- Generate structured summaries
- Extract key points
- Identify main contributions
- Quickly understand scope and limitations

### 4️. AI Safety Demonstration

Demonstrates:

- Context-grounded generation
- Similarity-based hallucination control
- Controlled retrieval depth
- Local-only inference pipeline

Useful for showcasing safe AI system design.

### 5️. RAG System Prototyping

VectorForge can serve as:

- A base RAG template
- A research prototype
- A starting point for hybrid retrieval systems
- A deployable foundation for production tools
---
## Notes & Design Decisions

- The system is intentionally fully local (no external APIs).

- Retrieval depth adapts based on query intent.

- Hallucination guardrails prevent low-confidence generation.

- Dynamic FAISS index rebuild ensures consistency after file deletion.

## Contributing

Contributions are welcome! Fork this repo, improve it, and submit a PR.
Suggestions for new models, UI improvements, or metric visualizations are highly encouraged.
Send me a mail on prathmeshbajpai123@gmail.com for further QnA.

---

## Author - Prathmesh Manoj Bajpai

[LinkedIn](https://www.linkedin.com/in/prathmesh-bajpai-8429652aa/)

---

## ⭐ Star the Repo

---
 
## 📦 Version History

---

### v2.5 – Citation-Grounded Generation + Structured Evidence Panel

- Inline citation formatting
- Structured always-visible sources section
- Session-safe snippet storage
- Improved UI hierarchy for research-style interaction

### v2.0 – Cross-Encoder Reranking

- Added second-stage reranking using ms-marco-MiniLM cross-encoder
- Improved precision of top-k context selection
- Two-stage retrieval pipeline (Recall + Precision)
- Cleaner source ranking

### v1.5 – Hybrid Retrieval Upgrade

- Added BM25 sparse retrieval
- Implemented weighted score fusion (alpha=0.7, beta=0.3)
- Normalized hybrid scoring
- Improved hallucination guardrail alignment
- Fixed Flask session cookie overflow issue

### v1.0 – Initial RAG System

- Dense semantic search using SentenceTransformers + FAISS
- Local LLM integration (Ollama + Llama3)
- Intent-aware retrieval
- Retrieval & latency metrics
- Clean SaaS-style UI


