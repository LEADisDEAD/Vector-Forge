import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from config import LLM_MODE, API_KEY, API_MODEL


class SemanticSearch:
    def __init__(self, embedding_model, indexer):
        self.embedding_model = embedding_model
        self.indexer = indexer
        self.documents = []
        self.doc_metadata = []
        self.uploaded_files = {}
        self.answer_cache = {}
        self.tokenized_docs = []
        self.bm25 = None
        
        if LLM_MODE == "local":
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        else:
            self.reranker = None

    def add_documents(self, documents, source_name=None):

        start_index = len(self.documents)

        self.documents.extend(documents)
        
        # Update BM25
        self.tokenized_docs = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        embeddings = self.embedding_model.encode(documents)
        self.indexer.add(embeddings)

        for i in range(len(documents)):
            self.doc_metadata.append({
                "source": source_name,
                "chunk_index": start_index + i
            })

      
        if source_name:
            if source_name in self.uploaded_files:
                self.uploaded_files[source_name] += len(documents)
            else:
                self.uploaded_files[source_name] = len(documents)
                
    def query(self, text, top_k=3):

        # Dense Retrieval
        dense_candidate_k = top_k * 5
        query_vector = self.embedding_model.encode([text])
        distances, indices = self.indexer.search(query_vector, dense_candidate_k)

        dense_results = {}

        for rank, idx in enumerate(indices[0]):
            similarity = float(distances[0][rank])

            dense_results[int(idx)] = {
                "chunk_id": int(idx),
                "dense_score": similarity,
                "source": self.doc_metadata[idx]["source"],
                "text": self.documents[idx]
            }

        # Sparse Retrieval--- BM25
        sparse_results = {}
        if self.bm25 is not None:
            tokenized_query = text.split()
            scores = self.bm25.get_scores(tokenized_query)

            top_sparse_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:dense_candidate_k]

            for idx in top_sparse_indices:
                sparse_results[int(idx)] = scores[idx]

        # Score Normalization 
        if dense_results:
            max_dense = max([v["dense_score"] for v in dense_results.values()])
        else:
            max_dense = 1

        if sparse_results:
            max_sparse = max(sparse_results.values())
        else:
            max_sparse = 1

        # Fusion 
        alpha = 0.7  # weight for dense
        beta = 0.3   # weight for sparse

        hybrid_results = []

        all_indices = set(dense_results.keys()).union(set(sparse_results.keys()))

        for idx in all_indices:

            dense_score = dense_results.get(idx, {}).get("dense_score", 0)
            sparse_score = sparse_results.get(idx, 0)

            normalized_dense = dense_score / max_dense if max_dense else 0
            normalized_sparse = sparse_score / max_sparse if max_sparse else 0

            final_score = (alpha * normalized_dense) + (beta * normalized_sparse)

            hybrid_results.append({
                "chunk_id": idx,
                "similarity_score": round(dense_score, 4),
                "final_score": round(final_score, 4),
                "source": self.doc_metadata[idx]["source"],
                "text": self.documents[idx]
            })

        # First stage Ranking (Hybrid) 
        hybrid_results = sorted(
            hybrid_results,
            key=lambda x: x["final_score"],
            reverse=True
        )

        #Cross-Encoder Reranking
        

        # If in production mode - API, skip reranking to reduce memory usage
        # Update: project cannot be deployed due to architectural limits and limited RAM 
        if LLM_MODE == "api":
            final_results = hybrid_results[:top_k]

            for i, result in enumerate(final_results):
                result["citation_id"] = i + 1

            return final_results


        # (local mode), apply reranking
        rerank_k = min(10, len(hybrid_results))
        top_candidates = hybrid_results[:rerank_k]

        if top_candidates:
            query_chunk_pairs = [
                (text, candidate["text"])
                for candidate in top_candidates
            ]

            rerank_scores = self.reranker.predict(query_chunk_pairs)

            for i, score in enumerate(rerank_scores):
                top_candidates[i]["rerank_score"] = float(score)

            top_candidates = sorted(
                top_candidates,
                key=lambda x: x["rerank_score"],
                reverse=True
            )

            final_results = top_candidates[:top_k]

            for i, result in enumerate(final_results):
                result["citation_id"] = i + 1

            return final_results

        # Fallback
        return hybrid_results[:top_k]
    
    def bm25_search(self, text, top_k=3):
        if self.bm25 is None:
            return []

        tokenized_query = text.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for idx in ranked_indices:
            results.append({
                "chunk_id": int(idx),
                "similarity_score": round(float(scores[idx]), 4),
                "final_score": round(float(scores[idx]), 4),
                "source": self.doc_metadata[idx]["source"],
                "text": self.documents[idx]
            })

        return results
    
    def query_with_context(self, text, top_k=3):
        query_vector = self.embedding_model.encode([text])
        distances, indices = self.indexer.search(query_vector, top_k)

        retrieved_chunks = []
        for idx in indices[0]:
            retrieved_chunks.append(self.documents[idx])

        # Combine into context block
        context = "\n\n".join(retrieved_chunks)

        return context
    
    def generate_answer_with_llm(self, question, results):
        
        if question in self.answer_cache:
            return self.answer_cache[question]
        
        # Context Size control --->also Token awareness
        
        max_chars = 2000
        context = ""

        for r in results:
            if len(context) + len(r["text"]) < max_chars:
                context += f"[{r['citation_id']}] {r['text']}\n\n"

        prompt = f"""
You are a precise and structured AI assistant.

Use ONLY the provided information below to answer the question.
Do NOT introduce external knowledge.
If the answer is not found, respond exactly with:
Answer not found in documents.

Formatting rules:

- Write in clear, natural language.
- Use short paragraphs.
- Avoid unnecessary symbols or decorative formatting.
- Do not repeat the question.
- Do not mention the word "context".
- When using information, cite it using only the number in square brackets like [1].
- Place citation numbers directly at the end of the relevant sentence.
- Do NOT write words like "Citation" or any explanation about the reference.

If the user asks to:
- Summarize → Provide a concise summary of key ideas.
- Explain → Provide a simple and clear explanation.
- List → Present each point on a new line.
- Compare → Show differences clearly on separate lines.
- Define → Provide a short and precise definition.

context:

Information:
{context}

User Question:
{question}

Provide a concise, well-structured answer.
"""

        if LLM_MODE == "local":

            response = ollama.chat(
                model="llama3",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0}
            )

            answer = response["message"]["content"]
           

        elif LLM_MODE == "api":
            
            from groq import Groq

            client = Groq(api_key=API_KEY)

            response = client.chat.completions.create(
                model=API_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            answer = response.choices[0].message.content
        
        import re

        def highlight_citations(text):
            return re.sub(r"\[(\d+)\]", r'<span class="citation">[\1]</span>', text)

        answer = highlight_citations(answer)

        #finally stored in cache
        self.answer_cache[question] = answer

        return answer
