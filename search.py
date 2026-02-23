import ollama



class SemanticSearch:
    def __init__(self, embedding_model, indexer):
        self.embedding_model = embedding_model
        self.indexer = indexer
        self.documents = []
        self.doc_metadata = []
        self.uploaded_files = {}
        self.answer_cache = {}

    def add_documents(self, documents, source_name=None):

        start_index = len(self.documents)

        self.documents.extend(documents)

        embeddings = self.embedding_model.encode(documents)
        self.indexer.add(embeddings)

        for i in range(len(documents)):
            self.doc_metadata.append({
                "source": source_name,
                "chunk_index": start_index + i
            })

        # 🔥 This part is important
        if source_name:
            if source_name in self.uploaded_files:
                self.uploaded_files[source_name] += len(documents)
            else:
                self.uploaded_files[source_name] = len(documents)
                
    def query(self, text, top_k=3):

    # 🔹 Step 1: Retrieve more candidates from FAISS
        candidate_k = top_k * 5
        query_vector = self.embedding_model.encode([text])
        distances, indices = self.indexer.search(query_vector, candidate_k)

        results = []

        for rank, idx in enumerate(indices[0]):
            similarity = float(distances[0][rank])

            doc_text = self.documents[idx]

            # 🔹 Better lexical boost (token overlap, not full string match)
            query_tokens = set(text.lower().split())
            doc_tokens = set(doc_text.lower().split())
            overlap = len(query_tokens.intersection(doc_tokens))

            keyword_bonus = 0.02 * overlap  # scaled boost

            final_score = similarity + keyword_bonus

            results.append({
                "chunk_id": int(idx),
                "similarity_score": round(similarity, 4),
                "final_score": round(final_score, 4),
                "source": self.doc_metadata[idx]["source"],
                "text": doc_text
            })

        # 🔹 True reranking happens here
        results = sorted(results, key=lambda x: x["final_score"], reverse=True)

        # 🔹 Return only top_k after reranking
        return results[:top_k]
    
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
        
        # Context Size control - also Token awareness
        
        max_chars = 2000
        context = ""

        for r in results:
            if len(context) + len(r["text"]) < max_chars:
                context += r["text"] + "\n\n"

        prompt = f"""
You are a precise and structured AI assistant.

Use ONLY the provided context to answer the question.
Do NOT introduce external knowledge.
If the answer is not found in the context, respond exactly with:
"Answer not found in documents."

Follow these formatting rules:

- Write in clear, natural language.
- Avoid excessive special characters or symbols.
- Use short paragraphs.
- If listing points, place each point on a new line.
- Do not repeat the question.
- Do not mention the word "context".

If the user asks to:
- Summarize → Provide a concise summary covering key ideas.
- Explain → Provide a clear explanation in simple language.
- List → Provide structured bullet-style points.
- Compare → Present differences clearly in separate lines.
- Define → Provide a short and precise definition.

Context:
{context}

User Question:
{question}

Provide a concise, well-structured answer.
"""

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        
        answer = response["message"]["content"]

        # 🔹 Store in cache
        self.answer_cache[question] = answer

        return answer
