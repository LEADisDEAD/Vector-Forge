from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self,model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    @property
    def dimension(self):
        return self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts):
        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        return embeddings