import faiss
import numpy as np

class FaissIndexer:
    def __init__(self, dimension, nlist=100):
        self.dimension = dimension
        self.nlist = nlist

        quantizer = faiss.IndexFlatIP(dimension)

        self.index = faiss.IndexIVFFlat(
            quantizer,
            dimension,
            nlist,
            faiss.METRIC_INNER_PRODUCT
        )

    def add(self, vectors):
        # IMPORTANTT: IVF needs to be trained before adding
        if not self.index.is_trained:
            self.index.train(vectors)

        self.index.add(vectors)

    def search(self, query_vector, top_k=2):
        # nprobe = number of clusters to search
        self.index.nprobe = min(10, self.nlist)
        distances, indices = self.index.search(query_vector, top_k)
        return distances, indices

    def total_vectors(self):
        return self.index.ntotal