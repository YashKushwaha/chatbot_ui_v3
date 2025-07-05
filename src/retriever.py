from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.schema import QueryBundle

def get_retriever(chroma_collection, embed_model, similarity_top_k=3):
    vec_store = ChromaVectorStore(chroma_collection = chroma_collection)
    index = VectorStoreIndex.from_vector_store(vec_store, embed_model=embed_model)
    retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k
        )
    return get_retriever

class SQUAD_DATA_RETRIEVER(BaseRetriever):
    def __init__(self, chromadb_client, collection_names, embed_model, similarity_top_k=3):
        self.client = chromadb_client
        self.collection_names = collection_names
        self.embed_model = embed_model
        self.similarity_top_k = similarity_top_k
        self.retrievers = self._load_retrievers()
    
    def _load_retrievers(self):
        retrievers = {}
        for collection_name in self.collection_names:
            collection = self.client.get_collection(collection_name)
            vec_store = ChromaVectorStore(chroma_collection=collection)
            index = VectorStoreIndex.from_vector_store(vec_store, embed_model=self.embed_model)
            retriever = VectorIndexRetriever(index=index, similarity_top_k=self.similarity_top_k)
            retrievers[collection_name] = retriever
        return retrievers

    def _retrieve(self, query_bundle: QueryBundle):
        all_nodes = []
        for retriever in self.retrievers.values():
            nodes = retriever.retrieve(query_bundle)
            all_nodes.extend(nodes)
        return all_nodes