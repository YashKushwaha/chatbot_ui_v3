from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.doc_store.base import BaseDocumentStore
from llama_index.core.schema import Document

class MongoDocumentStore(BaseDocumentStore):
    def __init__(self, mongo_collection):
        self.collection = mongo_collection

    def add(self, documents: list[Document]):
        """Store documents in MongoDB"""
        for doc in documents:
            self.collection.insert_one({
                "_id": doc.id_,
                "text": doc.text,
                "metadata": doc.metadata
            })

    def get(self, doc_id: str) -> Document:
        """Retrieve a document by ID"""
        doc_data = self.collection.find_one({"_id": doc_id})
        if doc_data:
            return Document(
                id_=doc_data["_id"],
                text=doc_data["text"],
                metadata=doc_data.get("metadata", {})
            )
        return None

    def delete(self, doc_id: str):
        self.collection.delete_one({"_id": doc_id})

    def get_context(self, context_hash):
        for i in self.collection.find(dict(context_hash=context_hash)):
            return i['context']


class CustomPostProcessor(BaseNodePostprocessor):
    def __init__(self, doc_store, score_threshold=0.5, top_k=5):
        self.doc_store = doc_store
        self.score_threshold = score_threshold
        self.top_k = top_k

    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_str: str = None) -> list[NodeWithScore]:
        # 1. Filter by score threshold
        filtered_nodes = [n for n in nodes if n.score and n.score >= self.score_threshold]

        # 2. Sort by score
        sorted_nodes = sorted(filtered_nodes, key=lambda n: n.score, reverse=True)

        # 3. Limit to top_k
        final_nodes = sorted_nodes[:self.top_k]

        # 4. Enrich nodes with external data
        for node in final_nodes:
            if node.metadata.get('collection') == 'qna':
                context_hash = node.metadata.get('context_hash')
                context = self.do_store.get_context(context_hash)
                node.text = context  # Replace or extend the node text as required

        return final_nodes

