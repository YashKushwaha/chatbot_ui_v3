from llama_index.core.tools.types import BaseTool, ToolMetadata

class RAGTool(BaseTool):
    
    def __init__(self, retriever, postprocessor, synthesizer, llm):
        self._metadata = ToolMetadata(
            name="RAGTool",
            description="Retrieves, processes, and synthesizes information using a RAG pipeline.",
        )
        self.retriever = retriever
        self.postprocessor = postprocessor
        self.synthesizer = synthesizer
        self.llm = llm

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(self, query: str) -> str:
        nodes = self.retriever._retrieve(query)
        processed_nodes = self.postprocessor.postprocess_nodes(nodes)
        compressed_context = self.synthesizer.synthesize(processed_nodes)
        
        prompt = f"Answer using the provided context:\n\nContext:\n{compressed_context}\n\nQuestion: {query}"
        response = self.llm.complete(prompt)

        return response
