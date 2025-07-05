from llama_index.core.tools import BaseTool

class RAGTool(BaseTool):
    def __init__(self, retriever, postprocessor, synthesizer, llm):
        self.retriever = retriever
        self.postprocessor = postprocessor
        self.synthesizer = synthesizer
        self.llm = llm

    def _run(self, query: str) -> str:
        nodes = self.retriever.retrieve(query)
        processed = self.postprocessor.postprocess_nodes(nodes, query)
        compressed_context = self.synthesizer.synthesize(query, processed).response
        final_prompt = f"Context:\n{compressed_context}\n\nQuestion:\n{query}\nAnswer concisely:"
        return self.llm.complete(final_prompt)

    @property
    def metadata(self):
        return {"name": "rag_tool", "description": "Retrieves context and answers user questions"}
