from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.schema import NodeWithScore
from llama_index.core.response import Response

class CustomSynthesizer(BaseSynthesizer):
    def __init__(self, llm=None):
        self.llm = llm  # Can be any LLM client

    def synthesize(self, query: str, nodes: list[NodeWithScore]) -> Response:
        # Combine node texts
        '''
        context_text = "\n".join([node.text for node in nodes])

        prompt = f"""
        Context:
        {context_text}

        Based on the above, answer the following:
        {query}
        """
        '''
        llm_response = '\n'.join({node.text for node in nodes})
        #llm_response = self.llm.complete(prompt)

        return Response(response=llm_response)

    def _get_prompts(self):
        # Return your prompt configurations
        return None

    def _update_prompts(self, prompts):
        # Optional: logic to update internal prompts
        pass

    def get_response(self, query_bundle, nodes, **kwargs):
        """
        Combines retrieved nodes and user query into a final response.
        """
        # Example: simple concatenation for illustration
        context = "\n".join([node.text for node in nodes])
        return f"Query: {query_bundle.query_str}\nContext:\n{context}"

    async def aget_response(self, query_bundle, nodes, **kwargs):
        """
        Async variant of get_response.
        """
        return self.get_response(query_bundle, nodes, **kwargs)