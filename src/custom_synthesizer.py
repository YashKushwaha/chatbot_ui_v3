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
