import chromadb
from llama_index.core.agent import AgentRunner
from llama_index.llms.ollama import Ollama

from src.custom_tools import RAGTool
from src. retriever import SQUAD_DATA_RETRIEVER
from src.embedding_client import RemoteEmbedding
from src.post_retrieval_processing import CustomPostProcessor
from src.custom_synthesizer import CustomSynthesizer

from pymongo import MongoClient

def get_mongo_db_client():
    client = MongoClient("mongodb://localhost:27017/")
    assert client.admin.command("ping") == {'ok': 1.0}
    return client

def get_chroma_db_client(port):
    client = chromadb.HttpClient(
        host="localhost",
        port=int(port))
    return client

def get_ollama_llm():
    model = "qwen3:14b"
    context_window = 1000

    llm = Ollama(
        model=model,
        request_timeout=120.0,
        thinking=True,
        context_window=context_window,
    )
    return llm

def get_agent():
    EMBEDDING_SERVER_PORT = 8020
    CHROMA_DB_PORT = 8010

    chromadb_client = get_chroma_db_client(CHROMA_DB_PORT)
    collection_names = ['qna', 'context']    
    embed_model = RemoteEmbedding(f"http://localhost:{EMBEDDING_SERVER_PORT}")
    retriever = SQUAD_DATA_RETRIEVER(chromadb_client, collection_names, embed_model,similarity_top_k=3)

    mongodb_client = get_mongo_db_client()
    doc_store = mongodb_client['chatbot_ui_v3']['context']
    postprocessor = CustomPostProcessor(doc_store, score_threshold=0.5, top_k=3)
    synthesizer = CustomSynthesizer()

    llm = get_ollama_llm()

    rag_tool = RAGTool(retriever, postprocessor, synthesizer, llm)
    agent = AgentRunner(tools=[rag_tool])
    return agent

if __name__ == '__main__':
    agent = get_agent()
    response = agent.chat("How does process X affect outcome Y?")
    print(response.response)
