## Custom RAG Pipelines with LLAMA INDEX

### Overview
This is my 3rd project exploring capabilities of llama index to build Gen AI applications
- [Version 1](https://github.com/YashKushwaha/chatbot_ui) - Explorers utilities for building chatbot/chat engines
- [Version 2](https://github.com/YashKushwaha/chatbot_ui_v2) - Explorers building agents, Function calling agent and ReAct agent was implemented

llama index abstracts a lot of steps which leads to loss of transparency and control over the behavior of the application. In this project, I have tried to build application using lower level APIs made available by llama index.

**Features of this project**
- Build a custom retriever that searches user query in multiple vector stores
- Custom pipeline to process results post retrieval, combined results are ranked based on similarity score and top 3 results are selected.
  - Also additional details to fetched from data base to enrich the nodes
- Custom response synthesizer which condenses results into single context than can be inserted as context into LLM query

### Data Pipeline

- [Standford Quastion Answering Dataset / squad](https://rajpurkar.github.io/SQuAD-explorer/) dataset available on [HuggingFace](https://huggingface.co/datasets/rajpurkar/squad) used to build knowledge base. Sample record
```json
{
  "id": "56be85543aeaaa14008c9063",
  "title": "Beyoncé",
  "context": "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles \"Crazy in Love\" and \"Baby Boy\".",
  "question": "When did Beyonce start becoming popular?",
  "answers": {
    "text": [
      "in the late 1990s"
    ],
    "answer_start": [
      269
    ]
  }
}
```

- Each record is broken into 2 records - QnA pair & context. Hash value of context is calculated to create an id for the text.
- Records are stored in mongo db collections - `qna` & `context`. This ensures that records are de duplicated and acts as long term storage / document store
- Records are fetched from mongo db collections, embeddings are calculated and stored into `chromadb` collections
  - collection 1 represents chunks of original text (i.e. paragraphs from wikipedia articles)
  - collection 2 represents synthetic questions that can be asked from a given piece of document. This method leverages query-to-question semantic alignment, using vector similarity to match user intent with pre-generated questions tied to source documents. 
  - To save space, instead of storing the original document in metadata of vector db, we store the details required to search the context in mongo db database
  


