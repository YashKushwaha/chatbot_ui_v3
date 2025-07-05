from pathlib import Path
import os
from datasets import load_dataset
from pymongo import MongoClient, ASCENDING, UpdateOne
import hashlib
import chromadb
from src.embedding_client import RemoteEmbedding

def get_context_hash(context: str) -> str:
    return hashlib.md5(context.encode('utf-8')).hexdigest()

def get_batches_from_collection(collection, batch_size=100):
    """
    Generator to yield batches of documents from a MongoDB collection.

    :param collection: pymongo collection object
    :param batch_size: Number of documents per batch
    """
    cursor = collection.find().batch_size(batch_size)
    batch = []
    
    for doc in cursor:
        batch.append(doc)
        if len(batch) == batch_size:
            yield batch
            batch = []
    
    if batch:
        yield batch  # Yield any remaining documents

class SQuADIterator:
    def __init__(self, dataset_split):
        self.dataset_split = dataset_split

    def transform(self, record):
        title = record['title']
        context = record['context']
        question = record['question']

        answer = '|'.join({i for i in record['answers'].get('text', [])})
        
        context_hash = get_context_hash(context)
        return {
            "context_record": dict(context_hash=context_hash, context=context, title=title),
            "question_record": dict(question=question, answer=answer, title=title, context_hash=context_hash, id = record['id'])
        }

    def __iter__(self):
        for record in self.dataset_split:
            yield self.transform(record)

    def batch(self, batch_size):
        """
        Yield records in batches of batch_size
        """
        batch = []
        for record in self:
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch  # Yield remaining records if any

def download_squad_dataset(dataset_name, download_location=None):
    dataset = load_dataset(dataset_name, cache_dir=download_location)
    return dataset

def create_dataset_summary(dataset, dataset_name, download_location):
    data_folder_name = dataset_name.replace("/", "___")
    summary_path = os.path.join(download_location, f"{data_folder_name}_dataset_summary.txt")
    # Example summary
    with open(summary_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Available Splits:\n")
        
        for split in dataset.keys():
            num_records = len(dataset[split])
            f.write(f"  - {split}: {num_records} records\n")
        
        for split in dataset.keys():
            f.write(f"\nExample Record from '{split}' split:\n")
            f.write(f"{dataset[split][0]}\n")

def get_mongo_db_client():
    client = MongoClient("mongodb://localhost:27017/")
    assert client.admin.command("ping") == {'ok': 1.0}
    return client

def get_chroma_db_client():
    client = chromadb.HttpClient(
        host="localhost",
        port=int(CHROMA_DB_PORT))
    return client

def upload_data_to_mongo_db(dataset):
    mongo_db_client = get_mongo_db_client()
    
    database = mongo_db_client['chatbot_ui_v3']  

    context_collection = database['context']
    context_collection.create_index([("context_hash", ASCENDING)], unique=True)

    qna_collection = database['qna']
    qna_collection.create_index([("id", ASCENDING)], unique=True)

    iterator = SQuADIterator(dataset['train'])
    print('Done with records:')
    batch_size = 128
    counter = 0
    for batch in iterator.batch(batch_size=batch_size):
        context_records = [item['context_record'] for item in batch]
        context_records = [UpdateOne({'context_hash': i['context_hash']}, {'$set': i}, upsert=True) for i in context_records]
        context_collection.bulk_write(context_records)

        question_records = [item['question_record'] for item in batch]
        question_records = [UpdateOne({'id': i['id']}, {'$set': i}, upsert=True) for i in question_records]
        qna_collection.bulk_write(question_records)
        counter+=batch_size
        if counter % (10*batch_size) == 0:
            print(counter, end='\t')


def prepare_context_data_for_vec_db(collection, embed_model, batch_size=64):
    for batch in get_batches_from_collection(collection, batch_size=batch_size):
        documents = [i['context'] for i in batch]
        ids = [i['context_hash'] for i in batch]
        embeddings = embed_model._get_text_embeddings(documents)
        metadatas = [dict(title = i['title'], context_hash = i['context_hash'],
                          collection = collection.name, database = collection.database.name) for i in batch]
        yield dict(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

def prepare_qna_data_for_vec_db(collection, embed_model,batch_size=64):
    for batch in get_batches_from_collection(collection, batch_size=batch_size):
        documents = [i['question'] for i in batch]
        ids = [i['id'] for i in batch]
        embeddings = embed_model._get_text_embeddings(documents)
        metadatas = [dict(title = i['title'], context_hash = i['context_hash'],answer = i['answer'],
                          collection = collection.name, database = collection.database.name) for i in batch]
        yield dict(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
 
def upload_data_to_vec_db(embed_model, context_data_mongo_db_collection, qna_data_mongo_db_collection):
    vec_db_client = get_chroma_db_client()
    batch_size =64
    context_collection = vec_db_client.get_or_create_collection('context')
    qna_collection = vec_db_client.get_or_create_collection('qna')

    print('Adding context data to vec db')
    counter=0
    for batch in prepare_context_data_for_vec_db(context_data_mongo_db_collection, embed_model, batch_size):
        context_collection.add(**batch)
        counter+=batch_size
        if counter % 100*batch == 0:
            print(counter, end='\t')
 
    print()
    print('Adding qna data to vec db')
    counter=0    
    for batch in prepare_qna_data_for_vec_db(qna_data_mongo_db_collection, embed_model, batch_size):
        qna_collection.add(**batch) 
        counter+=batch_size
        if counter % 100*batch == 0:
            print(counter, end='\t')

    print()
if __name__ == '__main__':
    CHROMA_DB_PORT = 8010
    CACHE_DIR = os.path.join(Path(__file__).resolve().parents[1] , "local_only", "data")
    os.makedirs(CACHE_DIR, exist_ok=True)

    DATASET_NAME = "rajpurkar/squad_v2"

    dataset = download_squad_dataset(dataset_name = DATASET_NAME, download_location = CACHE_DIR)
    #create_dataset_summary(dataset, dataset_name = DATASET_NAME, download_location = CACHE_DIR)    
    #upload_data_to_mongo_db(dataset)

    EMBEDDING_SERVER_PORT = 8020
    embed_model = RemoteEmbedding(f"http://localhost:{EMBEDDING_SERVER_PORT}")

    mongo_db_client = get_mongo_db_client()
    database = mongo_db_client['chatbot_ui_v3']  
    context_collection = database['context']
    qna_collection = database['qna']
    upload_data_to_vec_db(embed_model, context_collection, qna_collection)