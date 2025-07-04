from pathlib import Path
import os
from datasets import load_dataset
from pymongo import MongoClient
import hashlib

def get_context_hash(context: str) -> str:
    return hashlib.md5(context.encode('utf-8')).hexdigest()

def batch_uploader(iterator, context_collection, question_collection, batch_size=100):
    context_batch = []
    question_batch = []

    for split_record in iterator:
        context_batch.append(split_record["context_record"])
        question_batch.append(split_record["question_record"])

        if len(context_batch) >= batch_size:
            context_collection.insert_many(context_batch)
            question_collection.insert_many(question_batch)
            context_batch.clear()
            question_batch.clear()

    # Final flush
    if context_batch:
        context_collection.insert_many(context_batch)
        question_collection.insert_many(question_batch)


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
            "question_record": dict(question=question, answer=answer, title=title, context_hash=context_hash)
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

def upload_data_to_mongo_db(dataset):
    mongo_db_client = get_mongo_db_client()
    
    database = mongo_db_client['chatbot_ui_v3']  
    context_collection = database['context']
    qna_collection = database['qna']

    iterator = SQuADIterator(dataset['train'])
    print('Done with records:')
    batch_size = 128
    counter = 0
    for batch in iterator.batch(batch_size=batch_size):
        context_records = [item['context_record'] for item in batch]
        question_records = [item['question_record'] for item in batch]

        context_collection.insert_many(context_records)
        qna_collection.insert_many(question_records)
        counter+=batch_size
        if counter % (10*batch_size) == 0:
            print(counter, end='\t')


if __name__ == '__main__':
    CACHE_DIR = os.path.join(Path(__file__).resolve().parents[1] , "local_only", "data")
    os.makedirs(CACHE_DIR, exist_ok=True)

    DATASET_NAME = "rajpurkar/squad_v2"

    dataset = download_squad_dataset(dataset_name = DATASET_NAME, download_location = CACHE_DIR)
    create_dataset_summary(dataset, dataset_name = DATASET_NAME, download_location = CACHE_DIR)    
    upload_data_to_mongo_db(dataset)
