import os
from typing import Optional

import chromadb
import openai
from dotenv import load_dotenv

from ingest_qasper import IngestQasper


class RetrievalQasper:
    def __init__(self, chroma_dir: str, chroma_collection_name: str = 'qasper'):
        if not os.path.exists(chroma_dir):
            raise ValueError(f"{chroma_dir} is not exist")
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection(name=chroma_collection_name)
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')

    def query(self, query: str, n_results: int = 5, doi: Optional[str] = None):
        query_vector = IngestQasper.embed_openai(query)
        if doi is None:
            result = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results
            )
        else:
            result = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where={'doi': doi}
            )
        return result


def main():
    retrieval = RetrievalQasper(chroma_dir='./Chroma')
    query = input('query: ')
    result = retrieval.query(query)
    for _id, doc in zip(result['ids'][0], result['documents'][0]):
        print(f'id: {_id}')
        print(doc)
        print('---'*10)


if __name__ == '__main__':
    main()
