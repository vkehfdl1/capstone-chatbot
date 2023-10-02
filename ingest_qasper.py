import json
import os
from typing import List

from tqdm import tqdm
import click

import openai
import chromadb
from dotenv import load_dotenv


class IngestQasper:
    def __init__(self, qasper_path: str, chroma_dir: str):
        load_dotenv(verbose=False)
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY is not set")
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not os.path.exists(qasper_path):
            raise ValueError(f"{qasper_path} is not exist")
        self.data = self.load_qasper(qasper_path)
        if not os.path.exists(chroma_dir):
            os.makedirs(chroma_dir)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection_name = 'qasper'
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def load_qasper(self, qasper_path: str):
        with open(qasper_path, 'rb') as r:
            data = json.load(r)
        return data

    def ingest(self):
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        for idx, doi in enumerate(tqdm(list(self.data.keys()))):
            if idx > 10:  # for testing. Need to delete for full ingestion
                break
            full_text = self.data[doi]['full_text']
            for text in full_text:
                section_name = text['section_name']
                paragraphs: List[str] = text['paragraphs']
                for i, paragraph in enumerate(paragraphs):
                    paragraph_id = f'{doi}_{section_name}_{i}'
                    embed_vectors = self.embed_openai(paragraph)
                    documents.append(paragraph)
                    embeddings.append(embed_vectors)
                    ids.append(paragraph_id)
                    metadatas.append({'doi': doi, 'section_name': section_name})

        # save to chroma
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings
        )

    @staticmethod
    def embed_openai(sentence: str) -> List[float]:
        response = openai.Embedding.create(
            model='text-embedding-ada-002',
            input=sentence
        )
        return response['data'][0]['embedding']


@click.command()
@click.option('--qasper_path', type=str, required=True, default='./data/qasper/qasper-dev-v0.3.json')
@click.option('--chroma_dir', type=str, required=True, default='./Chroma')
def main(qasper_path: str, chroma_dir: str):
    IngestQasper(qasper_path, chroma_dir).ingest()


if __name__ == '__main__':
    main()
