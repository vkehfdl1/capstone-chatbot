from typing import List

from RAGchain.pipeline.base import BasePipeline
from RAGchain.retrieval import BM25Retrieval, VectorDBRetrieval, HybridRetrieval
from RAGchain.schema import Passage
from RAGchain.benchmark.dataset import QasperEvaluator
from RAGchain.utils.embed import EmbeddingFactory
from RAGchain.utils.vectorstore import ChromaSlim
from RAGchain.DB import PickleDB
import chromadb
import click
from dotenv import load_dotenv
import os

root_path = os.path.dirname(__file__)

load_dotenv()

bm25_retrieval = BM25Retrieval(save_path=os.path.join("bm25.pkl"))
vectordb_retrieval = VectorDBRetrieval(vectordb=ChromaSlim(
    client=chromadb.PersistentClient(path="Chroma"),
    collection_name="QasperChromaSlim",
    embedding_function=EmbeddingFactory('multilingual_e5', device_type='mps').get()
))
hybrid_retrieval = HybridRetrieval([bm25_retrieval, vectordb_retrieval], method='rrf', rrf_k=8, p=80)
db = PickleDB(os.path.join("pickle", "pickle.pkl"))


class OnlyRetrievalPipeline(BasePipeline):
    def __init__(self, retrieval):
        self.retrieval = retrieval

    def run(self, query: str) -> tuple[str, List[Passage]]:
        try:
            passages = self.retrieval.retrieve(query, top_k=5)
        except:
            passages = [Passage(content="Sample Passage", id="sample_id", filepath="sample_filepath")]
        return "Sample Answer", passages


def retrieval_qasper(evaluate_size: int, random_seed: int = 42, save_filepath=None):
    pipeline = OnlyRetrievalPipeline(hybrid_retrieval)
    evaluator = QasperEvaluator(pipeline, evaluate_size=evaluate_size,
                                metrics=['Recall', 'Precision', 'F1_score'])
    evaluator.ingest([hybrid_retrieval], db)

    result = evaluator.evaluate()
    print(result.results)
    if save_filepath is not None:
        result.each_results.to_csv(save_filepath, index=False)


@click.command()
@click.option("--evaluate_size", type=int)
@click.option("--save_filepath", type=str)
def main(evaluate_size, save_filepath):
    retrieval_qasper(evaluate_size=evaluate_size, save_filepath=save_filepath)


if __name__ == "__main__":
    main()
