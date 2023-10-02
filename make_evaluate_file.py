import json
import os
from tqdm import tqdm

from retrieval_qasper import RetrievalQasper
import click


class MakeEvaluateFile:
    def __init__(self,
                 qasper_filepath: str,
                 chroma_dir: str, chroma_collection_name: str = 'qasper'):
        self.retrieval = RetrievalQasper(chroma_dir=chroma_dir, chroma_collection_name=chroma_collection_name)
        if not os.path.exists(qasper_filepath):
            raise ValueError(f"{qasper_filepath} is not exist")
        with open(qasper_filepath, 'rb') as r:
            self.qasper = json.load(r)

    def make_evaluator_file(self, save_path: str, n_results: int = 5):
        evaluate_result = []
        # retrieve passage
        for i, doi in tqdm(enumerate(list(self.qasper.keys()))):
            if i > 10:  # TODO: This is for testing. Remove this line.
                break
            data = self.qasper[doi]
            for qa in data['qas']:
                question_id = qa['question_id']
                question = qa['question']
                result = self.retrieval.query(query=question, n_results=n_results, doi=doi)
                retrieved_list = []
                for doc in result['documents'][0]:
                    retrieved_list.append(doc)

                # make answer with LLM
                answer = "This is a sample answer."

                # save
                evaluate_result.append({
                    "question_id": question_id,
                    "predicted_answer": answer,
                    "predicted_evidence": retrieved_list
                })

        # save file
        with open(save_path, 'w') as w:
            json.dump(evaluate_result, w, indent=4)


@click.command()
@click.option("--qasper_filepath", type=str, required=True)
@click.option("--save_path", type=str, required=True)
@click.option("--chroma_dir", type=str, required=True, default="./Chroma")
def main(qasper_filepath, save_path, chroma_dir):
    evaluate_instance = MakeEvaluateFile(qasper_filepath=qasper_filepath, chroma_dir=chroma_dir)
    evaluate_instance.make_evaluator_file(save_path=save_path)


if __name__ == "__main__":
    main()
