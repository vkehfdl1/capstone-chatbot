import os
from operator import itemgetter
from typing import List

import gradio as gr
from RAGchain.preprocess.text_splitter import RecursiveTextSplitter
from RAGchain.reranker import MonoT5Reranker
from RAGchain.schema import Passage, RAGchainPromptTemplate
from dotenv import load_dotenv
from langchain.document_loaders import PDFMinerLoader
from langchain.llms.openai import OpenAI
from langchain.llms.vllm import VLLMOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

text_splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
reranker = MonoT5Reranker()
model = VLLMOpenAI(model_name="meta-llama/Llama-2-7b-hf",
                   openai_api_base=os.getenv("VLLM_API_BASE"),
                   openai_api_key="EMPTY")
# model = OpenAI()
prompt = RAGchainPromptTemplate.from_template("""
    Answer user’s question about NLP paper using given paper passages.
    
    Question: {question}
    
    Paper passages:
    {passages}
    
    Answer:
""")
top_k = 5


def ingest(files) -> tuple[List[Passage], str]:
    loader = PDFMinerLoader(files[0])
    document = loader.load()[0]
    split_passages = text_splitter.split_document(document)
    return split_passages, "성공적으로 파일을 업로드했습니다."


def make_answer(history, reranked_passages: List[Passage]):
    user_query = history[-1][0]
    runnable = RunnablePassthrough.assign(
        passages=itemgetter("passages") | RunnableLambda(lambda passage: Passage.make_prompts(passage))
    ) | prompt | model | StrOutputParser()
    for s in runnable.stream({"question": user_query, "passages": reranked_passages}):
        if history[-1][1] is None:
            history[-1][1] = s
        else:
            history[-1][1] += s
        yield history
    return history


def retrieve(history, _corpus: List[Passage]) -> tuple[List[Passage], str]:
    user_query = history[-1][0]
    reranked_passages = reranker.rerank(user_query, _corpus)[:top_k]
    content = "\n".join([f"## {i+1}번째 참고 논문 paragraph\n{doc.content}" for i, doc in enumerate(reranked_passages)])
    return reranked_passages, content


def user(user_message, history):
    return "", history + [[user_message, None]]


with gr.Blocks() as demo:
    gr.HTML(
        f"""<div style="text-align: center; max-width: 500px; margin: 0 auto;">
                <div>
                <h1>데이터분석캡스톤디자인 챗봇 데모</h1>
                </div>
            </div>"""
    )

    with gr.Row():
        with gr.Column(scale=6):
            passage_markdown = gr.Markdown("# 여기에 검색된 논문 내용이 표시됩니다.\n### 아래에서 pdf 파일을 업로드하세요.")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=700)
            msg = gr.Textbox(label="질문", placeholder="질문을 입력하세요.")

    gr.HTML(
        """<h2 style="text-align: center;"><br>파일 업로드하기<br></h2>"""
    )
    corpus = gr.State(value=[])
    retrieve_passages = gr.State(value=[])
    upload_files = gr.Files()
    ingest_status = gr.Textbox(value="", label="업로드 상태")
    ingest_button = gr.Button("업로드")
    ingest_button.click(ingest, inputs=[upload_files], outputs=[corpus, ingest_status])

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        retrieve, inputs=[chatbot, corpus], outputs=[retrieve_passages, passage_markdown]
    ).then(
        make_answer, inputs=[chatbot, retrieve_passages], outputs=[chatbot]
    )

demo.launch(share=False, debug=False, server_name="0.0.0.0")
