# 데이터분석캡스톤디자인

프로젝트를 진행하면서 지속적으로 코드가 업데이트 되있기 때문에, 정돈이 되어있지 않은 상태인 점 양해 부탁드립니다. 

프로젝트 초반에는 외부 라이브러리를 잘 사용하지 않았으나, 이후 [Langchain](https://github.com/langchain-ai/langchain) 및 [RAGchain](https://github.com/NomaDamas/RAGchain) 라이브러리를 적극적으로 사용하여 구현하였습니다.

대부분의 기능은 위의 두 라이브러리를 통해 구현 가능하나, Rare F1 메트릭의 경우 라이브러리를 사용하지 않고 [scratch.ipynb](./scratch.ipynb) 파일에 구현하였습니다.

## 프로토타입 실행

프로토타입은 [app.py](./app.py) 파일을 실행하여 확인할 수 있습니다.

```bash
gradio app.py
```

초기 설정으로는 vLLM openai api 서버를 열어 작동해야 하나, app.py 파일 내에서 모델 부분을 `OpenAI`로 변경하면 OpenAI 모델을 바로 적용할 수 있습니다.

## 벤치마크

벤치마크는 RAGchain의 [`QasperEvaluator`](https://nomadamas.gitbook.io/ragchain-docs/ragchain-structure/benchmark/dataset-evaluator/qasper)를 사용합니다. metrics 파라미터에 BLEU, KF1, Recall, Precision, F1_score를 포함합니다.
Rare F1과 Token F1 점수는 scratch.ipynb 파일에서, `QasperEvaluator`를 통해 나온 판다스 데이터프레임을 불러와 계산했습니다.
