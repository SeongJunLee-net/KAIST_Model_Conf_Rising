# 📢 Motivation 
hallucination을 일으키는 모델 구성요소에 대한 분석을 주제로 연구를 시작했습니다. 
따라서, 관련 논문을 탐독을 우선적으로 진행후, 처음에는 hallucination prompt를 세팅하고 관련 가설을 세운 후,
Indirect Effect를 측정했지만 가설 오류로 IE를 해석하는데 어려움을 겪고 주제를 바꿔 진행하게 됐습니다.
LLM의 Confidence와 LLM 정확도 사이에 Correlation이 있기 때문에 모델의 Confidence의 영향을 미치는
Component를 Editing하는 방향으로 연구를 진행했습니다.

# 📢 Workflow
현재까지 설계한 Workflow는 다음과 같습니다.
1. Subject Token을 Entity로 대체함으로써 문장에 Ambiguity를 추가한다.
2. Clear 문장과 Corrupted 문장 사이에 Total Effect(TE)를 각 Sample별로 측정한다.
A. 이후 TE가 양수인 것을 분석한다.
이때, 저희가 의도한 바는 Ambiguity가 적절하게 추가된 상황이 ideal한 상황이기
때문에, TE가 Negative인 경우는 다루지 않습니다.
3. Entropy, Max prob. difference를 사용하여, Indirect Effect(IE)를 각 Sample별로 측정한다.
A. 이때, IE>0이면 Model의 Confidence를 올리는 방향
B. IE<0이면 Model의 Confidence를 낮추는 방향으로 해석한다.
4. 3에 따라서 IE를 정규화 과정이후 Sample마다의 방향벡터를 얻는다. Clear Input sentence
가 있을때, 이를 subject token 위치와 last token위치 layer의 representation에 방향벡터
를 더하여 Editing을 진행한 후 Confidence의 변화를 측정한다.
5. 4와 마찬가지로 Model의 Accuracy의 변화를 측정한다.

# 📢 Details
- 결과에 대한 정보는 Report에 있습니다.
- 모든 코드는 src/run_all.py에서 실행할 수 있지만, 데이터셋은 그 크기때문에, 깃허브에 올리지 않았습니다.
- 어떤 질문이든지 좋습니다.
