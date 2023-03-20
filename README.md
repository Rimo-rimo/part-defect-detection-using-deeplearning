# defective-product-detection
<img width="696" alt="image" src="https://user-images.githubusercontent.com/79796061/226343870-45a8e10e-76da-4199-936a-884c9a1abdf5.png">

# Overview

Rod Assembly라 불리는 자동차 부품이 있습니다. 이러한 부품을 생산하는 공장에서는 최대한 빨리 많은 부품을 만들어 내는것도 중요하지만, 만들어낸 부품에 불량은 없는지 검수를 하는 과정 또한 매우 중요합니다. **이번 프로젝트는 이러한 검수과정을 사람이 아닌, 카메라와 컴퓨터가 대신 하여 자동화 시스템을 구축하는 것입니다.**

전체적인 프로젝트 정보는 [해당 블로그](https://rimo.tistory.com/category/DeepLearning/%EB%B6%80%ED%92%88%20%EB%B6%88%EB%9F%89%20%EA%B2%80%EC%B6%9C)에서 자세하게 확인하실 수 있으며, 아래의 각 항목마다 관련된 링크를 첨부하였습니다.

# Dataset

<img width="778" alt="image" src="https://user-images.githubusercontent.com/79796061/226343954-e2604d65-ab62-463b-9550-fd61d26110fe.png">

- 정상 : 부품에 아무런 하자가 없는 양품
- 찍힘 : 공정 이동 과정에서 낙하로 인해 널링에 스크래치이 발생하는 형태
- 밀림 : 공정 과정중 기계에 의해 특정 부분이 밀린 형태
- 이중선 : 전조 과정에서 베어링 이상으로 인해 널링에 이중선이 발생하는 형태
- 미압입 : 리벳이 널링에 완벽하게 끼워지지 않은 형태

📔 [데이터 살펴보기](https://rimo.tistory.com/entry/TASK-%ED%95%B4%EA%B2%B0%ED%95%B4%EC%95%BC%ED%95%A0-%EB%AC%B8%EC%A0%9C)

# Problems

- 일관되지 않은 Bounding Box 라벨링
- 불량 경계의 모호함
- 제조 현장의 데이터 불균형

📔 [데이터의 문제점](https://rimo.tistory.com/entry/%EB%B6%80%ED%92%88-%EB%B6%88%EB%9F%89-%EA%B2%80%EC%B6%9C-DataCentric-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%9D%98-%EB%AC%B8%EC%A0%9C%EC%A0%90)

# Classification

<img width="676" alt="image" src="https://user-images.githubusercontent.com/79796061/226344033-c1c140f6-6231-442a-ab27-dae789a2c036.png">

널링 내의 요소들은 일관되지 않은 라벨링과 불량 경계의 모호함으로 인해 Object Detection 대신 Classification으로 문제를 해결하였습니다.

📔 [Classification & CAM](https://rimo.tistory.com/entry/Model-Centric-Classification-CAM)

# CAM Dashboard

<img width="785" alt="image" src="https://user-images.githubusercontent.com/79796061/226344075-47d98e71-5162-438c-a65b-7ae1b1021100.png">

공장측에 부품 불량의 위치정보도 함께 제공해 주기 위해, Classification 모델에 GradCAM 기법을 적용하였으며 히트맵으로 불량의 위치를 표현하고자 했습니다. 해당 히트맵 시각화를 최적화하기 위해 여러 실험들과 비교를 진행하였으며, 이 과정에서 효율성을 극대화 하고자 Streamlit으로 CAM 대시보드를 개발하였습니다. 이로써 아래와 같이 여러 의미 있는 최적화를 진행할 수 있었습니다.

1. Augmentation을 통한 CAM 노이즈 제거

<img width="772" alt="image" src="https://user-images.githubusercontent.com/79796061/226344174-033ffeb0-d469-4870-80ba-a025a48a7e35.png">

2. 전처리를 통한 이슈 해결

<img width="779" alt="image" src="https://user-images.githubusercontent.com/79796061/226344264-787ea402-0be1-4d0a-b06e-b70c71623104.png">

3. CAM으로 부터 Bounding Box를 얻자

<img width="774" alt="image" src="https://user-images.githubusercontent.com/79796061/226344323-55b15363-64cb-453b-b57b-37d3640fa49d.png">

📔 [CAM 대시보드 개발기](https://rimo.tistory.com/entry/4Model-Centric-CAM-%EB%8C%80%EC%8B%9C%EB%B3%B4%EB%93%9C-%EA%B0%9C%EB%B0%9C-feat-streamlit)

# Anomaly Detection

위와 같이 미압입에 대한 불량 유형은 정상 데이터에 비해 무려 4배나 적은 데이터를 가지고 있었습니다. 제조 현장에서 흔히 발견할 수 있는 데이터 불균형 현상입니다. 이러한 문제를 해결하기 위해 정상 데이터만을 학습하는 비지도 학습방법론으로 이상치 탐지를 수행하였습니다.
<img width="700" alt="image" src="https://user-images.githubusercontent.com/79796061/226344376-7ccef030-bfe0-4637-9092-a6286fb0241f.png">
<img width="738" alt="image" src="https://user-images.githubusercontent.com/79796061/226344409-2d423639-d089-4f78-9263-9f55c79073ff.png">


📔 [Anomaly Detection (feat.VAE)](https://rimo.tistory.com/entry/5-Model-Centric-Anomaly-Detectionfeat-VAE)

# Train

```python
python classification/train.py --model {model} --wandb {True/False} --name {train_name} --aug {aug_name} ;
```

```python
streamlit run classification/cam_dashboard.py
```
