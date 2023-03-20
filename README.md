# defective-product-detection

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/96af0927-53c7-4da4-aceb-af82e55f2c27/Untitled.png)

# Overview

Rod Assembly라 불리는 자동차 부품이 있습니다. 이러한 부품을 생산하는 공장에서는 최대한 빨리 많은 부품을 만들어 내는것도 중요하지만, 만들어낸 부품에 불량은 없는지 검수를 하는 과정 또한 매우 중요합니다. **이번 프로젝트는 이러한 검수과정을 사람이 아닌, 카메라와 컴퓨터가 대신 하여 자동화 시스템을 구축하는 것입니다.**

전체적인 프로젝트 정보는 [해당 블로그](https://www.notion.so/Rimo-tistory-ca56eccdd3b84d88a20097e4ae6cf12a)에서 자세하게 확인하실 수 있습니다.
[https://rimo.tistory.com/category/DeepLearning/부품 불량 검출](https://rimo.tistory.com/category/DeepLearning/%EB%B6%80%ED%92%88%20%EB%B6%88%EB%9F%89%20%EA%B2%80%EC%B6%9C)

# Dataset

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/039d9891-f4e7-4dee-a91e-4ace16896568/Untitled.png)

- 정상 : 부품에 아무런 하자가 없는 양품
- 찍힘 : 공정 이동 과정에서 낙하로 인해 널링에 스크래치이 발생하는 형태
- 밀림 : 공정 과정중 기계에 의해 특정 부분이 밀린 형태
- 이중선 : 전조 과정에서 베어링 이상으로 인해 널링에 이중선이 발생하는 형태
- 미압입 : 리벳이 널링에 완벽하게 끼워지지 않은 형태

# Problems

- 일관되지 않은 Bounding Box 라벨링
- 불량 경계의 모호함
- 제조 현장의 데이터 불균형

# Classification

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fc10d13f-f156-449d-9fb2-d720093e8d45/Untitled.png)

널링 내의 요소들은 일관되지 않은 라벨링과 불량 경계의 모호함으로 인해 Object Detection 대신 Classification으로 문제를 해결하였습니다.

# CAM Dashboard

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0fc4d268-0102-4c56-ba0e-a1338f3830ec/Untitled.png)

부품 불량의 위치정보도 함께 제공해주기 위해 Classification모델에 GradCAM기법을 적용 하였으며 히트맵으로 불량의 위치를 표현하고자 했습니다. 해당 히트맵 시각화를 최적화 하기위해 여러 실험들과 비교를 진행하였으며, 이 과정에서 효율성을 극대화 하기 위해 Streamlit으로 CAM 대시보드를 개발 하였습니다. 이로써 아래와 같이 여러 의미있는 최적화를 진행할 수 있었습니다.

1. Augmentation을 통한 CAM 노이즈 제거

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0765d295-d296-4de5-8d2b-78f2ada9d815/Untitled.png)

1. 전처리를 통한 이슈 해결

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1ff613ea-dedb-478d-95af-41afe2b111b0/Untitled.png)

1. CAM으로 부터 Bounding Box를 얻자

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/582cd61c-ba4a-4eba-bc17-7bb1270e3309/Untitled.png)

# Anomaly Detection

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/248a4a8b-1802-40bd-af8c-86f9c38c6c1c/Untitled.png)

위와 같이 미압입에 대한 불량 유형은 정상 데이터에 비해 무려 4배나 적은 데이터를 가지고 있었습니다. 제조 현장에서 흔히 발견할 수 있는 데이터 불균형 현상입니다. 이러한 문제를 해결하기 위해 정상 데이터만을 학습하는 비지도학습방법론으로 이상치탐지를 수행하였습니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b2923d1f-ba22-4eb6-8196-09eccf7a3a8b/Untitled.png)

# Train

```python
python classification/train.py --model {model} --wandb {True/False} --name {train_name} --aug {aug_name} ;
```

```python
streamlit run classification/cam_dashboard.py
```
