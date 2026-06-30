# TensorFlow GUI
TensorFlow Object Detection API 기반 Faster R-CNN 모델 학습을 GUI로 실행하기 위한 파일입니다.

## 주요 파일
### SVAIS.py
학습을 실행하기 위한 GUI 메인 파일입니다.

주요 기능은 다음과 같습니다.

- TensorFlow Object Detection API 학습 실행
- Faster R-CNN ResNet50 모델 학습 관리
- 학습 데이터 및 설정 파일 경로 지정
- GPU Boost / Normal 모드 실행
- TensorBoard 실행
- 학습 종료 후 체크포인트 및 결과 확인

### 학습에 필요한 파일 
TensorFlow Object Detection API 학습에 필요한 주요 실행 파일과 라이브러리를 포함합니다.

- 'model_main_tf2.py'
- 'model_main_tf2_FRCNN_res50.py'
- 'model_lib_v2.py'

## 목적

비전 검사 AI 모델 학습을 GUI 환경에서 쉽게 실행하고, 학습 과정과 결과를 관리하기 위한 프로젝트입니다.
