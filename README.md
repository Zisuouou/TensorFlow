# AI Vision Training GUI

본 GUI는 산업용 객체 탐지 모델 학습 업무에서 반복되는 전처리 및 학습 준비 과정을 줄이기 위해 제작되었습니다.

특히 Pascal VOC XML 기반 데이터셋을 사용하는 환경에서 이미지 증강 후 Bounding Box 좌표를 자동으로 보정하고, TensorFlow 학습 배치파이로가 연동하여 학습 진행률과 Loss를 GUI에서 확인할 수 있도록 구성했습니다.
## 주요 기능

- 이미지/XML 파일 검증
- BMP 이미지 JPG 변환
- Pascal VOC XML 클래스명 일괄 변경
- label_map.pbtxt 자동 생성
- TensorFlow config num_classes 수정
- 이미지 증강 기능
  - Horizontal Flip
  - Vertical Flip
  - Horizontal Shift
  - Vertical Shift
  - Rotation
  - CLAHE
  - Row by Column
- 증강 이미지에 맞춘 Bounding Box 좌표 보정
- TensorFlow 학습 실행
- 학습 진행률 ProgressBar 표시
- 실시간 Loss 표시
- Checkpoint 기반 모델 추출

## 사용 기술

- Python
- PyQt6
- TensorFlow
- TensorFlow Object Detection API
- OpenCV
- Pillow
- NumPy
- Pascal VOC XML

## 실행 방법

```bash
python SVAIS.py
