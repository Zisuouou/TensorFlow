**이 파일은 TensorFlow Object Detection API의 학습 엔진 쪽 핵심 파일로, 원본 model_lib_v2.py에 ProgressBar용 로그, 동적 train_steps, EarlyStopping, checkpoint 저장, TensorBoard loss 기록을 추가한 버전**

**학습 엔진 쪽에 가까운 파일로, 실제 TensorFlow Object Detection 학습을 돌리는 내부 엔진**

# 1. 이 파일의 전체 역할
- 이 파일은 TensorFlow Object Detection API에서 모델 학습과 평가를 담당하는 핵심 파일

- 주요 역할:

| 구분 | 역할 |
| -- | -- |
| 모델 생성 | config 파일을 읽어서 Faster R-CNN 같은 detection model 생성 |
| 데이터셋 생성 | TFRecord 기반 train/eval dataset 생성 |
| Loss 계산 | 모델 예측값과 정답 라벨을 비교해서 loss 계산 |
| Gradient 계산 | tf.GradientTape() 로 gradient 계산 |
| Optimizer 적용 | 계산된 gradient 로 모델 가중치 업데이트 | 
| Checkpoint 관리 | 학습 중간중간 ckpt 저장 |
| TensorBoard 기록 | loss, learning rate, steps_per_sec 기록 |
| EarlyStopping | loss가 개선되지 않으면 학습 조기 종료 |
| Eval 수행 | checkpoint 기준으로 mAP 등 평가 수행 |


# 2. 상단 import / 전역 설정

### 주요 import
- import tensorflow.compat.v1 as tf
- import tensorflow.compat.v2 as tf2

- 이 파일은 TensorFlow 2.x 기반이지만, Object Detection API 특성상 tf.compat.v1도 같이 사용

- 핵심 모듈:
    - from object_detection import eval_util
    - from object_detection import inputs
    - from object_detection import model_lib
    - from object_detection.builders import optimizer_builder
    - from object_detection.utils import config_util
    - from object_detection.utils import label_map_util

- 각각 역할:

| 모듈 | 역할 |
| -- | -- |
| inputs | train/eval dataset 생성 |
| model_lib | OD API 공통 모델 유틸 |
| optimizer_builder | config에 맞는 optimizer 생성 |
| config_util | pipeline config 읽기/저장 |
| label_map_util | label_map.pbtxt 읽기 |
| eval_util | detection 평가 지표 계산 |
| visualization_utils | 평가 이미지 시각화 |

### NUM_STEPS_PER_ITERATION = 200
- 이 값은 한 번의 outer loop 에서 몇 stpep 씩 학습할지 정하는 값
    - 즉, 학습 로그가 보통 200 step 으로 찍히게 되는 이유

### patience = 100
- EarlyStopping에서 사용하는 대기 횟수
- 현재 코드에서는 조건을 만족한 뒤 loss가 좋아지지 않으면 wait 증가, wait >= 100 이 되면 학습 종료


# 3. _compute_losses_and_predictions_dicts()
- 모델 예측값과 loss를 계산하는 핵심 함수

### 1단계 : 정담 라벨을 모델에 넣음
- 모델에게 현재 batch의 정담 bbox, class 정보를 전달

### 2단계 : 이미지 예측 수행
- 입력 이미지에 대해 모델이 예측을 수행함
- 예측 결과에는 보통 이런 정보가 들어옴
    - RPN 예측
    - box prediction
    - class prediction
    - feature map 관련 값

### 3단계 : Loss 계산
- 예측값과 정답값을 비교해서 loss 계산

### 4단계 : total_loss 생성
- 여러 loss들을 모두 더해서 최종 loss인 Loss/total_loss 를 만듦
- 이 값이 TensorBoard에서 가장 중요하게 보는 전체 loss

**정리하면 이 함수는: 이미지 넣기 -> 모델 예측 -> loss 계산 -> total_loss 반환을 담당**


# 4. _ensure_model_is_built()
- 이 함수는 모델이 실제 학습 전에 한번이라도 실행되도록 강제로 dummy computation을 수행하는 함수

- TensorFlow/Keras 모델은 처음부터 모든 변수가 만들어져 있는 게 아니라, 입력 데이터가 한 번 들어가야 weight들이 생성되는 경우가 있음
    - 그래서 checkpoint를 불러오기 전이나 EMA optimizer 를 쓰기 전에 
        - _ensure_model_is_built(...)를 호출해서 모델 구조와 변수들을 미리 생성

**쉽게 말하면 학습 시작 전에 모델 몸풀기 한 번 시키는 함수**


# 5. normalize_dict()
- 분산 학습에서 loss값을 GPU 개수 기준으로 나눠주는 함수

- 예를 들어 GPU 가 2개면 loss를 2로 나눠서 scale을 맞춤
- 즉:

| 상황 | 역할 |
| -- | -- |
| 단일 GPU | 거의 영향 없음 |
| 멀티 GPU | replica 개수만큼 loss 보정 |


# 6. reduce_dict()
- 분산 학습에서 각 GPU가 계산한 loss를 하나로 합치는 함수

- 예를 들어 GPU 2개가 각각 loss를 계산했다면:
    - GPU0 loss + GPU1 loss 
    - 합쳐서 최종 loss dict로 만듦


# 7. eager_train_step()
- 이 함수는 실제 학습 1step을 수행하는 함수

### 1단계 : 학습 모드 ON
- 모델을 training mode로 바꿈
- Dropout, BatchNorm 같은 레이어가 학습 모드로 동작하게 됨

### 2단계 : batch label 정리
- batch 형태로 들어온 라벨 데이터를 이미지별로 풀어줌
- Object Detection API는 내부적으로 label 구조가 복잡해서, 학습 전 이 처리가 필요함

### 3단계 : GradientTape 시작
- TensorFlow 에서 gradient를 계산하기 위해 사용하는 영역
- 이 안에서 loss를 계산하면, 나중에 그 loss 기준으로 gradient를 구할 수 있음

### 4단계 : loss 계산
- 앞에서 설명한 함수로 total loss를 계산

### 5단계 : gradient 계산
- 현재 loss를 줄이기 위해 모델 가중치를 어떻게 바꿔야 하는지 계산

### 6단계 : gradient clipping
- gradient가 너무 커지는 문제를 막는 기능
- 학습이 불안정하거나 loss가 튈 때 도움이 됨

### 7단계 : optimizer 적용
- 계산된 gradient를 모델 가중치에 적용

**즉, 이 함수의 진짜 의미: 이미지 batch 하나 학습**


# 8. validate_tf_v2_checkpoint_restore_map()
- checkpoint를 불러올 때 restore map 구조가 올바른지 검사하는 함수

- 검사 기준:
    - key가 문자열인지
    - value가 tf.Module 또는 tf.train.Checkpoint인지
    - nested dict 이면 재귀적으로 다시 검사

- 잘못된 구조면 TypeError 발생

**checkpoint 복원 전에 복원 대상 구조가 맞는지 확인하는 함수**


# 9. is_object_based_checkpoint()
- checkpoint가 TensorFlow 2.x 방식의 object-based checkpoint 인지 확인하는 함수

- 확인 방식:
    - _CHECKPOINTABLE_OBJECT_GRAPH가 있으면 TF2 스타일 checkpoint라고 판단


# 10. load_fine_tune_checkpoint()
- pre-trained checkpoint를 불러오는 함수

- 예를 들어 COCO로 사전 학습된 Faster R-CNN checkpoint를 가져와서, 우리 데이터셋으로 fine-tuning할 때 사용

- 흐름:
    - 1. checkpoint가 TF2 object-based checkpoint 인지 확인
    - 2. checkpoint version이 V2인지 확인
    - 3. 필요하면 _ensure_model_is_built() 로 모델 먼저 build
    - 4. 모델에서 restore map 가져오기
    - 5. restore map 구조 검증
    - 6. tf.train.Checkpoint 로 checkpoint 복원

**즉, 이 함수는 사전학습 모델 불러와서 현재 모델에 연결하는 함수**


# 11. get_filepath()
- 분산 학습 환경에서 checkpoint나 summary 저장 경로를 정하는 함수

- 단일 GPU나 일반 학습에서는 대부분 원래 경로를 그대로 씀
- 하지만 MultiWorker 학습에서는 chief worker와 non-chief worker가 있는데, non-chief worker는 임시 폴더를 쓰게 함

- 즉:

| 환경 | 저장 경로 |
| -- | -- |
| chief worker | 원래 경로 |
| non-chief worker | temp_worker_xxx 임시 경로 |


# 12. clean_temporary_directories()
- MultiWorker 학습 후 non-chief worker가 만든 임시 폴더를 삭제하는 함수

- 단일 GPU 환경에서는 크게 신경 안 써도 되는 함수


# 13. train_loop()
- 이 파일에서 가장 중요한 함수
- 실제 전체 학습 흐름ㅇ르 담당

- model_main_tf2.py에서 학습을 실행하면 결국 이 train_loop()가 호출되는 구조

### train_loop() 전체 흐름
1. pipeline config ㅣㄺ기
2. train_steps 결정
3. 모델 config / train config / input config 분리
4. 모델 생성
5. train dataset 생성
6. optimizer 생성
7. fine-tune checkpoint 로드
8. checkpoint manager 생성
9. TensorBoard summary writer 생성
10. 학습 반복
11. loss 기록
12. EarlyStopping 판단
13. checkpoint 저장
14. 학습 완료 메시지 출력

## 13-1. pipeline 읽기
- 예를들어 내가 쓰는 Faster R-CNN config:
    - faster_rcnn_resnet50_v1_800x1333_batch1.config 같은 파일을 읽는 부분

## 13-2. train_steps 결정
- 기본적으로 config 안의 num_steps를 읽음

- 난 커스텀 시켰음
    - 즉, 최종 학습 step을 config만 보는게 아니라
    - D:\AI_SVT_Training_mk\annotations\EpochNo.txt 에서 읽어ㅗ게 되어있음

    - 그리고 최소/최대 제한을 걸어뒀음
        - 최소 10,000 step, 최대 300,000 step

- 최종 구조:
    - EpochNo.txt 값 읽기
    - 실패하면 이미지 개수 x 20
    - 최소 10,000
    - 최대 300,000
    - train_steps 확저

- 이 구조로, 이 값은 콘솔에 [TRAIN_STEPS]로 출력하도록 커스텀

## 13-3. [TOTAL_STEPS], [TRAIN_STEPS] 출력
- 이 로그는 GUI ProgressBar랑 연결하기 위해 추가한 것
- AI_Training_GUI 쪽 TrainingThrea가 콘솔 출력을 읽어서 step이나 loss를 추출하는 구조였기에, 이 파일에서 이런 로그를 찍는게 중요

## 13-4. mixed precision / bfloat16 설정
- TPU나 bfloat16을 쓰는 환겨이면 mixed precision policy를 설저
- 일반 Windows + NVIDIA GPU 환경에서는 보통 이 부분이 핵심이 아님

## 13-5. 모델 설정
- config에 적힌 모델 구조를 기반으로 실제 detection model 객체를 만듦
- 예를들어 config가 Faster R-CNN ResNet50이면, 여기서 Faster R-CNN 모델이 생성

## 13-6. train dataset 생성
- TFRecord와 label map, config 설정을 바탕으로 학습 dataset을 만듦
- repeat()이 들어가 있어서 dataset은 계속 반복해서 공급

## 13-7. global_step 생성
- 현재 학습 step을 저장하는 변수
- TensorBoard x축, checkpoint 저장 기준, 로그 기준이 모두 이 global_step과 연결

## 13-8. optimizer 생성
- config 파일에 설정된 optimizer 를 만듦
- 예를 들어 config에 momentum optimizer, learning rate schedule 등이 있으면 여기서 반영

## 13-9. EMA optimizer 대응
- EMA, 즉 Exponential Moving Average를 사용하는 경우 모델 변수를 미리 만들고 shawdow copy를 생성

## 13-10. TensorBoard summary writer 생성
- TensorBoard 이벤트 파일을 저장하는 writer 만듦
- 보통 저장 위치는:
    - model_dir/train

## 13-11. fine-tune checkpoint 로드
- config 안에 fine_tune_checkpoint 경로가 있으면 pre-trained checkpoint를 불러옴
- 처음부터 랜덤하게 학습하는 게 아니라, COCO 등으로 학습된 모델에서 이어 학습하는 경우 이 부분이 작동

## 13-12. CheckpointManager 생성
- 학습 중 checkpoint를 저장하고 관리하는 객체
- 난 checkpoint_every_n=200으로 해놔서 200step 단위 저장 구조로 맞춤


# 14. train_loop() 내부 함수들
- train_loop() 안에는 중첩 함수가 3개 있음

## 14-1. train_step_fn(features, labels)
- 실제 train step 하나를 실행하는 내부 함수

- 하는 일:
    - 1. TensorBoard에 train image 기록
    - 2. eager_train_step() 호출
    - 3. global_step += 1
    - 4. loss dict 반환

- 즉, 진짜 학습 한 step은 여기서 시작해서 eager_train_step()으로 들어감

## 14-2. _sample_and_train(strategy, train_step_fn, data_iterator)
- dataset에서 batch 하나를 꺼내서 분산 전력으로 학습시키는 함수

- 흐름:
    - 1. data_iterator.next()로 features, labels 가져오기
    - 2. strategy.run()으로 GPU/replica에 학습 step 실행
    - 3. replica별 loss를 reduce_dict()로 합침

- 단일 GPU여도 TensorFlow distribution strategy를 통과하는 구조라 이 함수가 필요

## 14-3. _dict_train_step(data_iterator)
- 분산 학습용 train step 함수

- num_steps_per_iteration = 200 이면 내부적으로 200 step을 돌고 한 번 바끙로 나오는 구조
    - 그래서 GUI 입장에서는 매 step마다 바로바로 출력되는 게 아니라, 보통 200step 단위로 갱신하는 느낌이 남


# 15. 실제 학습 반복문
- for _ in range(global_step.value(), train_steps, num_steps_per_iteration):
    - 여기가 실제 학습 loop

- global_step부터 train_steps까지 num_steps_per_iteration 단위로 반복

## 15-1. 현재 loss 계산
- 200 step 학습을 수행한 뒤, 현재 total loss를 가져옴

## 15-2. TensorBoard에 Loss/total_loss 직접 기록
- 이 부분은 내가 추가한 핵심 커스텀

- 원래 OD API도 loss를 기록하지만, GUI/확인 편의를 위해 Loss/total_loss를 명확히 TensorBoard에 찍도록 한 구조

## 15-3. NaN 감지
- loss가 NaN이 되면 학습이 망가진 상태로 보고 즉시 종료

- NaN 값은 보통 이런 경우에 생길 수 있음:
    - learning rate가 너무 큼
    - bbox 값 이상
    - 이미지/라벨 데이터 문제
    - mixed precision 문제
    - gradient 폭주

## 15-4. step time / steps per sec 계산
- 현재 200step을 도는 데 거린 시간을 계산하고, 초당 step 수를 TensorBoard에 기록

- 즉, TensorBoard에서 학습 속도 확인 가능

## 15-5. learning_rate 기록
- loss dict에 learning rate를 추가해서 TensorBoard에 같이 기록
- 그래서 TensorBoard에서 loss뿐 아니라 learning rate 변화도 볼 수 있음

## 15-6. 콘솔 로그 출력
- 콘솔에:
    -  tensorflow Version:2.x.x | Wait:0 | Step:2000 | step_time:12.345s | avg_time:... 이런 정보 찍음

    - 그리고 loss dict도 출력

- 그래서 GUI 쪽에서 Loss/total_loss 문자열을 파싱할 수 있는 구조


# 16. EarlyStopping 로직
- 이 파일에서 중요한 커스텀 중 하나

- 전체 학습 step의 70% 이후부터 EarlyStopping 판단 시작
- 예를들어 train_steps = 40000 이면, 
    - 40000 x 0.7 = 28000
- 즉, 28,000step 이후부터 loss개선 여부 감시

## EarlyStopping 흐름

| 현재 로직 |  |
| -- | -- |
| 전체 step의 70% 이전 : | EarlyStopping 판단 안 함 |
| 전체 step의 70% 이후 : | best_loss 갱신 시작 |
| best_loss < 0.1인 경우 : | loss가 이전보다 좋아지지 않으면 wait += 1, loss가 좋아지면 wait = 0 |
| wait >= patience: | checkpoint 저장, 완료 메시지 출력, SystemExit으로 종료 |

- 핵심 조건 :
    - if best_loss < 0.1:
        - 즉, total loss가 0.1 아래로 내려간 이후부터 진짜 EarlyStopping 카운트가 의미있게 작동


# 17. checkpoint 저장 로직

- if global_step >= early_start:
    - if gs >= early_start and gs % checkpoint_every_n == 0:
        - cp_idx = gs // checkpoint_every_n
        - manager.save(checkpoint_number = cp_idx)

- 전체 step의 70% 이후부터 checkpoint 저장
- 그리고 step이 checkpoint_every_n 으로 나누어 떨어질 때 저장

- 기본값이 200이라 200 step 단위 저장 구조


# 18. 학습 완료 메시지
- 학습이 정상적으로 끝나거나 EarlyStopping이 걸리면 아래 문구 출력
    - print("It's done. Do NOT need to update Best Loss", flush=True)

    - 이 문구는 GUI 쪽에서 학습 종료를 감지하는 신호로 쓰는 구조

- 즉, 이 파일은 학습 엔진이고, GUI 파일은 이 문구를 콘솔에서 읽어서 "훈련완료" 팝업을 띄우는 쪽


# 19. prepare_eval_dict()
- 이 함수는 평가할 때 모델 예측값과 정답값을 evaluation module에 넣기 좋은 형태로 바꾸는 함수

- 하는 일:
    - 1. groundtruth box 가져오기
    - 2. class agnostic 모델인지 확인
    - 3. class id offset 적용
    - 4. original image가 있으면 원본 이미지 기준으로 평가 준비
    - 5. eval_util.result_dict_for_batched_example()로 eval dict 생성

- 즉, 평가용 데이터 포맷 변환 함수 라고 생각하면 됨


# 20. concat_replica_results()
- 분산 학습/평가에서 replica별 결과를 하나로 합치는 함수

    - 예를들어, GPU별 prediction 결과를 batch 방향으로 합침


# 21. eager_eval_loop()
- 평가 dataset을 돌면서 detection 성능을 계산하는 함수 
- 학습용이 아닌 **평가용 loop** 

- 하는 일:
    - 1. 모델을 eval mode로 변경
    - 2. label map에서 category index 생성
    - 3. eval dataset 반복
    - 4. 모델 예측 수행
    - 5. loss 계산
    - 6. detection 결과 postprocess 
    - 7. eval dict 생성
    - 8. evaluator 에 결과 누적
    - 9. TensorBoard에 평가 이미지 기록
    - 10. mAP 등 eval metric 기록
    - 11. TensorBoard에 metric 기록

- 평가 이미지도 TensorBoard에 기록할 수 이씀
    - detection 결과 이미지를 남김 

    
# 22. eval_continuously()
- checkpoint 폴더를 계속 감시하면서 새 checkpoint가 생기면 평가를 수행하는 함수 

- 흐름:
    - 1. pipeline config 읽기
    - 2. eval config 세팅
    - 3. eval dataset 생성
    - 4. detection model 생성
    - 5. checkpoint iterator 실행
    - 6. 새 checkpoint가 생기면 restore
    - 7. eager_eval_loop() 호출

- 즉, 이 함수는:
    - 학습 중 저장되는 checkpoint를 계속 보면서 자동 평가하는 함수

- 다만, 내 현재 GUI 학습 흐름에서는 주로 train_loop()가 핵심이고, eval_continuously()는 별도 eval 실행 시 사용하는 쪽


# 23. 이 파일에서 내가 커스텀한 포인트
- 아래 부분들이 원본 TensorFlow Object Detection API 에서 내 환경에 맞게 바뀐 핵심

| 커스텀 | 설명 |
| -- | -- |
| NUM_STEPS_PER_ITERATION = 200 | 200 step 단위 학습/로그 기준 |
| patience = 100 | EarlyStopping 대기 횟수 |
| EpochNo.txt 읽기 | GUI에서 설정한 이미지 개수 x 배율 값을 학습 step으로 사용 |
| 최소/최대 step 제한 | 최소 10,000 / 최대 300,000 |
| [TOTAL_STEPS], [TRAIN_STEPS] 출력 | GUI ProgressBar 연동용 |
| Loss/total_loss 직접 summary 기록 | TensorBoard에서 total loss 명확히 보기 위함 | 
| summary_writer.flush() | TensorBoard 기록이 바로 반영되도록 강제 flush |
| NaN 감지 | loss가 NaN이면 학습 중단 |
| step_time / avg_time / total_time 로그 | 학습 속도 확인용 |
| 70% 이후 EarlyStopping 시작 | 초반 불안정한 loss는 무시하고 후반부부터 판단 |
| 70% 이후 checkpoint 저장 | 의미 있는 후반 checkpoint 중심으로 저장 | 
| 완료 문구 출력 | GUI 에서 학습 종료 감지용 |


# 24. AI_TRAINING_GUI와 연결해서 보면
- 두 파일의 관계:

- SVAIS.py
 - └─ 훈련 시작 버튼 클릭
     - └─ 1)train_FRCNN_res50.bat 실행
         - └─ model_main_tf2_*.py 실행
             - └─ model_lib_v2.train_loop() 실행
                 - └─ 실제 TensorFlow 학습 수행

- 그리고 model_lib_v2.py에서 출력하는 로그를 gui 파일이 읽음

- 예:
- [TOTAL_STEPS] 38400
- [TRAIN_STEPS] 38400
- Loss/total_loss
- It's done. Do NOT need to update Best Loss

    - 이런 값들을 GUI에서 ProgressBar, Loss Label, 학습 완료 팝업으로 연결하는 구조


# 25. 한 줄 요약
**이 파일은 TensorFlow Object Detection API 의 실제 학습 루프를 담당하는 파일이고, 내 환경에 맞게 EpochNo.txt 기반, train step 설정, GUI ProgressBar 연동 로그, TensorBoard total loss 기록, NaN 감지, EarlyStopping, checkpoint 저장 로직이 추가된 커스텀 학습 엔진**