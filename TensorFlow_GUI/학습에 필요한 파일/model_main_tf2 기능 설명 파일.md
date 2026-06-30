**model_main_tf.py는 model_lib_v2.py를 직접 실행시키는 학습 실행 진입 파일**

**이 파일에는 직접 정의된 함수가 main() 하나 뿐, 나머지는 실행 옵션(FLAGS)과 GPU 초기 설정. 즉, Loss 계산&EarlyStopping은 model_lib_v2.py가 담당하고, *이 파일은 어떤 설정으로 그 엔진을 호출*할건지 결정**

| 파일 | 역할 |
| -- | -- |
| AI_TRAIN_GUI.py | 사용자가 누르는 GUI 화면 |
| model_main_tf2.py | 학습 설정을 정하고 학습 엔진을 호출하는 실행 파일 |
| model_lib_v2.py | 실제 Loss계산, 학습 반복, EarlyStopping, checkpoint 저장 수행 |

- 실행 흐름 :
- SVAIS.py
 - └─ 훈련 시작 버튼 클릭
     - └─ 1)train_FRCNN_res50.bat 실행
         - └─ model_main_tf2.py 실행
             - └─ model_lib_v2.train_loop() 호출
                 - └─ 실제 TensorFlow 학습 진행

- 이 파일은 직접 정의된 함수가 main() 하나뿐이고, 나머지는 실행 옵션과 GPU 환경 설정으로 구성되어 있음


# 1. 이 파일의 전체 역할
- model_main_tf2.py는 TensorFlow Object Detection API에서 학습을 시작할 때 실행하는 진입 파일

- 담당하는 일:

| 기능 | 설명 |
| -- | -- |
| GPU 지정 | 사용할 GPU 번호를 지정 |
| GPU 메모리 설정 | GPU 메모리 사용 비율 설정 |
| 실행 옵션 정의 | config 경로, 결과 폴더, 학습 step, checkpoint 주기 등 |
| train step 계산 | EpochNo.txt 를 읽어 최종 학습 step 결정 |
| 실행 모드 결정 | 학습 모드인지 평가 모드인지 구분 |
| 분산 전략 결정 | TPU, Multi GPU, 단일 GPU 전략 선택 |
| 학습 엔진 호출 | model_lib_v2.train_loop() 실행 |
| 평가 엔진 호출 | model_lib_v2.eval_continuously() 실행 |

- 반대로 이 파일에서 직접 하지 않는 일:

| 기능 | 실제 담당 파일 |
| -- | -- |
| Loss 계산 | model_lib_v2.py |
| Gradient 계산 | model_lib_v2.py |
| EarlyStopping | model_lib_v2.py |
| TensorBoard Loss 기록 | model_lib_v2.py |
| ProgressBar 표시 | training_gui.py |
| 모델 추출 | training_gui.py & 2)model_pb.bat |


# 2. 상단 주석 부분
- TensorFlow Object Detection API 원본 파일에서 제공하는 실행 방법 안내 

- 내 환경에서는 
    - SVAIS.py → train 배치파일 → model_main_tf2.py 형태로 실행되는 구조 


# 3. 환경변수 설정 부분 
### TF_CPP_MIN_LOG_LEVEL
- TensorFlow가 콘솔에 출력하는 C++ 로그를 줄이는 설정

- 값의 의미:

| 값 | 의미 |
| -- | -- |
| 0 | 모든 로그 표시 |
| 1 | INFO 로그 숨김 |
| 2 | INFO, WARNING 숨김 |
| 3 | INFO, WARNING, ERROR 대부분 숨김 |

- 장점
    - 콘솔이 깔끔해짐

- 단점 
    - CUDA나 GPU 관련 오류 원인을 확인할 때 중요한 로그까지 안 보일 수 있음

### CUDA_VISIBLE_DEVICES
- 사용할 GPU를 지정하는 부분

| 설정 | 사용 GPU |
| -- | -- |
| 0 | 첫번째 GPU |
| 1 | 두번째 GPU |
| 0,1 | 첫번째와 두번째 GPU |
| -1 | GPU 사용 안 하고 CPU만 사용 |


# 4. TensorFlow 및 model_lib_v2 import
- 이 파일은 TensorFlow 2 방식으로 설정
- import를 통해 실제 학습 엔진 파일을 불러옴

- 즉, 이 파일 안에서 직접 학습 반복문을 돌리는 게 아닌, model_lib_v2.train_loop(...) 를 호출해서 학습을 맡기는 구조


# 5. 실행 옵션 FLAGS 
- 이 파일에서는 absl.flags를 이용해서 실행 옵션을 정의

### pipeline_config_path
- 학습에 사용할 pipeline config 파일 경로

- 현재 기본값은:
    - configs/faster_rcnn_resnet50_v1_800x1333_batch1.config

- config 파일 안에는 보통 아래의 정보가 있음:
    - 사용할 모델 구조: Faster R-CNN ResNet50
    - 클래스 개수
    - 학습 batch size
    - learning rate
    - fine-tune checkpoint 경로
    - TFRecord 경로
    - label map 경로
    - 기본 num_steps

- 즉, 모델 학습 설정의 중심 파일

### num_train_steps
- 전체 학습 step 수를 받는 옵션
- 내 코드에선 나중에 EpochNo.txt 값을 읽어서 이 값을 덮어 씀

- 따라서 실제로 배치파일에서 전달한 값보다 
    - D:\AI_SVT_Training_mk\annotations\EpochNo.txt 에 저장된 값이 최종 학습 step 으로 사용

### eval_on_train_data
- 학습 데이터 자체를 평가 대상으로 사용할지 여부를 나타내는 옵션
- 정의는 되어 있지만 현재 커스텀 실행 흐름에서는 실질적으로 사용되지 않는 옵션에 가까움

### sample_1_of_n_eval_examples
- 평가 데이터를 몇 개 간격으로 샘플링할지 설정하는 옵션
- 예를 들어:
    - 1 = 모든 평가 이미지 사용
    - 5 = 5개 중 1개만 평가에 사용

- 이 값은 학습 모드가 아니라 checkpoint_dir이 지정된 평가 모드에서 사용

### sample_1_of_n_eval_on_train_examples
- 학습 데이터를 평가에 사용할 경우, 몇 장 중 한 장을 샘플링 할 지 설정하는 값
    - 기본값 5

### model_dir
- 학습 결과를 저장할 폴더 
- 현재 기본값은:
    - train_result

- 이 폴더 안에 보통 다음 파일들 생성:
    - train_result
        - checkpoint
        - ckpt-xxx.data-00000-of-00001
        - ckpt-xxx.index
        - train 
            - event.out.tfevents...

- 즉,

| 결과물 | 저장 위치 |
| -- | -- |
| checkpoint | train_result |
| TensorBoard event 파일 | train_result/train |

### checkpoint_dir
- 평가 전용 모드를 실행할 때 사용하는 checkpoint 폴더

- 이 값이 없으면 
    - 학습모드

- 이 값이 있으면
    - 평가모드

### eval_timeout
- 평가 모드에서 새로운 checkpoint가 생기기를 기다리는 최대 시간
- 현재 설정:
    - 3600초 = 1시간

- 새 checkpoint가 한 시간 동안 생성되지 않으면 평가 실행 종료

### use_tpu
- TPU를 사용할지 설정하는 값

- 내 환경에서는 Windows + NVIDIA GPU 기반으로 학습하고 있으니까 일반적으로 False 상태로 사용

### tpu_name
- Cloud TPU 를 사용할 때 TPU 이름을 넣는 옵션으로
    - 현재 작업 환경에서는 사실상 사용하지 않는 부분

### num_workers
- 분산학습에서 worker 수를 정하는 값

| 값 | 실행 방식 |
| -- | -- |
| 1 | MirroredStrategy | 
| 2 이상 | MultiWorkerMirroredStrategy |

- 현재 기본값은 1으므로 일반적으로 한 대의 PC안에서 GPU를 사용하는 구조

### checkpoint_every_n
- 몇 step 마다 checkpoint를 저장할지 정하는 값

- 현재는:
    - 200step 마다 checkpoint 기준 계산으로 설정

- 이 부분이 ai_training_gui.py 에서 모델 추출 시 사용자가 입력한 step 값을 200으로 나누는 로직과 연결

### record_summaries
- TensorBoard summary 기록 여부를 정하는 옵션
- True 면 학습 이미지나 모델 내부 summary 가 기록될 수 있음

- 단 , 주석 설명처럼 loss 값은 이 옵션과 관계없이 기록되는 구조


# 6. GPU 메모리 설정 부분
- config = tf.compat.v1.ConfigProto(
    - gpu_options=tf.compat.v1.GPUOptions(
        - per_process_gpu_memory_fraction=0.5
    - )
- )
- session = tf.compat.v1.Session(config=config)
- tf.compat.v1.keras.backend.set_session(session)

- 이 부분은 내 코드에서 별도로 추가된 GPU 메모리 제한

## 역할 
- per_process_gpu_memory_fraction=0.5
    - 는 TensorFlow 프로세스가 GPU 메모리의 약 50%까지만 사용하도록 제한하는 설정

    - 예를들어 RTX 3090이 24GB라면 대략:
        - 24GB x 0.5 = 약 12GB 정도 범위에서 사용하도록 제한하려는 목적

## 넣은 이유
- 다음과 같은 목적일 가능성이 큼:
    - GPU 메모리 전체 점유 방지
    - 다른 프로그램과 GPU 공유
    - 메모리 부족 오류 예방
    - GUI와 TensorBoard 등 동시에 사용하는 환경 고려

## 주의할 점
- Faster R-CNN은 이미지 크기와 batch size에 따라 GPU 메모리를 많이 사용할 수 있음

- 내 설정처럼:
    - Faster R-CNN ResNet50
    - 800 × 1333 이미지를 사용하는 경우, GPU 메모리 50% 제한 때문에 OOM 오류가 발생할 수도 있음

- allow_growth = True는 필요한 만큼만 GPU 메모리를 점진적으로 사용하는 설정 
    - 메모리 충돌이 생기면 아래 방식 검토 가능 
        - 지금 코드에선 주석처리 되어있음 _26.05.27

    
# 7. main(unused_argv) 함수 
- 이 파일에서 직접 정의된 유일한 함수이자, 실제 실행 흐름을 담당하는 핵심 함수 


# 8. 필수 flag 등록
- model_dir 과 pipeline_config_path 가 반드시 필요하다는 의미
- 현재 두 옵션 모두 기본값이 지정돼 있음
    - 그래서 기본 구조가 그대로 존재하면 실행 가능하도록 작성되어 있음


# 9. Soft Device Placement 설정
- TensorFlow 가 특정 연산을 GPU 에서 처리할 수 없을 경우, 자동으로 CPU 에 배치할 수 있도록 허용하는 설정

- 예를 들어:
    - GPU에서 지원되지 않는 연산 발견
        - -> 강제로 오류 발생시키지 않고 CPU 에서 처리

    - 이런식으로 동작

- 장점은
    - 실행 안정성이 좋아진다는 것,

- 단점은
    - 일부 연산이 CPU로 넘어가면서 속도가 느려질 수 있다는 점


# 10. 이미지 개수 계산
- 학습 데이터가 들어있는 annos 폴더 경로를 지정

- 그리고 폴더 안 이미지 개수를 셈
    - 지원하는 확장자:
        - .jpg
        - .jpeg
        - .png
        - .bmp
    
    - 이 이미지 개수는 EpochNo.txt 파일을 읽지 못했을 때 기본 train step 계산에 사용됨 


# 11. EpochNo.txt 읽기
- 학습 step 을 결정하기 위해 다음 파일을 읽음
- ai_training_gui.py 에서 사용자가 x10, x20, x30 을 선택하면 이 파일에 값이 저장 됐음


# 12. EpochNo.txt 읽기 실패 시 처리 
- 다음 상황이면 파일 읽기에 실패할 수 있음
    - EpochNo.txt 파일이 없음
    - 파일 안이 빈 값
    - 숫자가 아닌 문자가 들어 있음
    - 파일 접근 권한 문제 

- 그 경우 기본값으로 
    - 이미지 개수 x 20


# 13. 최소 / 최대 학습 step 제한
- 계산된 step이 너무 작거나 너무 커지지 않게 제한하는 부분

| 계산 결과 | 실제 적용 step |
| -- | -- |
| 4,000 | 10,000 |
| 38,340 | 38400 |
| 500,000 | 300,000 |

- 즉, 
    - 최소 10,000step, 최대 300,000step 으로 제한

    
# 14. FLAGS.num_train_steps 덮어쓰기
- 앞에서 계산한 최종 step 값을 실행 옵션에 다시 저장
- 원래 배치파일이나 config 에서 다른 값이 들어왔더라도, 이 코드가 실행되면 최종적으로 EpochNo.txt 기반 step 값이 적용


# 15. model_lib_v2.py 와 train step 계산이 중복

- 현재 model_main_tf2.py에서도 EpochNo.txt를 읽어서 train_steps를 정하고, model_lib_v2.py안의 train_loop()에서도 다시 EpochNo.txt를 읽어서 train)steps를 정함

- 즉 현재 구조:
- model_main_tf2.py
 - └─ EpochNo.txt 읽기
     - └─ FLAGS.num_train_steps 설정
         - └─ model_lib_v2.train_loop()
             - └─ EpochNo.txt 다시 읽기
                 - └─ train_steps 다시 설정

- model_main_tf2.py와 model_lib_v2.py 모두 현재 EpochNo.txt 기반 step 보정 로직 포함


# 16. 평가 모드 분기
- checkpoint_dir 값이 존재하면 학습을 하지 않고 평가만 수행

- 즉: 
    - checkpoint_dir 있음 -> 평가 모드
    - checkpoint_dir 없음 -> 학습 모드

## 평가 모드에서 하는 일
- 1. checkpoint 폴더 감시
- 2. 새로운 checkpoint 발견
- 3. 모델에 checkpoint 복원
- 4. eval dataset으로 추론
- 5. mAP 등 평가 metric 계산
- 6. TensorBoard 에 기록 

### wait_interval = 10
- 새 checkpoint가 생겼는지 10초 간격으로 확인한다는 의미

### timeout=FLAGS.eval_timeout
- 기본 값이 3600초로:
    - 1시간 동안 새 checkpoint가 없으면 평가 종료


# 17. 학습 모드 분기
- checkpoint_dir이 없으면 실제 학습을 진행
- 학습 전에 어떤 장치 전략으로 돌릴지 선택


# 18. TPU 학습 전략
- TPU를 사용하는 경우의 로직
    - 난 엔비디아 GPU 에서 작업해서 실행 안 됨


# 19. MultiWorker 학습 전략
- 여러 대의 컴퓨터 또는 여러 worker를 사용해 분산 학습하는 경우에 사용


# 20. 일반 GPU 학습 전략
- 내가 평소 사용하는 Window + NVIDIA GPU 환경에서 실행 

### MirroredStrategy
- 여러 GPU가 있으면 각 GPU에 동일한 모델을 복제해서 학습하는 방식
- GPU가 한 개여도 TensorFlow Object Detection API 구조상 MirroredStrategy 로 실행할 수 있음

### HierarchicalCopyAllReduce
- 여러 GPU 사이에서 gradient 값을 합치는 방식 중 하나
- 현재 PC에서 GPU가 하나뿐이면 큰 체감은 없지만, 코드상으로 멀티 GPU 대응이 가능하도록 작성


# 21. strategy.scope()
- 선택한 GPU/분산 전략 안에서 모델 학습을 실행하겠다는 의미

- 이 영역 안에서 생성되는:
    - 모델
    - optimizer
    - checkpoint
    - 변수
- 등이 해당 strategy 기준으로 관리


#  22. model_lib_v2.train_loop() 호출
- 여기가 이 파일에서 가장 중요한 실행 구간으로
    - model_main_tf2.py가 결정한 설정을 model_lib_v2.py로 넘김

| 전달값 | 의미 |
| -- | -- |
| pipeline_config_path | 사용할 Faster R-CNN config 경로 |
| model_dir | checkpoint와 TensorBoard 저장 폴더 |
| train_steps | 최종 학습 step |
| use_tpu | TPU 사용 여부 |
| checkpoint_every_n | checkpoint 저장 기준 step |
| record_summaries | TensorBoard summary 기록 여부 |

- 그리고 실제로 model_lib_v2.train_loop() 안에서:
    - 모델 생성
    - TFRecord 데이터 로드
    - loss 계산
    - gradient 적용
    - TensorBoard 기록
    - checkpoint 저장
    - EarlyStopping 이 진행


# 23. 마지막 checkpoint 저장 코드
- 의도는 학습이 정상 종료된 뒤 마지막 checkpoint를 한 번 더 저장하려는 것

## 현재 구조에서는 이 부분이 실행되지 않을 가능성이 큼
- model_main_tf2.py 안의 최종 checkpoint 저장 로직은 현재 model_lib_v2.py가 manager 를 반환하지 않는다면 실행되지 않음

- 다만 model_lib_v2.py 안에서 이미 checkpoint를 저장하고 있으므로 학습 결과가 무조건 없는 것은 아님
    - 문제는 학습 정상 종료 직후 마지막 checkpoint를 이 파일에서 한 번 더 보장 저장한다는 의도가 실제로는 동작하지 않을 수 있음


# 24. 프로그램 실행 시작점
- tf.compat.v1.app.run()은 내부적으로 main() 함수를 호출

- 즉:
- python model_main_tf2.py 실행
    - tf.compat.v1.app.run()
    - main() 실행
    - 학습 또는 평가 시작  흐름


# 25. AI_TRAINING_GUI 와 연결해서 보면
- 흐름:
- SVAIS GUI에서 훈련 시작 클릭
    -    ↓
- TrainingThread.run()
    -    ↓
- 1)train_FRCNN_res50.bat 실행
    -    ↓
- model_main_tf2.py 실행
    -    ↓
- EpochNo.txt에서 train_steps 확인
    -    ↓
- GPU strategy 설정
    -    ↓
- model_lib_v2.train_loop() 실행
    -    ↓
- TensorFlow 실제 학습
    -    ↓
- Loss / checkpoint / TensorBoard 기록
    -    ↓
- 학습 완료 로그 출력
    -    ↓
- SVAIS GUI가 완료 감지

- 파일별 역할

| 기능 | 담당 파일 |
| -- | -- |
| 버튼 클릭 / ProgressBar 표시 | gui.py 파일 |
| train step 전달 / GPU 전략 선택 | model_main_tf2.py |
| 실제 학습 loop | model_lib_v2.py |
| 모델 export | 2)model_pb.bat 또는 exporter 파일 |


# 26. XLA 버튼과 현재 파일의 연결 여부
- gui.py 에서 학습 실행 시:
    - env["ENABLE_XLA"] = "1" if self.enable_xla else "0"
    - 형태로 XLA ON/OFF 상태를 환경변수로 넘김

- 그런데 현재 model_main_tf2.py 안에는:
    - XLA를 실제 적용하는 코드가 없음


# 27. 이 파일에서 체크하면 좋은 부분

| 위치 | 현재 상태 | 확인할 점 |
| -- | -- | -- |
| TF_CPP_MIN_LOG_LEVEL = '3' | 로그 대부분 숨김 | 오류 분석 시 로그가 부족할 수 있음 |
| CUDA_VISIBLE_DEVICES = '0' | 첫번째 GPU만 사용 | 다중 GPU 활용 시 수정 필요 |
| GPU memory 0.5 | GPU 메모리 50% 제한 | Faster R-CNN에서 OOM 가능성 확인 | 
| allow_growth | 주석 처리 됨 | 메모리 유동 할당이 필요하면 활성화 검토 |
| EpochNo.txt 읽기 | 학습 step 동적 실행 | model_lib_v2.py와 중복 |
| 