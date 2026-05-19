**이 파일의 특징은 1. 실행 전 안전 초기화 2. 학습 스레드 3. 스플래시 화면 4. 메인 GUI/전처리,증강,훈련 기능**

**흐름 : 라이센스 확인 -> 폴더 선택 -> 파일 정리/라벨맵 생성 -> 이미지,XML 증강 -> TensorFlow 학습 실행 -> 모델 추출 흐름**

# 1. 전체 구조
- 파일 안에는 크게 3개의 클래스가 있음

| 구분 | 역할 |
| -- | -- |
| 전역 함수들 | 관리자 권한 확인, Qt 플러그인 안정화, 경로 검사 |
| TrainingThread | 학습 배치파일 실행, 콘솔 로그 읽기, ProgressBar/Loss 갱신, 학습 종료 처리 |
| SplashScreen | 프로그램 실행 시 뜨는 GUI 로딩 화면 |
| ClassAugChanger | 실제 메인 GUI 버튼, 이미지 변환, XML 수정, 학습 실행, 모델 추출 등 대부분 기능 담당 |


# 2. 전역 함수들
### is_admin()
- 현재 프로그램이 관리자 권한으로 실행 중인지 확인하는 함수
- 주요 역할은:
    - ctypes.windll.shell32.IsUserAnAdmin() 호출
    - 관리자 권한이면 True
    - 실패하거나 권한이 없으면 False

- 파일 상단에서 이 함수 결과를 보고, 관리자 권한이 아니면 ShellExecuteW(..., "runas", ...)로 프로그램을 관리자 권한으로 실행
- 이 GUI가 nvida-smi, shutdown, 파일 이동 같은 권한 민감 작업을 하기에 넣은 구조

### remove_incompatible_qt_styles()
- PyQt6 실행 시 충돌 가능성이 있는 qmoderwindowstyle.dll 을 자동 제거하는 함수

- 핵심 기능:
    - PyInstaller로 묶인 경우 _MEIPASS 기준으로 경로 찾음
    - PyQt6/Qt6/plugins/styles/qmoderwindowsstyle.sll 존재 여부 확인
    - 있으면 삭제
    - 삭제 실패 시 에러 로그 출력

- 즉, Qt 스타일 DLL때문에 GUI 실행이 불안정해지는 문제를 예방하는 함수

### check_qt_plugin_compatibility()
- Qt 스타일 플러그인 DLL의 메타데이터가 정상인지 검사하는 함수

- 주요 처리:
    - plugins/styles 폴더 탐색
    - qmoderwindowsstyle 이름이 들어간 DLL 확인
    - DLL 앞부분 64바이트를 읽어서 QTMETDATA 포함 여부 검사
    - 없으면 실행 불안정 가능성 경고 출력

- 즉, 삭제까지 하지는 않고 의심 DLL 감지용 검사 함수

### check_qt_platform_plugin()
- Qt 플랫폼 플러그인인 qwindows.dll 이 존재하는지 확인하는 함수

- 역할:
    - PyQt6/Qt6/plugins/platforms/qwindows.dll 경로 확인
    - 파일이 없으면 오류 메시지 출력

- 다만 현재 safe_qt_init() 에서는 이 함수가 호출되지 않고 있어서, 필요시 safe_qt_init() 안에 넣어도 됨

### check_path_exists(path, description="경로")
- 특정 경로가 존재하는지 확인하는 공통 유틸 함수

- 예를 들어:
    - D:\AI_SVT_Training_mk\configs
    - D:\AI_SVT_Training_mk\annotations\annos

- 이런 필수 경로가 없으면 콘솔에 출력

### apply_safe_qt_style()
- Qt 스타일을 안정적인 기본 스타일인 "Fusion"으로 강제 적용하는 함수

- 목적은:
    - 외부 스타일 DLL 문제 회피
    - Windows 환경에서 PyQt6 스타일 충돌 방지
    - GUI 테마 안정화

### safe_qt_init()
- Qt 실행 전 안전 초기화를 한 번에 수행하는 통합 함수

- 흐름 순서:
    - remove_incompatible_qt_styles()
    - check_qt_plugin_compatibility()
    - 필수 XML 폴더 경로 화긴
    - config 경로 확인
    - apply_safe_qt_style()
    - apply_safe_qt_style()

- 즉, main() 에서 QApllication 만들기 전에 실행되는 사전 점검 함수


# 3. TrainingThread 클래스
- 이 클래스는 GUI 가 멈추지 않게 학습 실행을 별도 스레드에서 처리하는 역할

### TrainingThread().\__init__(...)
- 학습 스레드 실행에 필요한 설정값을 저장

- 받는 값:

| 인자 | 의미 |
| -- | -- |
| max_steps | GUI에서 계산된 전체 학습 step |
| shutdown_enable | 학습 종료 후 PC 종료 여부 |
| shutdown_hour | 종료 기준 시 |
| shutdown_minute | 종료 기준 분 |
| enable_xla | XLA_ON/OFF |
| parect | 부모 객체 전달 용 (내가 만들 때 오타냈음) |

- 내부에 self.max_steps, self.enable_xla 등 저장해서 run()에서 사용


### TrainingThread._finalize_and_maybe_shutdown(process, best_loss, best_step)
- 학습이 끝났을 때 공통 마무리를 처리하는 함수

- 하는일:
    - 1. ProgressBar를 최대 step으로 보내고 done_signal 을 emit 해서 GUI에 학습 완료를 알림
    - 2. GPU 클럭 제한을 원복
        - nvidia-smi -rgc, nvidia-smi -rmc를 실행해서 GPU Boost 상태를 정상화 하려는 구조
    - 3. PC 종료 옵션이 켜져 있고 현재 시ㅏㄴ이 설정 이후면 
        - shutdown /s /t 60 /f 명령으로 60초 뒤 강제 종료 예약을 걸어둠
    - 4. 학습 프로세스를 terminate() 로 종료 시도

- 즉, 이 함수는 학습 종료 후 정리 담당 핵심 함수

### TrainingThread.run()
- 실제 학습 배치파일을 실행하고 콘솔 로그를 실시간으로 읽는 함수

- 실행하는 배치파일:
    - D:\AI_SVT_Training_mk\1)train_FRCNN_res50.bat

- 주요 흐름:
    - 1. ENABLE_XLA 환경변수를 "1" 또는 "0"으로 설정
    - 2. subprocess.Popen()으로 학습 배치파일 실행
    - 3. stdout을 한 줄씩 읽음
    - 4. [Step] 1234 형태 로그에서 현재 step 추출
    - 5. [LOSS] 0.1234 형태에서 loss 추출
    - 6. Loss/total_loss 로그에서 best loss와 best step 갱신
    - 7. 현재 step이 max_steps 이상이면 학습 종료 처리
    - 8. 학습 완료 문구를 감지하면 _finalize_and_maybe_shutdown() 호출

- 즉, 이 함수는 콘솔 로그 파싱 -> GUI ProgressBar 업데이트 -> 실시간 Loss 표시 -> 학습 종료 감지 담당


# 4. SplashScreen 클래스
- 프로그램 시작할 때 뜨는 로딩 화면

### SplashScreen.\__init__()
- 스플래시 화면 UI 구성

- 구성 요소:
    - 창 테두리 없는 로딩 화면
    - 제목
    - 부제
    - 반짝이 표시용
    - "Loading AI Training Tool..." 라벨
    - 배경색, 글씨색, 둥근 테두리 스타일

### SplashScreen.center_on_screen()
- 스플래시 화면을 모니터 중앙에 배치하는 ㅎ마수

- QApplication의 primary screen 크기를 가져와서:
    - 화면 가로 중앙
    - 화면 세로 중앙
- 위치로 move() 시킴

### SplashScreen.start_animation(on_finished)
- 스플래시 화면 애니메이션 시작하는 함수

- 하는 일:
    - 1. 화면 중앙 배치
    - 2. 스플래시 표시
    - 3. 작은 크기에서 원래 크기로 확대되는 zoom-in 애니메이션 실행
    - 4. 0.25초 후 반짝이 표시
    - 5. 2초 후 finished() 호출

- on_finished에는 보통 메인 윈도우를 여는 함수가 들어감

### SplashScreen.finish(on_finished)
- 스플래시를 닫고 메인 GUI 를 여는 함수

- 흐름:
    - 스플래시 close()
    - -> on_finished()
    - -> 메인 창 show()


# 5. ClassAugChanger 클래스
- 이 클래스가 이 파일의 핵심
- 메인 GUI, 버튼 기능, 이미지 증강, XML 수정, 학습 실행, 모델 추출 모두 여기에 들어 있음

## 5-1. 초기화/UI 관련 함수

### ClassAugChanger.\__init__()
- 메인 GUI 객체 생성 시 처음 실행되는 함수

- 하는 일:
    - progress_signal 을 progress_bar_update() 에 연결
    - initUI() 호출해서 전체 GUI 생성
    - 증강 폴더명 설정
        - 예: aug_H_Flip, aug_V_Shift, aug_Rotation
    - shift 값 설정
        - self.num_shift = [1, 3, 5, 7, 9]
    - rotation 값 설정
        - self.num_rot = [90, 180, 270]
    - RGB/BGR/Gray 출력 폴더명 설정
    - 라이센스 확인 전 모든 버튼 비활성화

### disable_all_buttons_initial()
- 라이센스 확인 전에는 대부분의 버튼 비활성화 하는 함수

- 비활성화 대상:
    - 폴더 선택
    - bmp to jpg
    - 파일 확인
    - 라벨 맵 생성
    - 클래스명 변경
    - 이미지 증강 버튼
    - 학습 시작
    - 모델 추출 등

- 단, 라이센스 확인 버튼만 활성화
- 즉, 라이센스 통과 전 기능 사용 방지용 함수

### initUI()
- GUI 전체를 구성하는 가장 큰 함수

- 주요 구성:
    - 메인 창 제목/크기 설정
    - 전체 StyleSheet 적용
    - annos 폴더 이미지 개수 확인
    - EpochNo.txt 생성/읽기
    - trian step 최소/최대 제한 적용
        - 최소 10,000
        - 최대 300,000
    - 라이센스 확인 버튼
    - Help 버튼
    - 폴더 선택 버튼
    - bmp to jpg 버튼
    - 파일 확인 / 라벨맵 생성 / 클래스 번호 변경 버튼
    - 클래스명 변경 입력창
    - XLA ON/OFF 버튼
    - 이미지 증강 버튼
    - XML 객체 추가 버튼
    - Auto Label XML 조정 버튼
    - Row/Column 입력창
    - Color/Gray/RGB/BGR 변환 버튼
    - 학습 ProgressBar
    - 훈련 종료 후 PC 종료 옵션
    - 훈련 시작 / 훈련 결과 확인 / 모델 추출 버튼
    - checkpoint 입력창
    - epoch multipier 라디오 버튼 x10, x20, x30
    - 실시간 Loss 표시 라벨

- 즉, 이 함수가 실제 화면에 보이는 거의 모든 버튼과 입력창을 만드는 함수

### _on_xla_change(is_on: bool)
- XLA ON/OFF 상태를 내부 UI와 내부 변수에 반영하는 함수

- 하는 일:
    - XLA 버튼 텍스트를 "XLA: ON" 또는 "XLA: OFF"로 변경
    - 버튼 체크 상태 동기화
    - 내부 상태값 self.xla_enabled 저장

### update_loss_label(loss, step)
- 학습 중 실시간 Loss 표시를 갱신하는 함수

### progress_bar_update(step)
- 학습 step 기준으로 ProgressBar 를 업데이트하는 함수

- 핵심:
    - step이 self.train_steps를 넘지 않도록 제한
    - ProgressBar value 설정
    - 퍼센트 계산
    - "75.21%" 형태로 표시

- 즉, 학습 로그에서 받은 step 을 GUI 진행률로 변환하는 함수

### onTrainingDone(best_info)
- 학습 완료 후 GUI를 복구하는 함수

- 하는 일:
    - 학습 완료 메시지 출력
    - 최적 Loss 와 해당 Step 표시
    - ProgressBar 0%로 초기화
    - 학습 버튼, 증강 버튼, XLA 버튼, ㄹ디오 버튼 다시 활성화

- 즉, 학습 종료 후 사용자가 다시 작업할 수 있게 UI를 복원하는 함수


# 6. 입력창 prefix 유지 함수들
- 이 함수들은 사용자가 입력창 앞의 안내 문구를 지워버리지 못하게 하는 역할

### on_text_edited(text)
- 기존 클래스명: prefix를 유지하는 함수
- 사용자가 앞부분을 지우면 다시 붙여줌

### on_text_edited_1(text)
- 변경할 클래스명: prefix를 유지하는 함수

### on_row_edited(text)
- 행 번호: prefix를 유지하는 함수
- Row by Column 기능에서 행 수 입력칸에 사용

### on_ckpt_edited(text)
- 체크포인트 입력: prefix를 유지하는 함수
- 모델 추출 시 checkpoint 번호 입력칸에 사용됨


# 7. 입력값 추출 함수들
- 위 prefix 를 제외하고 실제 입력값만 가져오는 함수들

### get_real_classname()
- 기존 클래스명: ABC에서 ABC만 반환

### get_real_classname_1()
- 변경할 클래스명: DEF에서 DEF만 반환

### get_row_number()
- 행 번호: 3 에서 3만 반환

### get_col_number()
- 열 번호: 4 에서 4만 반환

### get_ckpt_number()
- 체크포인트 입력: 52000에서 52000만 반환


# 8. Epoch / 학습 step 관련 함수
### on_mul_changed(button)
- x10, x20, x30 라디오 버튼이 바뀌었을 때 실행되는 함수

- 하는 일:
    - 1. 선택된 multiplier 확인
    - 2. annos 폴더 이미지 개수 계산
    - 3. EpochNo.txt 에 이미지 개수 x multiplier 저장
    - 4. 최소 10,000 / 최대 300,000 으로 clamp
    - 5. self.train_steps 갱신
    - 6. ProgressBar maximum 갱신
    - 7. 팝업으로 변경 결과 표시

- 예를 들어 이미지가 1,000장이고 ×20이면: 1000 x 20 = 20000
    - 이 값이 학습 step 기준이 됨 

### on_annos_changed(path)
- D:\AI_SVT_Training_mk\annotations\annos 폴더가 변경될 때 자동 실행되는 함수

- 하는 일:
    - annos 폴더 안 이미지 개수 다시 계산
    - self.num_images 갱신
    - update_epochno_file() 호출

- 즉, 이미지가 추가/삭제되면 EpochNo 를 자동 갱신하려는 목적

### update_epochno_file()
- 현재 이미지 개수와 선택된 multiplier 를 기준으로 EpochNo.txt 를 다시 쓰는 함수
- 기본 multiplier는 20, 라데ㅣ오 버튼이 있으면 현재 값을 사용함

- 저장 위치:
    - D:\AI_SVT_Training_mk\annotations\EpochNo.txt


# 9. 라이센스 / Help 관련 함수
### get_last_8_mac()
- 라이센스 확인 버튼을 눌렀을 때 실행되는 함수

- 흐름:
    - 1. uuid.getmode()로 MAC 기반 숫자 가져오기
    - 2. 마지막 8자리 추출
    - 3. 앞 4자리와 뒤 4자리로 나눔
    - 4. (tail * 3.14) + (head * 3.14) 방식으로 license key 계산
    - 5. D:/svtdata/{MAC끝8자리}.lic 파일 확인
    - 6. .lic 파일의 binary double 값을 읽음
    - 7. 계산한 license key와 같으면 Licensed 처리
    - 8. 다르면 No license 처리

- 라이센스가 통과되면 폴더 선택 버튼만 활성화되고, 이후 폴더 선택을 해야 나머지 버튼들이 활성화 됨

### show_manual()
- Help 버튼을 눌렀을 때 FlowChart.pdf 를 찾아서 여는 함수

- 검색 순서:
    - 1. exe 또는 py 파일 위치
    - 2. 사용자가 선택한 폴더
    - 3. D:\AI_SVT_Training_mk\FlowChart.pdf
    - 4. 네트워크 경로:
        - svtechnas\SVT_Project
        - aisvt\AI_RESOURCE
    - 5. 로컬 드라이브
        - C:\
        - D:\

- 찾으면 _open_pdf()로 열고, 못 찾으면 경고창 띄움

### _search_pdf_in_roots(roots, filename="FlowChart.pdf")
- 여러 root 경로를 돌면서 PDF 파일을 찾는 함수
- 예를 들어 C:\, D:\, NAS 경로를 os.walk()로 탐색해서 FlowChart.pdf가 있으면 해당 경로를 반환

### _open_pdf(pdf_path)
- PDF 파일을 실제로 여는 함수


# 10. 폴더 선택 / 버튼 활성화 관련 함수

### directory()
- 폴더 선택 버튼을 눌렀을 때 실행

- 하는 일:
    - 1. 폴더 선택 창 열기
    - 2. 선택한 폴더 경로를 self.folder_path 에 저장 
    - 3. 기능 버튼 활성화
    - 4. 선택 폴더 안 이미지 목록 생성
    - 5. 선택 폴더 안 XML 목록 생성
    - 6. 증강 파일이 아닌 원본 이미지/XML 만 따로 필터링
    - 7. refresh_original_list() 호출

- 증강 파일 제외 패턴:
    - _HF, _VF, _HS숫자, _VS숫자, _clahe, _arr, _expand, _shrink, _left, _right, _up, _down, _RT숫자
        - 이미 증강 된 파일을 또 증강하지 않기 위한 필터 

### enable_all_buttons_after_folder()
- 폴더 선택 후 기능 버튼들을 활성화하는 함수
- 라이센스 통과 후 폴더를 선택해야 실제 작업 버튼들이 활성화되는 구조

### refresh_original_list()
- 현재 선택 폴더를 다시 스캔해서 원본 이미지/XML 목록 갱신

- 역할:
    - 전체 이미지 목록 생성
    - 전체 XML 목록 생성
    - 증강 파일명 패턴 제외
    - self.original_img_list
    - self.original_xml_list
- 를 다시 저장

# 11. 파일 정리 / 라벨맵 / config 관련 함수

### bmptojpg()
- 선택 폴더 안의 .bmp 파일을 .jpg 로 변환하는 함수

- 처리 흐름:
    - 1. jpg_images 폴더 생성
    - 2. bmp_images 폴더 생성
    - 3. BMP 파일을 열어서 RGB JPG로 저장
    - 4. 원본 BMP 파일은 bmp_images 폴더로 이동

### checkAllFiles()
- 이미지와 XML 짝이 안 맞는 파일을 삭제하는 함수

- 하는 일:
    - XML은 있는데 같은 이름의 JPG가 없으면 XML 삭제 
    - JPG는 있는데 같은 이름의 XML이 없으면 JPG 삭제
    - XML 안에 <object>가 없으면 XML 삭제

- 즉, 학습에 들어가면 안 되는 빈 파일/짝 안 맞는 파일 정리용

### changeClassName()
- XML 안의 클래스명을 일괄 변경하는 함수

### label_map()
- XML 파일들에서 클래스명을 수집해서 label_map.pbtxt 를 생성하는 함수

- 흐름:
    - 1. count_classes_in_folder() 로 클래스명 수집
    - 2. 클래스 개수와 이름을 UI에 표시
    - 3. label_map.pbtxt 에 저장

- 라벨맵 생성과 클래스 수 계산 로직은 count_classes_in_folder(), extract_classes_from_xml()과 연결

### count_classes_in_folder(xml_folder_list)
- 여러 XML 파일에서 클래스명을 모아서 class set 을 만드는 함수
- 클래스 개수, 클래스명 set 반환

### extract_classes_from_xml(xml_file)
- XML 파일 하나에서 <object><name> 값을 모두 추출하는 함수

### changeClassNum()
- config 파일 안의 num_classes 값을 label map 의 최대 id 값으로 바꾸는 함수

- 흐름:
    - 1. maxid()로 label_map.pbtxt 의 최대 id 확인
    - 2. config 파일 읽기
    - 3. 정규식으로 num_classes:\d+ 찾기
    - 4. num_classes:{maxid}로 치환
    - 5. config 파일 저장

- 즉, 라벨맵 클래스 수와 TensorFlow config의 class 수를 맞추는 기능

### maxid()
- label_map.pbtxt 에서 가장 큰 id: 값을 찾는 함수


# 12. 학습 실행 / 모델 추출 함수

### runCommand()
- 훈련 시작 버튼을 눌렀을 때 실행되는 핵심 함수

- 전체 흐름:
    - 1. D:\AI_SVT_Training_mk\train_result 폴더가 이미 있으면 학습 중단 -> 기존 결과 덮어쓰기 방지
    - 2. XML 클래스명과 label_map.pbtxt 클래스명 비교
    - 3. XML에는 있는데 label_map에는 없는 라벨이 있으면 학습 중단
    - 4. ProgressBar 초기화
    - 5. GPU Boost 실행 
        - nvidia-smi -lgc 2500, 2500
    - 6. PC 종료 옵션 확인
    - 7. XLA ON/OFF 상태 확인
    - 8. TrainingThread 생성
    - 9. Progress/Loss/Done signal 연결
    - 10. 학습 스레드 시작
    - 11. 학습 중 버튼들 비활성화

- 즉, 학습 실행 전 검증 + GPU 설정 + 학습 스레드 시작 + GUI 잠금 담당

### pbmodel()
- 모델추출 버튼을 눌렀을 때 실행

- 하는 일:
    - 1. checkpoint 입력값 가져오기
    - 2. 숫자가 아니면 경고
    - 3. 입력값을 200으로 나눠서 checkpoint 번호로 변환
    - 4. train_result/checkpoint 파일의 첫줄을
        - model_checkpoint_path: "ckpt-{번호}" 로 수정
    - 5. D:\AI_SVT_Training_mk\2)model_pb.bat 실행
    - 6. 모델 추출 중 학습 시작 버튼 비활성화

### checkTrainingComplete()
- 훈련 파일 결과 확인 버튼 기능
- 확인하는 조건:
    - D:\AI_SVT_Training_mk\train_result 폴더가 존재하고 내부에 파일이 있는가?
    - 있으면 훈련 완료 메시지, 없으면 아직 완료되지 않았다는 경고


# 13. CLAHE 증강 함수
**CLAHE는 이미지 대비를 지역적으로 보정하는 방식**
### clahe_aug()
- 원본 이미지에 CLAHE 를 적용하는 함수

- 처리 흐름:
    - 1. 증강 파일 패턴을 제외하고 원본 이미지만 필터링
    - 2. 이미지를 grayscale로 읽음
    - 3. cv2.createCLAHE(clipimit=5.0, tileGridSize=(8, 8)) 생성
    - 4. CLAHE 적용
    - 5. aug_clahe 폴더에 저장
    - 6. ProgressBar 갱신
    - 7. 완료 메시지 출력

- 주의할 점은 이 함수는 이미지는 생성하지만 XML을 같이 복사/변환하지는 않음


# 14. Horizontal Flip 관련 함수

### imgflip()
- 원본 이미지를 좌우 반전해서 저장하는 함수

### HFlip_bbox_coordinates(image_width, xmin, xmax)
- 좌우 반전 시 bbox의 x좌표를 계산하는 함수

- 계산 방식:
    - new_xmin = image_width - xmin
    - new_xmax = image_width - xmax

- 단, 실제 xmlHflip() 에서는 반환된 값을 다시 순서를 바꿔서 넣음

### xmlHflip()
- 좌우 빈전 이미지에 맞춰 XML bbox를 수정하는 함수

- 하는 일:
    - filename을 _HF.jpg로 변경
    - path도 _HF.jpg로 변경
    - bbox의 xmin, xmax 좌표를 좌우 반전 기준으로 변경
    - XML 을 aug_H_Flip 폴더에 젖ㅇ
    - annos 폴더에도 복사

- 이미 _HF.xml 이 있으면 중복 실행하지 않도록 return 함


# 15. Vertical Flip 관련 함수

### imgVflip()
- 원본 이미지를 상하 반전해서 저장하느 함수

### VFlip_bbox_coordinates(image_height, ymin, ymax)
- 상하 반전 시 bbox의 y좌표를 계산하는 함수

- 계산 방식:
    - new_ymin = image_height - ymin
    - new_ymax = image_height - ymax

### xmlVflip()
- 상하 반전 이미지에 맞춰 XML bbox 를 수정하는 함수

- 수정 대상:
    - filename
    - path
    - ymin
    - ymax


# 16. Horizontal Shift 관련 함수

### imgHshift()
- 이미지를 오른쪽으로 이동시키는 함수

- 이동 값은:
    - self.num_shift = [1, 3, 5, 7, 9]
- 즉, 한 이미지당 5개의 증강 이미지가 생김

- OpenCV의 warmAffine() 을 사용해서 x 방향으로 이동시킴

### Hshift_bbox_coordinates(num_shift, xmin, xmax)
- 수평 이동한 만큼 bbox x좌표도 같이 이동시키는 함수
    - xmin + shift
    - xmax + shift

### xmlHshift(xml_file, i)
- 특정 XML 하나를 특정 shift 값 i 만큼 수평 이동 기준으로 수정하는 함수 

- 하는 일:
    - filename/path를 _HSi.jpg로 변경
    - bbox 의 xmin, xmax 에 i 더함
    - _HSi.xml 로 저장
    - annos 폴더에 복사

### xmlHshift_all()
- 모든 원본 XML 에 대해 xmlHshift()를 반복 실행하는 함수
    - 즉, 모든 XML × [1,3,5,7,9] 만큼 XML 생성


# 17. Vertical Shift 관련 함수

### imgVshift()
- 이미지를 아래쪽으로 이동시키는 함수
- 이동 값은 수평 이동과 동일하게 [1, 3, 5, 7, 9]

### Vshift_bbox_coordinates(num_shift, ymin, ymax)
- 수직 이동한 만큼 bbox y좌표를 이동시키는 함수
    - ymin + shift
    - ymax + shift

### xmlVshift(xml_file, i)
- 특정 XML 하나를 수직 이동 기준으로 수정하는 함수

- 수정 대상:
    - filename/path를 _VSi.jpg 로 변경
    - bbox의 ymin, ymax 에 i 더함
    - _VSi.xml 로 저장
    - annos 폴더에 복사

### xmlVshift_all()
- 모든 원본 XML에 대해 xmlVshift()를 반복 실행하는 함수


# 18. Row by Column 관련 함수

### imgRowColumn(img)
- 이미지를 사용자가 입력한 행/열 개수를 나눈 뒤, 조각 순서를 한 칸씩 멀어서 재배치하는 함수

- 예를 들어:
    - 행: 2
    - 열: 2
- 이면 이미지를 4조각으로 나누고, 조각 순서를 바꿔서 새 이미지로 저장함

### xmlRowColumn(xml_file)
- Row by Column으로 이미지 조각 위치가 바뀐 것에 맞춰 XML bbox 도 이동시키는 함수

- 하는 일:
    - filename/path를 _arr.jpg로 변경 
    - bbox 좌표를 rowNcolumn_coordinates()로 변환
    - _arr.xml로 저장 

### rowNcolumn_coordinates(image_width, image_height, xmin, xmax, ymin, ymax)
- Row by Column 변환 후 bbox 가 어느 위치로 이동해야 하는지 계산하는 함수

- 현재 로직 조건:
    - 1. bbox가 첫 번째 column보다 오른쪽에 있으면 왼쪽으로 이동
    - 2. bbox가 아래 행에 있고 첫 column에 있으면 오른쪽 위로 이동
    - 3. bbox가 첫 행/첫 column에 있으면 마지막 column/마지막 row 쪽으로 이동

- 즉, 이미지 조각 순서를 바꾼 것에 맞춰 bbox 위치도 같이 바꾸는 계산 함수

### RowColumn_all()
- Row by Column 기능 전체 실행 함수

- 흐름:
    - 1. 행/열 입력값 확인
    - 2. 출력 폴더 생성
    - 3. 이미 _arr.jpg 또는 _arr.xml 이 있으면 중복 실행 방지
    - 4. 모든 원본 이미지에 imgRowColumn() 실행
    - 5. 모든 원본 XML에 xmlRowColumn() 실행
    - 6. ProgressBar 갱신
    - 7. 완료 메세지 출력


# 19. Rotation 관련 함수

### imgRotation()
- 이미지를 90도, 180도, 270도 회전해서 저장하는 함수

- OpenCV의 cv2.getRotationMatrix2D() 와 cv2.warpAffine()을 사용

### rotate_vertices(vertices, angle, bbox_center, image_height, image_width)
- bbox 꼭짓점 하나를 bbox 중심 기준으로 회전시키는 함수

- 하는 일:
    - 각도를 radian으로 변환
    - 회전 좌표 계산
    - 이미지 바깥으로 나가면 0 또는 image width/height로 보정

### maxNminDistance(rotated_vertices, bbox_center)
- 회전된 bbox 꼭짓점들 중 중심에서 가장 멀리 떨어진 x/y 거리를 계산하는 함수

- 반환값:
    - dis_x, dis_y
- 이 값은 최종 bbox의 반폭/반높이처럼 사용

### rotate_central_point(bbox_center, angle, image_center)
- bbox 중심점을 이미지 중심 기준으로 회전시키는 함수
- 즉, bbox 자체의 중심이 회전 후 어디로 이동하는지 계산

### final_vertices(new_bbox_center, distance_from_center)
- 새 bbox 중심점과 거리값을 이용해서 최정 bbox 좌표를 만드는 함수

- 반환값:
    - (new_xmin, new_ymin, nex_xmax, new_ymax)

### move_and_rotate_box(box_vertices, i, image_center, bbox_center, image_height, image_width)
- 회전 bbox 계산을 한 번에 묶은 함수

- 내부 순서:
    - 1. rotate_vertices()로 각 꼭짓점 회전
    - 2. maxNminDistance()로 bbox 크기 계산
    - 3. rotate_central_point()로 bbox 중심 이동
    - 4. final_vertices()로 최종 bbox 생성

- 즉, XML 회전 보정의 핵심 계산 함수

### xmlRotation(xml_file, i)
- 회전된 이미지에 맞춰 XML bbox를 수정하는 함수

- 하는 일:
    - filename/path를 _RT{i}.jpg 로 변경
    - 기존 bbox 좌표 읽기
    - bbox 꼭짓점 배열 생성
    - bbox 중심 계산
    - move_and_rotate_box()로 회전 후 bbox 계산
    - XML 저장
    - annos 폴더에 복사

- 회전 이미지와 XML 생성 로직은 이 부분이 핵심

### xmlRotation_all()
- 모든 원본 XML 에 대해 90/180/270도 회전 XML 을 생성하는 함수


# 20. 색상 변환 함수

### ColorToGray()
- 선택 폴더 안 이미지를 grayscale로 변환하는 함수

- 하는 일:
    - 원본 컬러 이미지 읽기
    - grayscale 변환 이미지 저장
    - 원본 컬러 이미지도 별도 폴더에 저장

### BGRtoRGB()
- OpenCV 기준 BGR 이미지를 RGB로 변환해서 저장하는 함수

- 각 이미지마다:
    - RGB 변환본 저장
    - 원본 BGR도 저장

### RGBtoBGR()
- RGB 이미지를 BGR로 변환해서 저장하는 함수
- 다만, cv2.imread()는 기본적으로 BGR로 읽기 때문에, 실제로는 "파일을 RGB라고 가정하고 BGR로 변환한다" 는 구조


# 21. XML 객체 추가 / Auto Label 조정 함수

### transform_xml_boxes()
- annos 폴더 안 XML 파일에 bbox 변형 object 추가하는 함수

- 기준값:
    - mm_per_pixel = 0.03
    - offset_mm = 0.15
    - offset_px = 5

- 내부에 보조함수 6개:

| 내부 함수 | 기능 |
| -- | -- |
| modify_expand() | bbox를 상하좌우 확장 |
| modify_shrink() | bbox를 상하좌우 축소 |
| modify_left() | bbox를 왼쪽 이동 |
| modify_right() | bbox를 오른쪽 이동 |
| modify_up() | bbox를 위로 이동 |
| modify_down() | bbox를 아래로 이동 |

- 전체 흐름:
    - 1. XML 파일 목록 읽기
    - 2. object 가 이미 100개 이상이면 중복 증강으로 판단하고 중단
    - 3. 기존 object를 deepcopy
    - 4. 확장/축소/상하좌우 이동 object 추가
    - 5. 이미지 영역 밖으로 나가거나 bbox가 이상하면 제외
    - 6. 원본 XML 파일에 object 추가 저장

- 즉, 이 함수는 이미지 파일을 새로 만들지 않고 XML 내부 object만 늘리는 방식

### adjust_bbox_xml(input_xml, output_xml, shift_x=0, shift_y=0, expand_x=0, expand_y=0)
- Auto Label 조정용으로 XML 하나를 변형해서 새 XML로 저장하는 함수

- 하는 일:
    - 1. 입력 XML 읽기
    - 2. 출력 XML 파일명 기준으로 <filename> 변경
    - 3. <path>를 annos 경로로 고정
    - 4. bbox 확장/축소 적용
    - 5. bbox 위치 이동 적용
    - 6. 음수 좌표 방지
    - 7. XML 저장
    - 8. XML을 annos 폴더에 복사
    - 9. 원본 JPG도 새 이름으로 복사
    - 10. 새 JPG도 annos 폴더에 복사

- 즉, 이 함수는 XML과 JPG를 같이 새 파일로 만들어주는 보조 함수

### process_xml_variants()
- Auto Label XML 조정 버튼의 전체 실행 함수

- 각 변형값:

| 변형 | 의미 |
| -- | -- |
| expand | bbox +3px 확장 |
| shrink | bbox -3px 축소 |
| left | bbox 왼쪽 3px 이동 |
| right | bbox 오른쪽 3px 이동 |
| up | bbox 위쪽 3px 이동 |
| down | bbox 아래쪽 3px 이동 |

- 흐름:
    - 1. 원본 JPG만 필터링
    - 2. 하위 폴더 생성
    - 3. 이미 증강 파일이 있으면 중복 실행 방지
    - 4. 각 JPG에 대응되는 XML 확인
    - 5. adjust_bbox_xml() 호출해서 변형 XML/JPG 생성
    - 6. ProgressBar 갱신
    - 7. 완료 메세지 표시

- 이 로직은 파일별로 XML과 JPG를 따로 만들어서 실제 학습 데이터로 넣을 수 있게 만드는 쪽


# 22. main()
- 프로그램 시작점

- 실행 순서:
    - 1. safe_qt_init() 실행
    - 2. QApplication 생성
    - 3. SplashScreen() 생성
    - 4. ClassAugChanger() 메인 GUI 생성
    - 5. 스플래시 애니메이션 시작
    - 6. 스플래시 끝나면 메인 GUI show
    - 7. app.exec() 실행

- 즉, 전체 프로그램을 실제로 띄우는 진입점


**이 파일은 "데이터 준비 + XML 라벨 관리 + 증강 + TF 학습 실행 + 모델 추출" 까지 한 번에 묶은 사내 GUI 툴, 실제 핵심 흐름은 ClassAugChanger.initUI()에서 버튼을 만들고, 버튼 클릭 시 imgHflip/xmlHflip, runCommand, pbmodel 같은 함수들이 실행되는 구조**

