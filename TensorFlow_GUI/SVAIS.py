import sys, os, subprocess, ctypes, re, cv2, time, shutil, glob, copy
from PIL import Image as PILImage, ImageEnhance, ImageChops, ImageFilter
from PIL.Image import Transpose     # 버전이슈 _06.05 추가

from PyQt6.QtWidgets import *

from PyQt6.QtCore import Qt, pyqtSignal, QThread, QMetaObject, QTimer, QTime, QFileSystemWatcher, QUrl, QEasingCurve, QRect, QPropertyAnimation
from PyQt6.QtGui import QMovie, QPixmap, QDesktopServices   # 11.18

import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm

import platform

import threading

from collections import deque       # 05.13
from tkinter import *           # 05.23 
from datetime import datetime   # 05.23

import tensorflow as tf     # 06.10
from tensorboard.backend.event_processing import event_accumulator      # 06.10
from PyQt6.QtWidgets import QSizePolicy     # 06.13
from pathlib import Path        # 07.07

import struct # 09.15
import uuid # 09.15

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 26.03.06
from PyQt6.QtWidgets import QGraphicsOpacityEffect

## total_step 고정값 X, configs의 num_steps 에서 읽어옴 
# EarlyStopping 방식이라 loss 개선이 안 되면 강제 종료 (wait >= 99)
# Extract model → Ckpt 입력 시 숫자 저장 + Run Training File 버튼 비활성화 

# ProgressBar 멈춤 → [Step] 로그가 멈췄을 때 타임아웃 처리 (15분 경과시 강제 종료) _05.27추가
# shutdown_after_* 값 로그로 확인 + 관리자 권한 포함 강제 종료 처리 _05.27추가
# flush = True : 자꾸 CONSOLE 완료가 안 됨, 모든 print문에 강제로 출력 _05.28추가

# 06.09_오류발생 가능성 방지 코드 삽입

# annos 폴더 고정:
# XML 추가/Auto Label 등 일부 기능은 경로가 고정
# 사용자 선택 폴더:
# 증강 등은 사용자가 선택한 폴더(self.folder_path) 기준

# Gpu(Boost+Normal)_06.04
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if not is_admin():
    # 관리자 권한으로 재실행
    ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    sys.exit()

# Qt6 버전때문에 생길 오류 방지코드 _06.09
# 1. 호환되지 않는 Qt 스타일 DLL 자동 제거
def remove_incompatible_qt_styles():
    try:
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))  # PyInstaller 대응
        style_path = os.path.join(base_dir, "PyQt6", "Qt6", "plugins", "styles")
        style_file = os.path.join(style_path, "qmodernwindowsstyle.dll")
        if os.path.exists(style_file):
            os.remove(style_file)
            print("[INFO] qmodernwindowsstyle.dll 자동 제거 완료", flush=True)
    except Exception as e:
        print(f"[ERROR] DLL 제거 실패: {e}", flush=True)

# 2. DLL 메타데이터 유효성 확인
def check_qt_plugin_compatibility():
    try:
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        plugin_path = os.path.join(base_dir, "PyQt6", "Qt6", "plugins", "styles")
        if os.path.exists(plugin_path):
            for file in os.listdir(plugin_path):
                if file.endswith(".dll") and "qmodernwindowsstyle" in file.lower():
                    dll_path = os.path.join(plugin_path, file)
                    with open(dll_path, "rb") as f:
                        data = f.read(64)
                        if b"QTMETADATA" not in data:
                            print(f"[경고] {file} 메타데이터 이상 감지 (실행 불안정 가능성)", flush=True)
    except Exception as e:
        print(f"[ERROR] Qt 스타일 DLL 검사 실패: {e}", flush=True)

def check_qt_platform_plugin():
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    qt_platform_path = os.path.join(base_dir, "PyQt6", "Qt6", "plugins", "platforms", "qwindows.dll")
    if not os.path.exists(qt_platform_path):
        print("[오류] Qt 플랫폼 플러그인 (qwindows.dll) 누락됨", flush=True)

def check_path_exists(path, description="경로"):
    if not os.path.exists(path):
        print(f"[경고] {description} 경로 존재하지 않음: {path}", flush=True)

# 3. 스타일 안전 적용
def apply_safe_qt_style():
    try:
        from PyQt6.QtWidgets import QApplication
        QApplication.setStyle("Fusion")  # 안정적인 기본 스타일
    except Exception as e:
        print(f"[경고] Qt 스타일 적용 실패: {e}", flush=True)

# 4. 전체 통합 실행 (main 함수 실행 전 호출)
def safe_qt_init():
    remove_incompatible_qt_styles()
    check_qt_plugin_compatibility()
    check_path_exists(r"D:\\AI_SVT_Training_mk\\annotations\\annos", "XML 폴더")
    check_path_exists(r"D:\\AI_SVT_Training_mk\\configs", "Config 파일 경로")
    apply_safe_qt_style()

# 기존 exe 코드
class TrainingThread(QThread):
    progress_signal = pyqtSignal(float)
    done_signal = pyqtSignal(tuple)     # 06.05 수정, tuple형태로 best_loss전달
    loss_signal = pyqtSignal(float, int)    # float : Loss, int : Step 전달

    # 시간 걸어서 shutdown 추가 | 사용자 선택 시각_250613
    def __init__(self, max_steps, shutdown_enable = False, shutdown_hour=0, shutdown_minute=0, 
                enable_xla=False, parect=None):     # enable_xla=False 09.02추가
        super().__init__(parect)        # 06.25 parect 추가, 위에도 동일
        self.shutdown_enable = shutdown_enable
        self.shutdown_hour = shutdown_hour
        self.shutdown_minute = shutdown_minute
        self.max_steps = max_steps      # GUI 가 넘겨준 train_steps 저장
        self.enable_xla = enable_xla        # 저장 _09.02 

    # 무조건 종료_05.27
    def _finalize_and_maybe_shutdown(self, process, best_loss, best_step):
        """공통 마무리 처리:
        - 진행바/시그널 emit
        - GPU 클럭 복원
        - (체크박스 ON) AND (현재시간 >= 설정시각)일 때만 PC 종료 예약
        - 하위 프로세스 정리
        """
        # 1) 진행바/시그널
        try:
            if self.max_steps is not None:
                self.progress_signal.emit(self.max_steps)
            self.done_signal.emit((best_loss, best_step))
        except Exception:
            pass

        # 2) GPU 설정 복원 (안전하게 시도)
        try:
            print("→ GpuNormal.bat 실행", flush=True)
            import subprocess, time
            subprocess.run(['nvidia-smi', '-rgc'], check=True)
            subprocess.run(['nvidia-smi', '-rmc'], check=True)
            time.sleep(2)
        except Exception:
            pass

        # 3) "설정 시각 이상"일 때만 종료
        try:
            if self.shutdown_enable:
                from datetime import datetime
                now = datetime.now()
                threshold = now.replace(
                    hour=self.shutdown_hour,
                    minute=self.shutdown_minute,
                    second=0,
                    microsecond=0
                )
                if now >= threshold:
                    delay = 60  # 최소 60초로 여유
                    print(f"[INFO] (조건충족) PC를 {delay}초 후에 종료합니다.", flush=True)
                    import subprocess
                    # 예약 명령은 부모 프로세스 종료 이후에도 유지되도록 Popen 사용
                    subprocess.Popen(f"shutdown /s /t {delay} /f", shell=True)
                else:
                    print("[INFO] 설정 시각 전이므로 종료하지 않습니다.", flush=True)
        except Exception as e:
            print(f"[WARN] 종료 예약 처리 중 예외: {e}", flush=True)

        # 4) 학습 프로세스 정리 (가능하면 정리)
        try:
            process.terminate()
        except Exception:
            pass

    def run(self):
        current_step = 0
        # loss율 _06.05
        best_loss = float('inf')
        best_step = 0
        done_flag = False
        final_step = None
        # 09.02
        env = os.environ.copy()
        env["ENABLE_XLA"] = "1" if self.enable_xla else "0"

        cmd_command = 'D:\\AI_SVT_Training_mk\\1)train_FRCNN_res50.bat'
        process = subprocess.Popen(cmd_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, env=env)
        
        prev_line = ""  # 이전 줄 저장

        for line in iter(process.stdout.readline, ''):
            raw = line.rstrip("\n")
            # [Step] 로그가 있으면 현재 스텝 추출, 진행바 갱신, 해당 줄은 출력하지 않고 다음 줄로 넘어감
            if "[Step]" in raw:
                m = re.search(r'\[Step\]\s*(\d+)', raw)
                if m:
                    current_step = int(m.group(1))
                    self.progress_signal.emit(current_step)   # GUI 진행바 갱신
                continue
            # [LOSS] 로그가 있으면 loss 추출, 실시간 loss 갱신, 해당 줄은 출력하지 않고 다음 줄로 넘어감
            if "[LOSS]" in raw:
                m = re.search(r'\[LOSS\]\s*([0-9.]+)', raw)
                if m:
                    loss = float(m.group(1))
                    self.loss_signal.emit(loss, current_step)
                continue
            # Step이 포함된 줄에서 숫자 추출, 진행바 갱신
            if "Step" in raw:
                sm = re.search(r'\bStep:?\s*(\d+)\b', raw)
                if sm:
                    current_step = int(sm.group(1))
                    self.progress_signal.emit(current_step)
            # Loss/total_loss가 포함된 줄에서 loss 추출, 최적 loss/step 갱신, 실시간 loss 갱신
            if "Loss/total_loss" in raw:
                lm = re.search(r'["\']Loss/total_loss["\']\s*:\s*([0-9]*\.?[0-9]+)', raw)
                if lm:
                    loss_val = float(lm.group(1))
                if loss_val < best_loss:
                    best_loss, best_step = loss_val, current_step
                self.loss_signal.emit(loss_val, current_step)

            # 최대 스텝 도달 시 종료 (훈련 종료 시 마무리 정리)
            if self.max_steps is not None and current_step >= self.max_steps:
                print("→ Training finished (by step >= max).", flush=True)
                self._finalize_and_maybe_shutdown(process, best_loss, best_step)
                return
            
            print(raw, flush=True)

            # 종료 메세지 감지 추가
            if "It's done. Do NOT need to update Best Loss" in line:
                done_flag = True

            total_match = re.search(r'\[TOTAL_STEPS\]\s*(\d+)', line)
            if total_match:
                final_step = int(total_match.group(1))
                prev_line = line
                continue

            step_match = re.search(r'\[Step\]\s*(\d+)', line)
            if step_match:
                current_step = int(step_match.group(1))
                if final_step is not None and current_step >= self.max_steps:       
                    done_flag = True

            loss_match = re.search(r'\[LOSS\]\s*([0-9.]+)', prev_line + line)
            if loss_match:
                loss = float(loss_match.group(1))
                if loss < best_loss:
                    best_loss = loss
                    best_step = current_step
                self.loss_signal.emit(loss, current_step)

                if done_flag:
                    # 공통 마무리 처리로 통합
                    self._finalize_and_maybe_shutdown(process, best_loss, best_step)

                    # 남은 출력 수거 시도 (있으면 출력)
                    try:
                        remaining,_ = process.communicate(timeout=2)
                        if remaining:
                            print(remaining.strip(), flush=True)
                    except Exception:
                        pass

                    break  # 반복 X

                # 정상 훈련 중: 스텝 값 그대로 emit
                self.progress_signal.emit(current_step)
                prev_line = line       # 현재 줄을 다음 반복에서 이전 줄로 저장
        
        process.stdout.close()
        process.wait()

# 26.03.06 : 처음 실행 시 스플래시 화면 + Qt 초기화 안전 처리
class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )

        self.resize(760, 360)
        self.setStyleSheet("""
            QWidget {
                background-color: #FFFBE6;
                border: 2px solid #DDECCC;
                border-radius: 24px;
            }
            QLabel {
                background: transparent;
                border: none;
            }
        """)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(30, 30, 30, 30)
        root_layout.setSpacing(10)

        self.title_label = QLabel("SVAIS", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #356246;
                font-size: 30pt;
                font-weight: 700;
            }
        """)

        self.sub_label = QLabel("(Smart Vision Artificial Intelligence Solution)", self)
        self.sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub_label.setStyleSheet("""
            QLabel {
                color: #4E7D5B;
                font-size: 11pt;
                font-weight: 500;
            }
        """)

        self.flash_label = QLabel("✦", self)
        self.flash_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.flash_label.setStyleSheet("""
            QLabel {
                color: #9DBB86;
                font-size: 42pt;
                font-weight: 700;
            }
        """)
        self.flash_label.hide()

        self.loading_label = QLabel("Loading AI Training Tool...", self)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("""
            QLabel {
                color: #6B8E63;
                font-size: 10pt;
                font-weight: 600;
            }
        """)

        root_layout.addStretch(1)
        root_layout.addWidget(self.title_label)
        root_layout.addWidget(self.sub_label)
        root_layout.addSpacing(10)
        root_layout.addWidget(self.flash_label)
        root_layout.addSpacing(6)
        root_layout.addWidget(self.loading_label)
        root_layout.addStretch(1)

        self.animations = []

    def center_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + (geo.height() - self.height()) // 2
            self.move(x, y)

    def start_animation(self, on_finished):
        self.center_on_screen()
        self.show()

        end_rect = self.geometry()
        start_w = int(end_rect.width() * 0.88)
        start_h = int(end_rect.height() * 0.88)
        start_x = end_rect.center().x() - start_w // 2
        start_y = end_rect.center().y() - start_h // 2
        start_rect = QRect(start_x, start_y, start_w, start_h)

        zoom_in = QPropertyAnimation(self, b"geometry", self)
        zoom_in.setDuration(500)
        zoom_in.setStartValue(start_rect)
        zoom_in.setEndValue(end_rect)
        zoom_in.setEasingCurve(QEasingCurve.Type.OutBack)

        self.animations = [zoom_in]
        zoom_in.start()

        QTimer.singleShot(250, self.flash_label.show)
        QTimer.singleShot(2000, lambda: self.finish(on_finished))

    def finish(self, on_finished):
        self.close()
        on_finished()

class ClassAugChanger(QMainWindow):
    progress_signal = pyqtSignal(int)       # 추가
    done_signal = pyqtSignal()              # 추가

    def __init__(self):
        super().__init__()
        self.progress_signal.connect(self.progress_bar_update)
        self.initUI()
        self.aug_list = []
        self.folder_name_clahe = str("aug_clahe")
        self.folder_name_hs = str("aug_H_Shift")
        self.folder_name_vs = str("aug_V_Shift")
        self.folder_name_hf = str("aug_H_Flip")
        self.folder_name_vf = str("aug_V_Flip")
        self.folder_name_rt = str("aug_Rotation")
        self.folder_name_clahe = str("aug_clahe")
        self.folder_rbc = str("aug_RowByColumn")
        self.num_shift = [*range(1, 10, 2)]
        self.num_rot = [*range(90, 360, 90)]        # 90, 180, 270도 회전
        self.output_folder_rgb = "output_rgb_images"
        self.output_folder_bgr = "output_bgr_images"
        self.folder_gray = "gray_scale_images"
        self.folder_color = "color_images"
        # 09.16
        self.disable_all_buttons_initial()      # 모든 버튼 비활성화(라이센스 확인 전)

    # 09.16 _ 라이센스 확인 버튼만 활성화, 나머지 모두 비활성화
    def disable_all_buttons_initial(self):
        for btn in [
            self.set_location, self.bmp2jpg_button, self.check_all_button, self.label_map_button,
            self.change_button, self.change_class_name_button, self.hf_button, self.hs_button,
            self.rot_button, self.vf_button, self.vs_button, self.clahe_button,
            self.xml_transform_button, self.mm003_button, self.RowByColumn_button,
            self.colortogray_button, self.rgb_button, self.bgr_button,
            self.run_train, self.check_train_result_button, self.pbmodel_button
        ]:
            btn.setEnabled(False)
        self.get_license.setEnabled(True)  # 라이센스 확인 버튼만 활성화

    # 입력커서(클래스명)
    def on_text_edited(self, text):
        if not text.startswith(self.prefix):
            cursor_pos = self.input_delete_classsname.cursorPosition()
            new_text = self.prefix + text[len(self.prefix):] if len(text) > len(self.prefix) else self.prefix
            self.input_delete_classsname.setText(new_text)
            self.input_delete_classsname.setCursorPosition(max(cursor_pos, len(self.prefix)))
        elif len(text) < len(self.prefix):
            self.input_delete_classsname.setText(self.prefix)
            self.input_delete_classsname.setCursorPosition(len(self.prefix))

    def on_text_edited_1(self, text):
        if not text.startswith(self.prefix_1):
            cursor_pos = self.input_new_classsname.cursorPosition()
            new_text = self.prefix_1 + text[len(self.prefix_1):] if len(text) > len(self.prefix_1) else self.prefix_1
            self.input_new_classsname.setText(new_text)
            self.input_new_classsname.setCursorPosition(max(cursor_pos, len(self.prefix_1)))
        elif len(text) < len(self.prefix_1):
            self.input_new_classsname.setText(self.prefix_1)
            self.input_new_classsname.setCursorPosition(len(self.prefix_1))

    # 입력커서(행렬 변환)_06.18
    def on_row_edited(self, text):
        if not text.startswith(self.prefix_row):
            cursor_pos = self.row_input.cursorPosition()
            new_text = self.prefix_row + text[len(self.prefix_row):] if len(text) > len(self.prefix_row) else self.prefix_row
            self.row_input.setText(new_text)
            self.row_input.setCursorPosition(max(cursor_pos, len(self.prefix_row)))
        elif len(text) < len(self.prefix_row):
            self.row_input.setText(self.prefix_row)
            self.row_input.setCursorPosition(len(self.prefix_row))

    def on_col_edited(self, text):
        if not text.startswith(self.prefix_col):
            cursor_pos = self.column_input.cursorPosition()
            new_text = self.prefix_col + text[len(self.prefix_col):] if len(text) > len(self.prefix_col) else self.prefix_col
            self.column_input.setText(new_text)
            self.column_input.setCursorPosition(max(cursor_pos, len(self.prefix_col)))
        elif len(text) < len(self.prefix_col):
            self.column_input.setText(self.prefix_col)
            self.column_input.setCursorPosition(len(self.prefix_col))

    # 06.19 (ckpt)
    def on_ckpt_edited(self, text):
        if not text.startswith(self.prefix_ckpt):
            cursor_pos = self.enter_chkp.cursorPosition()
            new_text = self.prefix_ckpt + text[len(self.prefix_ckpt):] if len(text) > len(self.prefix_ckpt) else self.prefix_ckpt
            self.enter_chkp.setText(new_text)
            self.enter_chkp.setCursorPosition(max(cursor_pos, len(self.prefix_ckpt)))
        elif len(text) < len(self.prefix_ckpt):
            self.enter_chkp.setText(self.prefix_ckpt)
            self.enter_chkp.setCursorPosition(len(self.prefix_ckpt))

    # 입력란에서 실제 클래스명만 추출
    def get_real_classname(self):
        text = self.input_delete_classsname.text()
        if text.startswith(self.prefix):
            return text[len(self.prefix):]
        return ""

    def get_real_classname_1(self):
        text = self.input_new_classsname.text()
        if text.startswith(self.prefix_1):
            return text[len(self.prefix_1):]
        return ""

    # 입력란에서 실제 행렬 변환 행, 열 숫자만 추출
    def get_row_number(self):
        text = self.row_input.text()
        if text.startswith(self.prefix_row):
            return text[len(self.prefix_row):]
        return ""

    def get_col_number(self):
        text = self.column_input.text()
        if text.startswith(self.prefix_col):
            return text[len(self.prefix_col):]
        return ""

    # 06.19 (ckpt)
    def get_ckpt_number(self):
        text = self.enter_chkp.text()
        if text.startswith(self.prefix_ckpt):
            return text[len(self.prefix_ckpt):]
        return ""


    def initUI(self):
        self.setWindowTitle('SVT model prep')
        self.setGeometry(100, 100, 800, 400)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.setStyleSheet("""
            QWidget {
                background-color: #FFFBE6;
            }

            QPushButton {
                background-color: #DDECCC;
                color: #4E7D5B;
                border: 1px solid #BBD4B3;
                border-radius: 8px;
                padding: 6px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #CFE3B9;
            }

            QPushButton:pressed {
                background-color: #BBD4B3;
            }

            QProgressBar {
                border: 2px solid #D6D6D6;
                border-radius: 5px;
                background-color: #FAFDEB;
            }

            QProgressBar::chunk {
                background-color: #C5DDB3;
            }

            QLineEdit {
                background-color: #FFFBE6;
                color: #4E7D5B;
                border: 1px solid #BBD4B3;
                border-radius: 6px;
                padding: 4px;
            }

            QLineEdit::placeholder {
                color: #556B2F;
            }        
        """)

        # 06.25 추가
        # (1) annos 폴더 경로 설정
        annos_dir = r"D:\AI_SVT_Training_mk\annotations\annos"
        
        # (2) 이미지 파일 개수 세기
        try:
            num_images = len([
                f for f in os.listdir(annos_dir)
                if os.path.isfile(os.path.join(annos_dir, f)) and f.lower().endswith(('.jpg','.jpeg','.png','.bmp')) 
            ])
        except (OSError, FileNotFoundError):
            num_images = 0
            self.num_images = num_images

        self.num_images = num_images
        self.update_epochno_file()

        # (3) EpochNo.txt 초기 생성(기본 x20)
        annotation_dir = r"D:\AI_SVT_Training_mk\annotations"
        os.makedirs(annotation_dir, exist_ok=True)
        epoch_file = os.path.join(annotation_dir, "EpochNo.txt")
        # 06.26 한줄 추가
        if not os.path.exists(epoch_file):
            with open(epoch_file, "w", encoding="utf-8") as f:
                f.write(str(self.num_images * 20))

        # (4) EpochNo.txt 읽어서 클램핑
        try:
            with open(epoch_file, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            iMaxTrainStep = int(raw)
        except (ValueError, OSError):
            # 파일 이상 시 기본 mul = 20
            iMaxTrainStep = self.num_images * 20
            
        iMaxTrainStep = min(iMaxTrainStep, 300000)
        iMaxTrainStep = max(iMaxTrainStep, 10000)
        train_steps = iMaxTrainStep     # 06.25
        self.train_steps = train_steps      # 06.25

        print(f"[TRAIN_STEPS] {train_steps}", flush=True)       # 06.25

        layout = QVBoxLayout(central_widget)
        self.main_layout = layout
        
        # XLA 위젯 선언 (레이아웃 추가는 나중에)
        self.rb_xla_on = QRadioButton("On")
        self.rb_xla_off = QRadioButton("Off")
        self.rb_xla_on.setChecked(True) # 기본값을 ON으로 변경

        self.btn_xla = QPushButton("XLA: ON")
        self.btn_xla.setObjectName("btn_xla")
        self.btn_xla.setCheckable(True)
        self.btn_xla.setToolTip("XLA (JIT) 활성화")

        class_buttons_layout = QHBoxLayout()
        aug_layout = QGridLayout()
        input_layout = QHBoxLayout()
        rowNcol_layout = QHBoxLayout()
        img_conversion_layout = QHBoxLayout()
        mac_address_layout = QHBoxLayout()		# 09.15
        train_layout = QHBoxLayout()
        inference_layout = QHBoxLayout()

        # 09.15 _라이센스
        # 1안
        self.label_license = QLabel("SVAIS\n(Smart Vision Artificial Intelligence Solution)")
        font = self.label_license.font()
        font.setPointSize(17)
        font.setBold(True) 
        self.label_license.setFont(font)
        self.label_license.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.label_license.setStyleSheet("""
            QLabel {
                font-weight: 600;
                line-height: 1.2;
            }
        """)
        layout.addWidget(self.label_license)
        # 09.15 _라이센스
        # 09.16 _라이센스 확인 버튼 + MAC address + 시리얼번호를 한 줄에 배치
        mac_address_layout = QHBoxLayout()
        self.get_license = QPushButton('라이센스 확인', self)
        self.get_license.clicked.connect(self.get_last_8_mac)
        mac_address_layout.addWidget(self.get_license)      # 09.16
        
        # 09.15 _라이센스    
        self.label_license_check = QLabel("MAC address")
        self.label_license_check.setStyleSheet("color: green; font-weight: bold;")
        mac_address_layout.addWidget(self.label_license_check)
        # layout.addLayout(mac_address_layout)

        # 09.15 _라이센스 시리얼 번호 확인
        self.serial_label = QLabel("시리얼번호: --------")
        font_small = self.serial_label.font()
        font_small.setPointSize(9)
        self.serial_label.setFont(font_small)
        self.serial_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.serial_label.setStyleSheet("color: 	#006400;")       # #ff3399 핑크색 
        mac_address_layout.addStretch(1)    # 오른쪽 여백 09.16
        mac_address_layout.addWidget(self.serial_label)      # 09.16
        layout.addLayout(mac_address_layout)   # 09.16

        # 11.18
        self.btn_manual = QPushButton("Help", self)
        self.btn_manual.clicked.connect(self.show_manual)
        layout.addWidget(self.btn_manual)   

        # Help 버튼 크기 조정
        same_width = max(self.get_license.sizeHint().width(),
                         self.btn_manual.sizeHint().width())
        same_height = self.get_license.sizeHint().height()
        self.get_license.setFixedWidth(same_width)
        self.btn_manual.setFixedWidth(same_width)
        self.btn_manual.setFixedHeight(same_height)

        self.iKeyInfoOrg = uuid.getnode()
        self.iKeyInfoOrg_str = str(self.iKeyInfoOrg)[-8:]
        self.last_8_digits = self.iKeyInfoOrg_str.zfill(8)  # 8자리로 맞추기 위해 앞에 0 추가
        self.last_8_digits = int(str(self.iKeyInfoOrg)[-8:])
        self.serial_label.setText("시리얼번호:" + str(self.last_8_digits))
        

        # 1. 06.25 _annos 읽어오기
        self.annos_dir = r"D:\AI_SVT_Training_mk\annotations\annos"
        # 2. QFileSystemWatcher 생성 및 연결
        self.watcher = QFileSystemWatcher([self.annos_dir], parent=self)
        self.watcher.directoryChanged.connect(self.on_annos_changed)

        train_result_path = r"D:\AI_SVT_Training_mk\train_result\checkpoint"
        self.train_result_path = train_result_path.replace("\\", "/")
        config_path = r"D:\AI_SVT_Training_mk\configs\faster_rcnn_resnet50_v1_800x1333_batch1.config"
        self.config_file_path = config_path.replace("\\", "/")
        path = r"D:\AI_SVT_Training_mk\annotations\annos"
        labelmap_path = os.path.join(r"D:\AI_SVT_Training_mk\annotations\label_map.pbtxt")
        self.labelmap_path = labelmap_path.replace("\\", "/")

        self.label_check = QLabel("Check Training Files")
        font = self.label_check.font()
        font.setPointSize(15)
        self.label_check.setFont(font)
        self.label_check.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.label_check)
        # 👻🐸🐥🐣🦄🍀🌸🥑
        # 버전 변경한거 확인용
        # 오른쪽 정렬
        self.label_check_version = QLabel("🐸SVT_Training_Tool🐸")
        font_small = self.label_check_version.font()
        font_small.setPointSize(9)
        self.label_check_version.setFont(font_small)
        self.label_check_version.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.label_check_version.setStyleSheet("color: 	#006400;")       # #ff3399 핑크색 
        layout.addWidget(self.label_check_version)

        self.set_location = QPushButton('폴더 선택', self)
        self.set_location.clicked.connect(self.directory)
        layout.addWidget(self.set_location)

        self.bmp2jpg_button = QPushButton('.bmp to .jpg', self)
        self.bmp2jpg_button.clicked.connect(self.bmptojpg)
        layout.addWidget(self.bmp2jpg_button)

        self.check_all_button = QPushButton('파일 확인', self)
        self.check_all_button.clicked.connect(self.checkAllFiles)
        class_buttons_layout.addWidget(self.check_all_button)

        self.label_map_button = QPushButton('라벨 맵 생성', self)
        self.label_map_button.clicked.connect(self.label_map)
        class_buttons_layout.addWidget(self.label_map_button)

        self.change_button = QPushButton('클래스 번호 변경', self)
        self.change_button.clicked.connect(self.changeClassNum)
        class_buttons_layout.addWidget(self.change_button)
        layout.addLayout(class_buttons_layout)

        self.info_label = QLabel(self)
        self.info_label.setStyleSheet("background-color: rgba(255, 255, 255, 150); padding: 5px")
        layout.addWidget(self.info_label)

        self.input_delete_classsname = QLineEdit(self)
        self.prefix = "기존 클래스명: "
        self.input_delete_classsname.setText(self.prefix)
        input_layout.addWidget(self.input_delete_classsname)

        self.input_delete_classsname.textEdited.connect(self.on_text_edited)
        
        self.input_new_classsname = QLineEdit(self)
        self.prefix_1 = "변경할 클래스명: "
        self.input_new_classsname.setText(self.prefix_1)
        input_layout.addWidget(self.input_new_classsname)
        layout.addLayout(input_layout)

        self.input_new_classsname.textEdited.connect(self.on_text_edited_1)
        
        # 클래스명 변경 버튼과 XLA 버튼을 한 줄에 배치
        change_class_layout = QHBoxLayout()
        self.change_class_name_button = QPushButton("클래스명 변경")
        self.change_class_name_button.clicked.connect(self.changeClassName)
        change_class_layout.addWidget(self.change_class_name_button)
        change_class_layout.addWidget(self.btn_xla) # XLA 버튼 추가
        layout.addLayout(change_class_layout) # QHBoxLayout을 메인 레이아웃에 추가

        self.aug_check = QLabel("Choose Image Augmentation Options")
        font = self.aug_check.font()
        font.setPointSize(15)
        self.aug_check.setFont(font)
        self.aug_check.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.aug_check)

        self.clear_aug_label = QLabel("Augmentation Info")
        layout.addWidget(self.clear_aug_label)

        self.hf_button = QPushButton('수평 뒤집기(H Flip)')
        self.hf_button.clicked.connect(self.imgHflip)
        self.hf_button.clicked.connect(self.xmlHflip)
        aug_layout.addWidget(self.hf_button, 0, 0)

        self.hs_button = QPushButton('수평 이동(H Shift)')
        self.hs_button.clicked.connect(self.imgHshift)
        self.hs_button.clicked.connect(self.xmlHshift_all)
        aug_layout.addWidget(self.hs_button, 0, 1)

        self.rot_button = QPushButton('회전(Rotation)')
        self.rot_button.clicked.connect(self.imgRotation)
        self.rot_button.clicked.connect(self.xmlRotation_all)
        aug_layout.addWidget(self.rot_button, 0, 2)

        self.vf_button = QPushButton('수직 뒤집기(V Flip)')
        self.vf_button.clicked.connect(self.imgVflip)
        self.vf_button.clicked.connect(self.xmlVflip)
        aug_layout.addWidget(self.vf_button, 1, 0)

        self.vs_button = QPushButton('수직 이동(V Shift)')
        self.vs_button.clicked.connect(self.imgVshift)
        self.vs_button.clicked.connect(self.xmlVshift_all)
        aug_layout.addWidget(self.vs_button, 1, 1)

        self.clahe_button = QPushButton('CLAHE')
        self.clahe_button.clicked.connect(self.clahe_aug)
        aug_layout.addWidget(self.clahe_button, 1, 2)
        layout.addLayout(aug_layout)

        # xml 변경 버튼 추가 (하나의 xml 파일에 object 추가) _PJS
        self.xml_transform_button = QPushButton("XML 객체 추가(단일 이미지에 복수 객체 추가)", self)
        self.xml_transform_button.clicked.connect(self.transform_xml_boxes)
        layout.addWidget(self.xml_transform_button)   

        # Auto Label XML 버튼 추가 (JPG, XML 따로따로 생성) _PJS (실제 훈련에 들어갈 jpg, xml 파일 생성 버튼)
        self.mm003_button = QPushButton('Auto Label XML 조정(복수 이미지에 복수 파일 추가)', self)
        self.mm003_button.clicked.connect(self.process_xml_variants)
        layout.addWidget(self.mm003_button)

        self.prefix_row = "행 번호: "
        self.prefix_col = "열 번호: "


        self.row_input = QLineEdit(self)
        self.row_input.setText(self.prefix_row)
        rowNcol_layout.addWidget(self.row_input)
        self.row_input.textEdited.connect(self.on_row_edited)

        self.column_input = QLineEdit(self)
        self.column_input.setText(self.prefix_col)
        rowNcol_layout.addWidget(self.column_input)
        layout.addLayout(rowNcol_layout)
        self.column_input.textEdited.connect(self.on_col_edited)

        self.RowByColumn_button = QPushButton('행렬 변환(Row by Column)')
        self.RowByColumn_button.clicked.connect(self.RowColumn_all)
        layout.addWidget(self.RowByColumn_button)

        self.colortogray_button = QPushButton('Color to Gray')
        self.colortogray_button.clicked.connect(self.ColorToGray)
        img_conversion_layout.addWidget(self.colortogray_button)

        self.rgb_button = QPushButton('BGR to RGB')
        self.rgb_button.clicked.connect(self.BGRtoRGB)
        img_conversion_layout.addWidget(self.rgb_button)

        self.bgr_button = QPushButton('RGB to BGR')
        self.bgr_button.clicked.connect(self.RGBtoBGR)
        img_conversion_layout.addWidget(self.bgr_button)
        layout.addLayout(img_conversion_layout)

        self.train_label = QLabel("Proceed Training")
        font = self.train_label.font()
        font.setPointSize(15)
        self.train_label.setFont(font)
        self.train_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.train_label)

        # ProgressBar 추가_PJS
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('%p%')     # % 포멧
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)    # 숫자 가운데정렬
        layout.addWidget(self.progress_bar)
        self.progress_bar.setMaximum(self.train_steps)  # 06.26

         # 👻🐸🐥🐣🦄🍀🌸🥑

        # 선택 시각 이후에 종료
        shutdown_layout = QHBoxLayout()
        self.shutdown_checkbox = QCheckBox("🥑훈련 종료 후 설정된 시각 이후 PC 종료🥑")
        shutdown_layout.addWidget(self.shutdown_checkbox)

        self.shutdown_time_edit = QTimeEdit(self)
        self.shutdown_time_edit.setDisplayFormat("HH:mm")
        self.shutdown_time_edit.setTime(QTime(00, 0))   # 기본값 00:00
        # self.shutdown_time_edit 크기 조정
        self.shutdown_time_edit.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred))
        self.shutdown_time_edit.setMaximumWidth(100)  # 최대 너비 설정
        self.shutdown_time_edit.setMinimumWidth(80)   # 최소 너비 설정
        # StyleSheet로 디자인 변경
        self.shutdown_time_edit.setStyleSheet("""
            QTimeEdit {
                border: 2px solid #BBD4B3;
                border-radius: 6px;
                padding: 4px;
                background-color: #FFFFFF;
                font-size: 14px;
            }
            QTimeEdit::up-button, QTimeEdit::down-button {
                width: 16px; height: 16px;
                subcontrol-origin: border;
            }
            QTimeEdit:hover {
                background-color: #F1FFF0;
            }
            QTimeEdit:focus {
                border-color: #66BB6A;
            }
        """)   
        shutdown_layout.addWidget(self.shutdown_time_edit)
        layout.addLayout(shutdown_layout) 

        # ProgressBar 색상 변경 << background 여기서 변경 해주면 됨
        self.progress_bar.setStyleSheet("""
        QProgressBar {
            border: 2px solid grey;
            border-radius: 5px;
        }
        QProgressBar::chunk {
            background-color: 	#E6E6FA;   
            width: 20px;
        }
        """)
        
        self.run_train = QPushButton('훈련 시작', self)
        self.run_train.clicked.connect(self.runCommand)
        train_layout.addWidget(self.run_train)
        
        # 버튼 추가 _PJS
        self.check_train_result_button = QPushButton('훈련 파일 결과 확인', self)
        self.check_train_result_button.clicked.connect(self.checkTrainingComplete)
        train_layout.addWidget(self.check_train_result_button)

        self.pbmodel_button = QPushButton('모델 추출', self)
        self.pbmodel_button.clicked.connect(self.pbmodel)
        train_layout.addWidget(self.pbmodel_button)
        layout.addLayout(train_layout)

        self.prefix_ckpt = "체크포인트 입력: "       # 06.19

        self.enter_chkp = QLineEdit(self)
        self.enter_chkp.setText(self.prefix_ckpt)           # 이 부분도 글자 고정시킬거임
        layout.addWidget(self.enter_chkp)

        self.enter_chkp.textEdited.connect(self.on_ckpt_edited)

        self.label_ckpt_status = QLabel("")
        self.label_ckpt_status.setStyleSheet("color: green; font-weight: bold;")    
        layout.addWidget(self.label_ckpt_status)

        # 06.24 _Radiobutton 추가
        radio_layout = QHBoxLayout()

        # 버튼 간격 설정
        radio_layout.setSpacing(20)
        # 레이아웃 바깥 여백 설정
        radio_layout.setContentsMargins(10, 5, 10, 5)  

        # 1) 라디오 버튼 생성
        self.radio10 = QRadioButton("× 10")
        self.radio20 = QRadioButton("× 20")
        self.radio30 = QRadioButton("× 30")
        self.radio20.setChecked(True)  # 기본 20

        # 2) 버튼 그룹에 ID 연결
        self.bg_mul = QButtonGroup(self)
        self.bg_mul.addButton(self.radio10, 10)
        self.bg_mul.addButton(self.radio20, 20)
        self.bg_mul.addButton(self.radio30, 30)

        # 3) 레이아웃에 추가
        radio_layout.addStretch(1)          # 왼쪽 여백
        radio_layout.addWidget(self.radio10)
        radio_layout.addWidget(self.radio20)
        radio_layout.addWidget(self.radio30)
        
        # EpochNo 오른쪽에 XLA 라디오 버튼 추가
        radio_layout.addSpacing(20)
        xla_label = QLabel("XLA:")
        radio_layout.addWidget(xla_label)
        radio_layout.addWidget(self.rb_xla_on)
        radio_layout.addWidget(self.rb_xla_off)

        # 09.16 _svt 로고 추가
        radio_layout.addStretch(1)          # 오른쪽 여백
        logo_label = QLabel(self)
        logo_pixmap = QPixmap("D:/AI_SVT_Training_mk/svt_logo.png")
        logo_label.setPixmap(logo_pixmap)
        logo_label.setScaledContents(True)
        logo_label.setMaximumSize(100, 50)  # 작고 깔끔하게
        logo_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        radio_layout.addWidget(logo_label)

        layout.addLayout(radio_layout)  # layout은 initUI에서 정의된 메인 레이아웃

        # 4) 시그널 연결
        self.bg_mul.buttonClicked.connect(self.on_mul_changed)
        # 09.03 _xla
        self.btn_xla.toggled.connect(lambda s: self.rb_xla_on.setChecked(s) if s else self.rb_xla_off.setChecked(True))
        self.rb_xla_on.toggled.connect(lambda s: self._on_xla_change(s))
        self.rb_xla_off.toggled.connect(lambda s: self._on_xla_change(False) if s else None)

        # 초기상태 동기화 (ON으로)
        self._on_xla_change(True)

        # 06.05 _loss율
        self.label_loss_status = QLabel("실시간 Loss: -")
        layout.addWidget(self.label_loss_status)

    # 09.03 _xla
    def _on_xla_change(self, is_on: bool):
        # 버튼 라벨/체크 상태 동기화
        if hasattr(self, 'btn_xla'):
            self.btn_xla.blockSignals(True)
            self.btn_xla.setChecked(is_on)
            self.btn_xla.setText("XLA: ON" if is_on else "XLA: OFF")
            self.btn_xla.blockSignals(False)
        # 내부 상태 플래그
        self.xla_enabled = bool(is_on)

    # 06.05 _loss율
    def update_loss_label(self, loss, step):
        QTimer.singleShot(0, lambda: self.label_loss_status.setText(f"Step {step} → Loss: {loss:.4f}"))

    # 06.25 _Radiobutton + EpochNo.txt 추가
    def on_mul_changed(self, button):
        mul = self.bg_mul.id(button)    # 0, 1, 2 -> 10, 20, 30
        
        # 매번 annos 이미지 개수 다시 계산
        annos_dir = r"D:\AI_SVT_Training_mk\annotations\annos"
        try:
            num_images = len([
                f for f in os.listdir(annos_dir)
                if os.path.isfile(os.path.join(annos_dir, f)) and
                    f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
            ])
        except (OSError, FileNotFoundError):
            num_images = 0

        # EpocNo.txt 업데이트
        annotations_dir = r"D:\AI_SVT_Training_mk\annotations"
        os.makedirs(annotations_dir, exist_ok=True)
        epoch_file = os.path.join(annotations_dir, "EpochNo.txt")
        with open(epoch_file, "w", encoding="utf-8") as f:
            f.write(str(self.num_images * mul))
        print(f"[EpochNo.txt] {self.num_images}×{mul} → {self.num_images*mul}", flush=True)     # 팝업창 추가
        
        # 06.26 변경된 EpochNo.txt 읽어서 train_steps 에 반영
        try:
            with open(epoch_file, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            iMaxTrainStep = int(raw)
        except (ValueError, OSError):
            iMaxTrainStep = self.num_images * mul
        iMaxTrainStep = min(iMaxTrainStep, 300000)
        iMaxTrainStep = max(iMaxTrainStep, 10000)
        self.train_steps = iMaxTrainStep
        self.progress_bar.setMaximum(self.train_steps)
            
        # 팝업창 추가
        msg = QMessageBox(self)
        msg.setWindowTitle("EpochNo 업데이트")
        msg.setText(f"훈련 스텝이 이미지개수 {self.num_images}개 × {mul} = {self.num_images*mul} 로 설정되었습니다.")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.addButton("확인", QMessageBox.ButtonRole.AcceptRole)
        msg.exec()  # 모달로 실행

        # self.num_images 갱신
        self.num_images  = num_images 


    def pbmodel(self):
        checkpoint_number = self.get_ckpt_number().strip()
        if not checkpoint_number.isdigit():
            QMessageBox.warning(self, "입력 오류", "유효한 숫자를 입력하세요.")
            return
        checkpoint_number = int(int(checkpoint_number)/200)
        path_chkp = self.train_result_path
        if path_chkp:
            try:
                with open(path_chkp, 'r') as file:
                    lines = file.readlines()
                if lines:
                    lines[0] = f'model_checkpoint_path: "ckpt-{checkpoint_number}"\n'
                with open(path_chkp, 'w') as file:
                    file.writelines(lines)
                self.label_ckpt_status.setText(f"체크포인트가 업데이트 되었습니다.: ckpt-{checkpoint_number}")
                self.label_ckpt_status.setStyleSheet("color: green; font-weight: bold;")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"파일 업데이트 중 오류 발생: {e}")
        try:
            cmd_pb_directory = 'D:\\AI_SVT_Training_mk\\2)model_pb.bat'
            if sys.platform.startswith('win'):
                self.cmd_pb = subprocess.Popen(['cmd', '/c', cmd_pb_directory],
                                                    creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                self.cmd_pb = subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', cmd_pb_directory])
        except Exception as e:
            print(f"오류가 발생했습니다.: {e}", flush=True)

        # 버튼 비활성화_05.21 추가
        self.run_train.setEnabled(False)


    def runCommand(self):
        # 10.02
        train_result_dir = r"D:\AI_SVT_Training_mk\train_result"
        if os.path.isdir(train_result_dir):
            QMessageBox.critical(           # 얘는 문자열만 받아야함
                self,
                "훈련불가",
                f"'{train_result_dir}' 폴더가 이미 존재합니다.\n"
                f"기존 결과를 삭제하거나 다른 위치로 이동 후 다시 시도하세요."
            )
            return
        
        # 07.11 _라벨맵과 xml 객체 라벨 비교
        _, xml_classes = self.count_classes_in_folder(self.xml_folder_list)
        # pbtxt 에서 읽어온 라벨명들
        map_labels = set()
        try:
            with open(self.labelmap_path, 'r', encoding= 'utf-8') as f:
                for line in f:
                    m = re.search(r"name\s*[:]\s*['\"](.+?)['\"]", line)
                    if m:
                        map_labels.add(m.group(1))
        except Exception as e:
            QMessageBox.critical(self, "라벨 맵 오류", f"label_map.pbtxt 파일을 열 수 없습니다:\n{e}")
            return
        
        missing = xml_classes - map_labels
        # extra = map_labels - xml_classes
        # if missing or extra:
        if missing :
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("라벨 불일치 오류")
            text = "" 
            if missing:
                text += f"훈련에 필요한 라벨이 label_map.pbtxt에 없습니다:\n{', '.join(missing)}\n\n"
            # if extra:
            #     text += f"label_map.pbtxt에 실제로 안 쓰이는 라벨이 등록되어 있습니다.\n{', '.join(extra)}\n\n"
            text += "라벨 맵을 확인 후 다시 시도해 주세요."    
            msg.setText(text)
            msg.addButton("확인", QMessageBox.ButtonRole.AcceptRole)
            msg.exec()
            return
        
        # 06.25 _라디오 버튼에서 곱하기 값 가져오기
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(self.train_steps)      # 06.26 (100 에서 self.train_steps로 변경)
        self.progress_bar.setFormat("0.00%")  # 초기 포맷 설정
       
        # Run Training File 버튼 누르면 GpuBoost 실행
        try:
            print("[LOG] GPU Boost 실행", flush=True)
            subprocess.run(['nvidia-smi', '-lgc', '2500,2580'], check=True)      
        except Exception as e:
            print(f"[ERROR] GPU Boost 실패: {e}", flush=True)


        # 샷다운 추가_250523 , 설정된 시각으로 추가_250613
        shutdown_flag = self.shutdown_checkbox.isChecked()
        qt = self.shutdown_time_edit.time()
        shutdown_hour = qt.hour()
        shutdown_minute = qt.minute()

        # 변경 : TrainingTread에 시, 분 인자 전달 _250613 
        self.train_thread = TrainingThread(
            shutdown_enable= shutdown_flag,
            shutdown_hour=shutdown_hour,
            shutdown_minute=shutdown_minute,
            # 06.25 추가
            max_steps=self.train_steps,
            enable_xla=self.rb_xla_on.isChecked()   # 09.02 추가
        )

        self.train_thread.progress_signal.connect(self.progress_bar_update)  # 연결
        self.train_thread.done_signal.connect(self.onTrainingDone)  # 06.05 _loss
        self.train_thread.loss_signal.connect(self.update_loss_label)   # 06.05 _실시간 loss
        
        def set_xla_enabled(enabled_status):
            self.btn_xla.setEnabled(enabled_status)
            self.rb_xla_on.setEnabled(enabled_status)
            self.rb_xla_off.setEnabled(enabled_status)

        # 09.02 _예외/비정상 종료 대비(백업 안전장치)
        self.train_thread.finished.connect(lambda: set_xla_enabled(True))

        # 훈련 시작
        self.train_thread.start()
        # 훈련시작 버튼 비활성화
        self.run_train.setEnabled(False)
        # 라디오 버튼 비활성화      # 06.25
        self.radio10.setEnabled(False)
        self.radio20.setEnabled(False)
        self.radio30.setEnabled(False)
        # 09.02 _xla 버튼 비활성화
        set_xla_enabled(False)

        # 07.10 _훈련 시작하면 모든 버튼 비활성화
        self.set_location.setEnabled(False)
        self.bmp2jpg_button.setEnabled(False)
        self.check_all_button.setEnabled(False)
        self.label_map_button.setEnabled(False)
        self.change_button.setEnabled(False)
        self.change_class_name_button.setEnabled(False)
        self.hf_button.setEnabled(False)
        self.hs_button.setEnabled(False)
        self.rot_button.setEnabled(False)
        self.vf_button.setEnabled(False)
        self.vs_button.setEnabled(False)
        self.clahe_button.setEnabled(False)
        self.xml_transform_button.setEnabled(False)
        self.mm003_button.setEnabled(False)
        self.RowByColumn_button.setEnabled(False)
        self.colortogray_button.setEnabled(False)
        self.rgb_button.setEnabled(False)
        self.bgr_button.setEnabled(False)
        self.check_train_result_button.setEnabled(False)
        self.pbmodel_button.setEnabled(False)

    ##### MAC ADDRESS _09.15 #####
    def get_last_8_mac(self):
        self.iKeyInfoOrg = uuid.getnode()
        self.iKeyInfoOrg_str = str(self.iKeyInfoOrg)[-8:]
        self.last_8_digits = self.iKeyInfoOrg_str.zfill(8)  # 8자리로 맞추기 위해 앞에 0 추가
        self.last_8_digits = int(str(self.iKeyInfoOrg)[-8:])
        #self.serial_label.setText("시리얼번호:" + str(self.last_8_digits))
        
        print(f"MAC Address: {self.iKeyInfoOrg_str}", flush=True)
        print(f"MAC Address: {self.iKeyInfoOrg}", flush=True)

        head = int(self.iKeyInfoOrg_str) // 10000
        tail = int(self.iKeyInfoOrg_str) % 10000
        # Compute license key
        self.license_key = (tail * 3.14) + (head * 3.14)
        # Construct the file path
        file_path = f"D:/svtdata/{self.iKeyInfoOrg_str}.lic"

        # Check if the file exists
        if os.path.exists(file_path):
            with open(file_path, "rb") as binary_file:
                # Read the first 4 bytes
                binary_data = binary_file.read(8)
                #print("Binary date:", binary_data)
                self.float_value = struct.unpack('<d', binary_data)[0]
                if self.float_value == self.license_key:
                    # self.run_train.setEnabled(True)
                    # self.set_location.setEnabled(True)
                    self.label_license_check.setText('Licensed')
                    # 09.16 _라이센스 통과 시 폴더 선택 버튼만 활성화
                    self.set_location.setEnabled(True)
                else:
                    self.label_license_check.setText('No license')
                    # self.run_train.setEnabled(False)
                    self.set_location.setEnabled(False)
                    self.disable_all_buttons_initial()
        else:
            self.label_license_check.setText('No license')
            # self.run_train.setEnabled(False)
            self.set_location.setEnabled(False)
            self.disable_all_buttons_initial()
            QMessageBox.warning(self, "라이센스 오류", f"라이센스 파일이 없습니다.\n라이센스 파일의 경로를 확인하세요.")

        return float(self.license_key)

    #     # pdf_path = os.path.dirname(os.path.abspath("FlowChart.pdf"))
    #     pdf_path = os.path.join(base_dir, "FlowChart.pdf")

    #     if not os.path.exists(pdf_path):
    #         QMessageBox.warning(
    #             self,
    #             "파일 없음",
    #             f"FlowChart.pdf 파일을 찾을 수 없습니다.\n"
    #             f"경로를 확인해주세요/\n\n현재 시도한 경로:\n{pdf_path}"
    #         )
    #         return

    #     try:
    #         # 윈도우에서는 기본 pdf 뷰어로 열기
    #         if sys.platform.startswith("win"):
    #             os.startfile(pdf_path)
    #         else:
    #             # 기타 OS에서는 QDesktopServide로 열기
    #             QDesktopServices.openUrl(QUrl.fromLocalFile(pdf_path))
    #     except Exception as e:
    #         QMessageBox.critical(
    #             self,
    #             "열기 오류",
    #             f"PDF를 여는 중 오류가 발생했습니다.\n\n{e}"
    #         )

    # 11.25 _ help 연동 (네트워크 경로까지 검색 버전)
    def show_manual(self):
        try:
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        except Exception:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        filename = "FlowChart.pdf"

        # 1) 가장 먼저 확인할 후보 경로들 (정확한 파일 경로)
        candidates = []

        # exe / py 위치
        candidates.append(os.path.join(base_dir, filename))

        # 사용자가 선택한 폴더 안
        if hasattr(self, "folder_path"):
            candidates.append(os.path.join(self.folder_path, filename))

        # 기본 프로젝트 폴더
        candidates.append(r"D:\AI_SVT_Training_mk\FlowChart.pdf")

        # 2) 네트워크 / 서버 루트 경로 목록 (회사 NAS 등)
        #    -> 여기 리스트에 원하는 서버 경로를 더 추가해도 됨
        network_roots = [
            r"\\svtechnas\SVT_Project",
            r"\\aisvt\AI_RESOURCE",
            # r"\\svtechnas\다른경로",  # 필요하면 추가
        ]

        # 3) 로컬 드라이브 루트
        local_roots = [
            "C:\\",
            "D:\\",
            # "E:\\", "F:\\",  # 필요시 추가
        ]

        # --- 우선순위 경로 먼저 확인 ---
        for path in candidates:
            if os.path.exists(path):
                self._open_pdf(path)
                return

        # --- 네트워크 루트들에서 검색 ---
        found = self._search_pdf_in_roots(network_roots, filename)
        if found:
            self._open_pdf(found)
            return

        # --- 로컬 드라이브 전체 검색 ---
        found = self._search_pdf_in_roots(local_roots, filename)
        if found:
            self._open_pdf(found)
            return

        # --- 최종 실패 ---
        QMessageBox.warning(
            self,
            "파일 없음",
            f"PC 및 지정된 서버 경로에서 '{filename}' 파일을 찾을 수 없습니다.\n"
            "FlowChart.pdf 위치를 확인해주세요."
        )

    def _search_pdf_in_roots(self, roots, filename="FlowChart.pdf"):
        """주어진 루트 목록(로컬 드라이브, 네트워크 서버 등)을 돌면서
        filename 을 찾으면 경로를 반환. 없으면 None.
        """
        for root_base in roots:
            # 네트워크 경로/드라이브가 실제로 접근 가능한지 먼저 확인
            if not os.path.exists(root_base):
                continue
            try:
                for root, dirs, files in os.walk(root_base):
                    if filename in files:
                        return os.path.join(root, filename)
            except Exception as e:
                # 접근 권한 실패, 네트워크 오류 등은 건너뛰기
                print(f"[WARN] 경로 탐색 실패: {root_base} ({e})", flush=True)
                continue
        return None

    def _open_pdf(self, pdf_path):
        try:
            if sys.platform.startswith("win"):
                os.startfile(pdf_path)
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(pdf_path))
        except Exception as e:
            QMessageBox.critical(
                self,
                "열기 오류",
                f"PDF를 여는 중 오류가 발생했습니다.\n\n{e}"
            )


    # 06.25 추가
    def progress_bar_update(self, step):
        # ProgBar 값 설정
        # self.progress_bar.setValue(step)
        # percent = step / self.train_steps * 100
        # 06.30 1. raw step 이 최대 스텝 넘지 않게 자르기
        raw = min(step, self.train_steps)
        self.progress_bar.setValue(raw)
        # 2. 퍼센트가 100% 이상으로 올라가지 않게 _ 1,2번만 06.30 추가
        percent = raw / self.train_steps * 100
        self.progress_bar.setFormat(f"{percent:.2f}%")

    # 06.05 _loss율
    def onTrainingDone(self, best_info):
        best_loss, best_step = best_info
        QMessageBox.information(self, "훈련 완료", f"모델 훈련이 완료됐습니다.\n\n"
                                                f"최적 Loss : {best_loss:.4f}\n"
                                                f"해당 Step : {best_step}")
        # --- 팝업 OK 누른 직후 프로그레스바를 0%로 리셋 _07.01
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0.00%")

        # 훈련 끝나면 다시 버튼 활성화
        self.run_train.setEnabled(True)
        # 06.25 라디오버튼 활성화 추가
        self.radio10.setEnabled(True)
        self.radio20.setEnabled(True)
        self.radio30.setEnabled(True)
        # 09.02 _xla 버튼 활성화
        self.btn_xla.setEnabled(True)
        self.rb_xla_on.setEnabled(True)
        self.rb_xla_off.setEnabled(True)

        # 07.10 _버튼 활성화
        self.set_location.setEnabled(True)
        self.bmp2jpg_button.setEnabled(True)
        self.check_all_button.setEnabled(True)
        self.label_map_button.setEnabled(True)
        self.change_button.setEnabled(True)
        self.change_class_name_button.setEnabled(True)
        self.hf_button.setEnabled(True)
        self.hs_button.setEnabled(True)
        self.rot_button.setEnabled(True)
        self.vf_button.setEnabled(True)
        self.vs_button.setEnabled(True)
        self.clahe_button.setEnabled(True)
        self.xml_transform_button.setEnabled(True)
        self.mm003_button.setEnabled(True)
        self.RowByColumn_button.setEnabled(True)
        self.colortogray_button.setEnabled(True)
        self.rgb_button.setEnabled(True)
        self.bgr_button.setEnabled(True)
        self.check_train_result_button.setEnabled(True)
        self.pbmodel_button.setEnabled(True)

    def checkTrainingComplete(self):            # 버튼 추가에 관한 내용_PJS
        checkpoint_path = os.path.join(r"D:\\AI_SVT_Training_mk\\train_result")
        if os.path.isdir(checkpoint_path) and os.listdir(checkpoint_path):
            QMessageBox.information(self, "훈련 완료", "훈련이 완료됐습니다.\ntrain_result 폴더가 존재합니다.")
        else:
            QMessageBox.warning(self, "훈련 완료X", "아직 훈련이 완료되지 않았습니다.")

    def directory(self):
        options = QFileDialog.Option(QFileDialog.Option.ReadOnly)
        self.chosen_directory = QFileDialog.getExistingDirectory(self, "Open Directory", "", options=options)
        if self.chosen_directory:
            self.folder_path = self.chosen_directory.replace("\\", '/')     
            # 09.16
            self.enable_all_buttons_after_folder()

            # 07.15
            all_imgs = [
                os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path)
                if os.path.isfile(os.path.join(self.folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]
            all_xmls = [
                os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path)
                if os.path.isfile(os.path.join(self.folder_path, f)) and f.lower().endswith('.xml')
            ]

            self.img_folder_list = all_imgs
            self.xml_folder_list = all_xmls

            if not hasattr(self, 'original_img_list'):
                pattern = re.compile(r'_(HF|VF|HS\d+|VS\d+|clahe|arr|expand|shrink|left|right|up|down|RT\d+)\.', re.IGNORECASE)
                self.original_img_list = [
                    os.path.join(self.folder_path, f) for f in all_imgs
                    if not pattern.search(f)
                ]
                self.original_xml_list = [
                    os.path.join(self.folder_path, f) for f in all_xmls
                    if not pattern.search(f)  # 원본 XML 필터링
                ]
                self.refresh_original_list()

    # 09.16 
    def enable_all_buttons_after_folder(self):
        # 폴더 선택 후 모든 기능 버튼 활성화
        for btn in [
            self.run_train, self.bmp2jpg_button, self.check_all_button,
            self.label_map_button, self.change_button, self.change_class_name_button,
            self.hf_button, self.hs_button, self.rot_button, self.vf_button,
            self.vs_button, self.clahe_button, self.xml_transform_button,
            self.mm003_button, self.RowByColumn_button, self.colortogray_button,
            self.rgb_button, self.bgr_button, self.check_train_result_button,
            self.pbmodel_button
        ]:
            btn.setEnabled(True)

    def refresh_original_list(self):
        pattern = re.compile(r'_(HF|VF|HS\d+|VS\d+|clahe|arr|expand|shrink|left|right|up|down|RT\d+)\.', re.IGNORECASE)
        all_imgs = [
            os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path)
            if os.path.isfile(os.path.join(self.folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        all_xmls = [
            os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path)
            if os.path.isfile(os.path.join(self.folder_path, f)) and f.lower().endswith('.xml')
        ]
        self.original_img_list = [f for f in all_imgs if not pattern.search(os.path.basename(f))]
        self.original_xml_list = [f for f in all_xmls if not pattern.search(os.path.basename(f))]


    def bmptojpg(self):
        src_path = self.folder_path
        dst_path = os.path.join(src_path, "jpg_images/")  # jpg images path
        if not os.path.isdir(dst_path):  # make dst dir if it's not existed
            os.mkdir(dst_path)
        bmp_path = os.path.join(src_path, 'bmp_images/')
        os.makedirs(bmp_path, exist_ok=True)
        src_path_bmp = (img for img in set(glob.glob(src_path + "/*.bmp" or src_path + "/*.BMP")))
        # bmpfiles=[img for img in set(glob.glob(src_path+"/*.bmp" or src_path+"/*.BMP"))]

        for img in tqdm(src_path_bmp, desc='iterate list'):
            time.sleep(0.1)
            images = PILImage.open(img)
            name = img.split("\\")[-1]
            name = name.split(".")[0] + ".jpg"
            new_name = os.path.join(dst_path, name)
            new_name = new_name.replace("\\", "/")
            # images.save(new_name) # 원래 이거임
            images.convert("RGB").save(new_name)    # 07.21 포토샵 사진 jpg변경 안돼서 추가
            images.close()
        for img in set(glob.glob(src_path + "/*.bmp" or src_path + "/*.BMP")):
            img = img.replace("\\", "/")
            print(f"file:{img}", flush=True)
            shutil.move(img, bmp_path)

    def changeClassName(self):
        original_fileName = str(self.input_delete_classsname.text())
        modi_fileName = str(self.input_new_classsname.text())

        prefix = "기존 클래스명: "
        if original_fileName.startswith(prefix):
            original_fileName = original_fileName[len(prefix):]
        else:
            original_fileName = original_fileName
        prefix_1 = "변경할 클래스명: "
        if modi_fileName.startswith(prefix_1):
            modi_fileName = modi_fileName[len(prefix_1):]
        else:
            modi_fileName = modi_fileName
        for xml_file in self.xml_folder_list:
            targetXML = open(xml_file, 'rt', encoding='utf-8')
            tree = ET.parse(targetXML)
            root = tree.getroot()
            for obj in root.iter('object'):
                class_name = obj.find('name')
                original = class_name.text
                modified = original.replace(original_fileName, modi_fileName)
                class_name.text = modified
            tree.write(xml_file)
        self.info_label.setText(f'기존 클래스 {original_fileName} 이(가) {modi_fileName}로 변경되었습니다.')

    def checkAllFiles(self):
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.xml'):
                xml_path = os.path.join(self.folder_path, filename)
                image_path = os.path.join(self.folder_path, os.path.splitext(filename)[
                    0] + '.jpg')
                if not os.path.exists(image_path):
                    os.remove(xml_path)
            elif filename.endswith('.jpg'):
                image_path = os.path.join(self.folder_path, filename)
                xml_path = os.path.join(self.folder_path, os.path.splitext(filename)[0] + '.xml')
                if not os.path.exists(xml_path):
                    os.remove(image_path)
        for xml_file in self.xml_folder_list:
            targetXML = open(xml_file, 'rt', encoding='utf-8')
            tree = ET.parse(targetXML)
            root = tree.getroot()
            target_tag = root.find("object")
            if target_tag is None:
                targetXML.close()
                os.remove(xml_file)
        self.info_label.setText('비어있는 jpg, xml 파일이 삭제되었습니다.')

    def label_map(self):
        self.num_classes, self.class_names = self.count_classes_in_folder(self.xml_folder_list)
        self.info_label.setText(f'Number of classes: {self.num_classes}. Class names: {self.class_names}')
        with open(self.labelmap_path, "w", encoding='utf-8') as a:
            for i, class_name in enumerate(self.class_names, 1):
                self.pbtxt_content = f"item {{ \n id: {i}\n name:'{class_name}'\n display_name:'{class_name}'\n}}\n"
                a.write(self.pbtxt_content)

    def count_classes_in_folder(self, xml_folder_list):
        class_set = set()
        for xml_file in xml_folder_list:
            classes = self.extract_classes_from_xml(xml_file)
            class_set.update(classes)
        return len(class_set), class_set

    def extract_classes_from_xml(self, xml_file):
        classes = set()
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('.//object'):
            class_name = obj.find('name').text
            classes.add(class_name)
        return classes

    def changeClassNum(self):
        maxid = self.maxid()
        with open(self.config_file_path, "r") as config_file:
            config_content = config_file.read()
        pattern = r"num_classes:\d+"
        new_config_content = re.sub(pattern, f"num_classes:{maxid}", config_content)
        with open(self.config_file_path, "w") as config_file:
            config_file.write(new_config_content)
        self.info_label.setText(f"최대ID로 config 파일 구성이 업데이트 되었습니다.:{maxid}")

    def maxid(self):
        self.max_id = -1
        id_pattern = re.compile(r"id:\s*(\d+)")
        with open(self.labelmap_path, 'r') as labelmap:
            for line in labelmap:
                match = id_pattern.search(line)
                if match:
                    id_value = int(match.group(1))
                    if id_value > self.max_id:
                        self.max_id = id_value
        return self.max_id

############################# CLAHE ###########################
    def clahe_aug(self):
        pattern = re.compile(r'_(HF|VF|HS\d+|VS\d+|clahe|arr|expand|shrink|left|right|up|down|RT\d+)\.', re.IGNORECASE)
        original_img_list = [ f for f in self.original_img_list if not pattern.search(f) ]  # 원본 이미지 필터링
        # 05.07 추가
        total_files = len(original_img_list)  # original_img_list 변경 07.15
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        
        count = 0
        for i, img in enumerate(original_img_list):    # original_img_list 변경 07.15
            image_path = os.path.join(self.folder_path, img)
            image = cv2.imread(image_path, 0)
            base_name = os.path.basename(img).split('.')[0]
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
            cll = clahe.apply(image)
            os.makedirs(os.path.join(self.folder_path, self.folder_name_clahe), exist_ok=True)
            output_path = os.path.join(self.folder_path, self.folder_name_clahe, f"{base_name}.jpg")
            cv2.imwrite(output_path, cll)
            count += 1

            percent = int((i + 1) / total_files * 100)
            self.progress_bar.setValue(percent)
            QApplication.processEvents()

        self.progress_bar.setValue(100)
        self.clear_aug_label.setText('CLAHE 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", f"총 {count}개의 CLAHE 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)

        self.clear_aug_label.setText('CLAHE 폴더가 준비됐습니다.')

############################# Horizontal Flip ###########################
    def imgHflip(self):
        pattern = re.compile(r'_(HF|VF|HS\d+|VS\d+|clahe|arr|expand|shrink|left|right|up|down|RT\d+)\.', re.IGNORECASE)
        original_img_list = [ f for f in self.original_img_list if not pattern.search(f) ]  # 원본 이미지 필터링
        # 05.07 추가(ProgressBar)
        total_files = len(original_img_list)       # original_img_list 변경 07.15
        self.progress_bar.setMaximum(100)       # 전체 개수 설정
        self.progress_bar.setValue(0)       # 초기화

        count = 0
        # 07.07
        output_dir = os.path.join(self.folder_path, self.folder_name_hf)
        os.makedirs(output_dir, exist_ok=True)

        # 증강된 파일 존재 여부 체크 _07.24
        already_augmented = False
        for file in original_img_list:
            base_name = os.path.basename(file).split('.')[0]
            output_image_path = os.path.join(output_dir, f"{base_name}_HF.jpg")
            if os.path.exists(output_image_path):
                already_augmented = True
                break
        # 이미 증강된 이미지가 존재하는 경우 _07.24
        if already_augmented:
            QMessageBox.information(self, "이미 증강됨", "이미 증강된 이미지가 존재합니다.\n증강을 다시 실행하지 않습니다.")
            self.progress_bar.setValue(0)
            return

        for idx, file in enumerate(self.original_img_list):   # original_img_list 변경 07.15
            image_path = os.path.join(self.folder_path, file)
            image = PILImage.open(image_path)
            base_name = os.path.basename(file).split('.')[0]
            flip_image = image.transpose(Transpose.FLIP_LEFT_RIGHT)

            # 07.07
            output_image_path = os.path.join(output_dir, f"{base_name}_HF.jpg")
            flip_image.save(output_image_path)
            shutil.copy(output_image_path, self.annos_dir)

            count += 1
            percent = int((idx + 1) / total_files * 100)
            self.progress_bar.setValue(percent)  # 진행률 업데이트
            QApplication.processEvents()  # UI 업데이트

        self.progress_bar.setValue(100)
        self.clear_aug_label.setText('수평 뒤집기(H Flip) 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", f"총 {count}개의 수평 뒤집기(H Flip) 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)       # 완료 후 0으로 리셋

    def HFlip_bbox_coordinates(self, image_width, xmin, xmax):
        return int(image_width - xmin), int(image_width - xmax)

    def xmlHflip(self):
        # 07.07
        output_dir = os.path.join(self.folder_path, self.folder_name_hf)
        os.makedirs(output_dir, exist_ok=True)

        # 증강된 파일 존재 여부 체크_07.24
        already_augmented = False
        for file in self.original_xml_list:
            base_name = os.path.basename(file).split('.')[0]
            output_image_path = os.path.join(output_dir, f"{base_name}_HF.xml")
            if os.path.exists(output_image_path):
                already_augmented = True
                break
        if already_augmented:
            self.progress_bar.setValue(0)
            return

        for idx, xml_file in enumerate(self.original_xml_list):     # original_xml_list 변경 07.15
            xml_path = os.path.join(self.folder_path, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            image_width = int(root.find('size/width').text)
            for fln in root.iter('filename'):
                original_filename = fln.text
                new_name = f"_HF.jpg"
                modified_filename = original_filename.replace('.jpg', new_name)
                fln.text = modified_filename
            for path in root.iter('path'):
                original_path = path.text
                new_name = f"_HF.jpg"
                modified_path = original_path.replace('.jpg', new_name)
                path.text = modified_path
            for obj in root.iter('object'):
                for bbox in obj.iter('bndbox'):
                    xmin = int(bbox.find('xmin').text)
                    xmax = int(bbox.find('xmax').text)
                    new_xmin, new_xmax = self.HFlip_bbox_coordinates(image_width, xmin, xmax)
                    bbox.find('xmin').text = str(new_xmax)
                    bbox.find('xmax').text = str(new_xmin)

            base_name, extension = os.path.splitext(os.path.basename(xml_file))
            new_xml_name = f"{base_name}_HF.xml"
            output_xml_path = os.path.join(self.folder_path, self.folder_name_hf, new_xml_name)
            tree.write(output_xml_path)

            # 07.07
            shutil.copy(output_xml_path, self.annos_dir)

        self.clear_aug_label.setText('수평 뒤집기(H Flip) 폴더가 준비됐습니다.')

############################# Vertical Flip ##########################
    def imgVflip(self):
        pattern = re.compile(r'_(HF|VF|HS\d+|VS\d+|clahe|arr|expand|shrink|left|right|up|down|RT\d+)\.', re.IGNORECASE)
        original_img_list = [ f for f in self.original_img_list if not pattern.search(f) ]  # 원본 이미지 필터링
        # 05.07 추가(ProgressBar)
        total_files = len(original_img_list)
        self.progress_bar.setMaximum(100)   
        self.progress_bar.setValue(0)

        count = 0
        # 07.07
        output_dir = os.path.join(self.folder_path, self.folder_name_vf)
        os.makedirs(output_dir, exist_ok=True)

        # 증강된 파일 존재 여부 체크
        already_augmented = False
        for file in original_img_list:
            base_name = os.path.basename(file).split('.')[0]
            output_image_path = os.path.join(output_dir, f"{base_name}_VF.jpg")
            if os.path.exists(output_image_path):
                already_augmented = True
                break

        if already_augmented:
            QMessageBox.information(self, "이미 증강됨", "이미 증강된 이미지가 존재합니다.\n증강을 다시 실행하지 않습니다.")
            self.progress_bar.setValue(0)
            return    

        for idx, file in enumerate(self.original_img_list):
            image_path = os.path.join(self.folder_path, file)
            image = PILImage.open(image_path)
            base_name = os.path.basename(file).split('.')[0]
            flip_image = image.transpose(Transpose.FLIP_TOP_BOTTOM)

            # 07.07
            output_image_path = os.path.join(output_dir, f"{base_name}_VF.jpg")
            flip_image.save(output_image_path)
            shutil.copy(output_image_path, self.annos_dir)

            count += 1
            percent = int((idx + 1) / total_files * 100)
            self.progress_bar.setValue(percent)  # 진행률 업데이트
            QApplication.processEvents()  # UI 업데이트

        self.progress_bar.setValue(100)
        self.clear_aug_label.setText('수직 뒤집기(V Flip) 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", f"총 {count}개의 수직 뒤집기(V Flip) 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)

    def VFlip_bbox_coordinates(self, image_height, ymin, ymax):
        return int(image_height - ymin), int(image_height - ymax)

    def xmlVflip(self):
        # 07.07
        output_dir = os.path.join(self.folder_path, self.folder_name_vf)
        os.makedirs(output_dir, exist_ok=True)

            # 증강된 파일 존재 여부 체크
        already_augmented = False
        for file in self.original_xml_list:
            base_name = os.path.basename(file).split('.')[0]
            output_image_path = os.path.join(output_dir, f"{base_name}_VF.xml")
            if os.path.exists(output_image_path):
                already_augmented = True
                break
        if already_augmented:
            self.progress_bar.setValue(0)
            return

        for idx, xml_file in enumerate(self.original_xml_list):
            xml_path = os.path.join(self.folder_path, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            image_height = int(root.find('size/height').text)
            for fln in root.iter('filename'):
                original_filename = fln.text
                new_name = "_VF.jpg"
                modified_filename = original_filename.replace('.jpg', new_name)
                fln.text = modified_filename
            for path in root.iter('path'):
                original_path = path.text
                new_name = "_VF.jpg"
                modified_path = original_path.replace('.jpg', new_name)
                path.text = modified_path
            for obj in root.iter('object'):
                for bbox in obj.iter('bndbox'):
                    ymin = int(bbox.find('ymin').text)
                    ymax = int(bbox.find('ymax').text)
                    new_ymin, new_ymax = self.VFlip_bbox_coordinates(image_height, ymin, ymax)
                    bbox.find('ymin').text = str(new_ymax)
                    bbox.find('ymax').text = str(new_ymin)

            base_name, extension = os.path.splitext(os.path.basename(xml_file))
            new_xml_name = f"{base_name}_VF.xml"
            output_xml_path = os.path.join(self.folder_path, self.folder_name_vf, new_xml_name)
            tree.write(output_xml_path)

            # 07.07
            shutil.copy(output_xml_path, self.annos_dir)
        self.clear_aug_label.setText('수직 뒤집기(V Flip) 폴더가 준비됐습니다.')

############################# Horizontal Shift ############################
    def imgHshift(self):
        pattern = re.compile(r'_(HF|VF|HS\d+|VS\d+|clahe|arr|expand|shrink|left|right|up|down|RT\d+)\.', re.IGNORECASE)
        original_img_list = [ f for f in self.original_img_list if not pattern.search(f) ]  # 원본 이미지 필터링
        # 05.07 추가
        total_files = len(original_img_list) * len(self.num_shift)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        
        count = 0
        current = 0

        output_dir = os.path.join(self.folder_path, self.folder_name_hs)
        os.makedirs(output_dir, exist_ok=True)

        # 증강된 파일 존재 여부 체크 (모든 shift에 대해)
        already_augmented = False
        for file in original_img_list:
            base_name = os.path.basename(file).split('.')[0]
            for i in self.num_shift:
                output_image_path = os.path.join(output_dir, f"{base_name}_HS{i}.jpg")
                if os.path.exists(output_image_path):
                    already_augmented = True
                    break
            if already_augmented:
                break
        if already_augmented:
            QMessageBox.information(self, "이미 증강됨", "이미 증강된 이미지가 존재합니다.\n증강을 다시 실행하지 않습니다.")
            self.progress_bar.setValue(0)
            return

        for i in self.num_shift:
            for j in range(len(self.original_img_list)):
                img = self.original_img_list[j]
                image_path = os.path.join(self.folder_path, img)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                width, height, _ = image.shape
                base_name = os.path.basename(img).split('.')[0]
                M = np.float32([[1, 0, i], [0, 1, 0]])
                shifted = cv2.warpAffine(image, M, (height, width))
                
                # 07.07
                output_image_path = os.path.join(output_dir, f"{base_name}_HS{i}.jpg")
                cv2.imwrite(output_image_path, cv2.cvtColor(shifted, cv2.COLOR_RGB2BGR))
                shutil.copy(output_image_path, self.annos_dir)

                count += 1

                current += 1
                percent = int(current / total_files * 100)

            # 05.07 추가
            self.progress_bar.setValue(percent)
            QApplication.processEvents()

        self.progress_bar.setValue(100)
        self.clear_aug_label.setText('수평 이동(H Shift) 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", f"총 {count}개의 수평 이동(H Shift) 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)

    def Hshift_bbox_coordinates(self, num_shift, xmin, xmax):
        return int(xmin + num_shift), int(xmax + num_shift)
    def xmlHshift(self, xml_file, i):
        xml_path = os.path.join(self.folder_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for fln in root.iter('filename'):
            original_filename = fln.text
            new_name = f"_HS{i}.jpg"
            modified_filename = original_filename.replace('.jpg', new_name)
            fln.text = modified_filename
        for path in root.iter('path'):
            original_path = path.text
            new_name = f"_HS{i}.jpg"
            modified_path = original_path.replace('.jpg', new_name)
            path.text = modified_path
        for obj in root.iter('object'):
            for bbox in obj.iter('bndbox'):
                xmin = int(bbox.find('xmin').text)
                xmax = int(bbox.find('xmax').text)
                new_xmin, new_xmax = self.Hshift_bbox_coordinates(i, xmin, xmax)
                bbox.find('xmin').text = str(new_xmin)
                bbox.find('xmax').text = str(new_xmax)

        base_name, extension = os.path.splitext(os.path.basename(xml_file))
        new_xml_name = f"{base_name}_HS{i}.xml"
        output_xml_path = os.path.join(self.folder_path, self.folder_name_hs, new_xml_name)
        tree.write(output_xml_path)

        # 07.07
        shutil.copy(output_xml_path, self.annos_dir)
        self.clear_aug_label.setText('수평 이동(H Shift) 폴더가 준비됐습니다.')

    def xmlHshift_all(self):
        output_dir = os.path.join(self.folder_path, self.folder_name_hs)
        os.makedirs(output_dir, exist_ok=True)

        # 증강된 파일 존재 여부 체크 (모든 shift에 대해)
        already_augmented = False
        for xml_file in self.original_xml_list:
            base_name = os.path.basename(xml_file).split('.')[0]
            for i in self.num_shift:
                output_xml_path = os.path.join(output_dir, f"{base_name}_HS{i}.xml")
                if os.path.exists(output_xml_path):
                    already_augmented = True
                    break
            if already_augmented:
                break
        if already_augmented:
            self.progress_bar.setValue(0)
            return
        # 실제 증강 실행
        for xml_file in self.original_xml_list:
            for i in self.num_shift:
                self.xmlHshift(xml_file, i)

############################# Vertical Shift ############################
    def imgVshift(self):
        pattern = re.compile(r'_(HF|VF|HS\d+|VS\d+|clahe|arr|expand|shrink|left|right|up|down|RT\d+)\.', re.IGNORECASE)
        original_img_list = [ f for f in self.original_img_list if not pattern.search(f) ]  # 원본 이미지 필터링        
        # 05.07 추가
        total_files = len(original_img_list) * len(self.num_shift)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        count = 0
        current = 0

        # 07.07
        output_dir = os.path.join(self.folder_path, self.folder_name_vs)
        os.makedirs(output_dir, exist_ok=True)

        already_augmented = False
        for file in original_img_list:
            base_name = os.path.basename(file).split('.')[0]
            for i in self.num_shift:
                output_image_path = os.path.join(output_dir, f"{base_name}_VS{i}.jpg")
                if os.path.exists(output_image_path):
                    already_augmented = True
                    break
            if already_augmented:
                break
        if already_augmented:
            QMessageBox.information(self, "이미 증강됨", "이미 증강된 이미지가 존재합니다.\n증강을 다시 실행하지 않습니다.")
            self.progress_bar.setValue(0)
            return

        for i in self.num_shift:
            for j in range(len(self.original_img_list)):
                img = self.original_img_list[j]
                image_path = os.path.join(self.folder_path, img)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                width, height, _ = image.shape
                base_name = os.path.basename(img).split('.')[0]
                M = np.float32([[1, 0, 0], [0, 1, i]])
                shifted = cv2.warpAffine(image, M, (height, width))
                        
                # 07.07
                output_image_path = os.path.join(output_dir, f"{base_name}_VS{i}.jpg")
                cv2.imwrite(output_image_path, cv2.cvtColor(shifted, cv2.COLOR_RGB2BGR))
                shutil.copy(output_image_path, self.annos_dir)

                count += 1

                current += 1
                percent = int(current / total_files * 100)
            # 05.07 추가
            self.progress_bar.setValue(percent)
            QApplication.processEvents()

        self.progress_bar.setValue(100)
        self.clear_aug_label.setText('수직 이동(V Shift) 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", f"총 {count}개의 수직 이동(V Shift) 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)

    def Vshift_bbox_coordinates(self, num_shift, ymin, ymax):
        return int(ymin + num_shift), int(ymax + num_shift)

    def xmlVshift(self, xml_file, i):
        xml_path = os.path.join(self.folder_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for fln in root.iter('filename'):
            original_filename = fln.text
            new_name = f"_VS{i}.jpg"
            modified_filename = original_filename.replace('.jpg', new_name)
            fln.text = modified_filename
        for path in root.iter('path'):
            original_path = path.text
            new_name = f"_VS{i}.jpg"
            modified_path = original_path.replace('.jpg', new_name)
            path.text = modified_path
        for obj in root.iter('object'):
            for bbox in obj.iter('bndbox'):
                ymin = int(bbox.find('ymin').text)
                ymax = int(bbox.find('ymax').text)
                new_ymin, new_ymax = self.Vshift_bbox_coordinates(i, ymin, ymax)
                bbox.find('ymin').text = str(new_ymin)
                bbox.find('ymax').text = str(new_ymax)

        base_name, extension = os.path.splitext(os.path.basename(xml_file))
        new_xml_name = f"{base_name}_VS{i}.xml"
        output_xml_path = os.path.join(self.folder_path, self.folder_name_vs, new_xml_name)
        tree.write(output_xml_path)

        # 07.07
        shutil.copy(output_xml_path, self.annos_dir)
        self.clear_aug_label.setText('수직 이동(V Shift) 폴더가 준비됐습니다.')

    def xmlVshift_all(self):
        output_dir = os.path.join(self.folder_path, self.folder_name_vs)
        os.makedirs(output_dir, exist_ok=True)
        already_augmented = False
        for xml_file in self.original_xml_list:
            base_name = os.path.basename(xml_file).split('.')[0]
            for i in self.num_shift:
                output_xml_path = os.path.join(output_dir, f"{base_name}_HS{i}.xml")
                if os.path.exists(output_xml_path):
                    already_augmented = True
                    break
            if already_augmented:
                break
        if already_augmented:
            QMessageBox.information(self, "이미 증강됨", "이미 증강된 XML이 존재합니다.\n증강을 다시 실행하지 않습니다.")
            self.progress_bar.setValue(0)
            return

        for xml_file in self.original_xml_list:
            for i in self.num_shift:
                self.xmlVshift(xml_file, i)

############################# Row by Column ############################
    def imgRowColumn(self, img):
        pattern = re.compile(r'_(HF|VF|HS\d+|VS\d+|clahe|arr|expand|shrink|left|right|up|down|RT\d+)\.', re.IGNORECASE)
        original_img_list = [ f for f in self.original_img_list if not pattern.search(f) ]  # 원본 이미지 필터링        
        self.rows = int(self.get_row_number())
        self.cols = int(self.get_col_number())
        image_path = os.path.join(self.folder_path, img)
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        sub_img_width = width // self.cols
        sub_img_height = height // self.rows
        cropped_images = []
        all = self.rows * self.cols
        for row in range(self.rows):
            for col in range(self.cols):
                x1 = col * sub_img_width
                y1 = row * sub_img_height
                x2 = (col + 1) * sub_img_width
                y2 = (row + 1) * sub_img_height
                sub_img = image[y1:y2, x1:x2]
                cropped_images.append(sub_img)
        rearranged_images = [cropped_images[i % len(cropped_images)] for i in range(1, all + 1, 1)]
        new_img = np.zeros_like(image)
        for row in range(self.rows):
            for col in range(self.cols):
                x1 = col * sub_img_width
                y1 = row * sub_img_height
                x2 = (col + 1) * sub_img_width
                y2 = (row + 1) * sub_img_height
                new_img[y1:y2, x1:x2] = rearranged_images.pop(0)
        base_name = os.path.basename(img).split('.')[0]
        os.makedirs(os.path.join(self.folder_path, self.folder_rbc), exist_ok=True)
        cv2.imwrite(os.path.join(self.folder_path, self.folder_rbc, f"{base_name}_arr.jpg"), new_img)

    def xmlRowColumn(self, xml_file):
        xml_path = os.path.join(self.folder_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_height = int(root.find('size/height').text)
        image_width = int(root.find('size/width').text)
        for fln in root.iter('filename'):
            original_filename = fln.text
            new_name = f"_arr.jpg"
            modified_filename = original_filename.replace('.jpg', new_name)
            fln.text = modified_filename
        for path in root.iter('path'):
            original_path = path.text
            new_name = f"_arr.jpg"
            modified_path = original_path.replace('.jpg', new_name)
            path.text = modified_path
        for obj in root.iter('object'):
            for bbox in obj.iter('bndbox'):
                ymin = int(bbox.find('ymin').text)
                ymax = int(bbox.find('ymax').text)
                xmin = int(bbox.find('xmin').text)
                xmax = int(bbox.find('xmax').text)
                new_xmin, new_xmax, new_ymin, new_ymax = self.rowNcolumn_coordinates(image_width, image_height, xmin, xmax,
                                                                                ymin, ymax)
                bbox.find('ymin').text = str(new_ymin)
                bbox.find('ymax').text = str(new_ymax)
                bbox.find('xmin').text = str(new_xmin)
                bbox.find('xmax').text = str(new_xmax)
        self.clear_aug_label.setText('행렬 변환(Row by Column) 폴더가 준비됐습니다.')

        base_name, extension = os.path.splitext(os.path.basename(xml_file))
        new_xml_name = f"{base_name}_arr.xml"
        output_xml_path = os.path.join(self.folder_path, self.folder_rbc, new_xml_name)
        tree.write(output_xml_path)
        self.clear_aug_label.setText('회전(Rotation) 폴더가 준비됐습니다.')

    def rowNcolumn_coordinates(self, image_width, image_height, xmin, xmax, ymin, ymax):
        if xmin > int(image_width / self.cols):
            new_xmin = xmin - int(image_width / self.cols)
            new_xmax = xmax - int(image_width / self.cols)
            new_ymin = ymin
            new_ymax = ymax
            return int(new_xmin), int(new_xmax), int(new_ymin), int(new_ymax)
        elif ymin > int(image_height / self.rows) and xmin < int(image_width / self.cols):
            new_xmin = xmin + int(image_width / self.cols)
            new_xmax = xmax + int(image_width / self.cols)
            new_ymin = ymin - int(image_height / self.rows)
            new_ymax = ymax - int(image_height / self.rows)
            return int(new_xmin), int(new_xmax), int(new_ymin), int(new_ymax)
        elif ymin < int(image_height / self.rows) and xmin < int(image_width / self.cols):
            new_xmin = xmin + int((image_width / self.cols)*(self.cols-1))
            new_xmax = xmax + int((image_width / self.cols)*(self.cols-1))
            new_ymin = ymin + int((self.rows-1) * (image_height / self.rows))
            new_ymax = ymax + int((self.rows-1) * (image_height / self.rows))
            return int(new_xmin), int(new_xmax), int(new_ymin), int(new_ymax)

    def RowColumn_all(self):
        row_text = self.row_input.text()
        col_text = self.column_input.text()
        if not row_text or not col_text:
            QMessageBox.warning(self, "입력 오류", "행과 열의 숫자를 입력해주세요.")
            return
            
        output_dir = os.path.join(self.folder_path, self.folder_rbc)
        os.makedirs(output_dir, exist_ok=True)

        already_augmented = False
        for img in self.original_img_list:
            base_name = os.path.basename(img).split('.')[0]
            output_image_path = os.path.join(output_dir, f"{base_name}_arr.jpg")
            if os.path.exists(output_image_path):
                already_augmented = True
                break
        if not already_augmented:
            for xml_file in self.original_xml_list:
                base_name = os.path.basename(xml_file).split('.')[0]
                output_xml_path = os.path.join(output_dir, f"{base_name}_arr.xml")
                if os.path.exists(output_xml_path):
                    already_augmented = True
                    break
        if already_augmented:
            QMessageBox.information(self, "이미 증강됨", "이미 증강된 Row by Column 이미지 또는 XML이 존재합니다.\n증강을 다시 실행하지 않습니다.")
            self.progress_bar.setValue(0)
            return

        total = len(self.original_img_list) + len(self.original_xml_list)
        current = 0
        for img in self.original_img_list:
            self.imgRowColumn(img)
            current += 1
            progress = int(current / total * 100)

            self.progress_bar.setValue(progress)
            QApplication.processEvents()

        for xml_file in self.original_xml_list:
            self.xmlRowColumn(xml_file)
            current += 1
            progress = int(current / total * 100)

            self.progress_bar.setValue(progress)
            QApplication.processEvents()

        # 06.18 추가
        self.progress_bar.setValue(100)     
        self.clear_aug_label.setText('행렬 변환(Row by Column) 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", "행렬 변환(Row by Column) 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)

############################# Rotation ############################
    def imgRotation(self):
        pattern = re.compile(r'_(HF|VF|HS\d+|VS\d+|clahe|arr|expand|shrink|left|right|up|down|RT\d+)\.', re.IGNORECASE)
        original_img_list = [ f for f in self.original_img_list if not pattern.search(f) ]  # 원본 이미지 필터링          
        # 05.07 추가
        total_files = len(original_img_list) * len(self.num_rot)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        count = 0
        current = 0

        # 07.07
        output_dir = os.path.join(self.folder_path, self.folder_name_rt)
        os.makedirs(output_dir, exist_ok=True)
        
        already_augmented = False
        for file in original_img_list:
            base_name = os.path.basename(file).split('.')[0]
            for angle in self.num_rot:
                output_image_path = os.path.join(output_dir, f"{base_name}_RT{angle}.jpg")
                if os.path.exists(output_image_path):
                    already_augmented = True
                    break
            if already_augmented:
                break
        if already_augmented:
            QMessageBox.information(self, "이미 증강됨", "이미 증강된 회전 이미지가 존재합니다.\n증강을 다시 실행하지 않습니다.")
            self.progress_bar.setValue(0)
            return


        for i in self.num_rot:
            for j in range(len(self.original_img_list)):
                replace_path = self.original_img_list[j]
                image_path = os.path.join(self.folder_path, replace_path)
                image = cv2.imread(image_path)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        # 이거 하면 R G B 중에서 R 이랑 B 랑 뒤집혀서 나옴(컬러일때)
                height, width = image.shape[:2]
                center_img = (width // 2, height // 2)
                base_name = os.path.basename(replace_path).split('.')[0]

                rotate_matrix = cv2.getRotationMatrix2D(center=center_img, angle=i, scale=1)
                rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
               
                # 07.07
                output_image_path = os.path.join(output_dir, f"{base_name}_RT{i}.jpg")
                cv2.imwrite(output_image_path, rotated_image)
                shutil.copy(output_image_path, self.annos_dir)

                count += 1

                current += 1
                percent = int(current / total_files * 100)

            # 05.07 추가
            self.progress_bar.setValue(percent)
            QApplication.processEvents()

        self.progress_bar.setValue(100)
        self.clear_aug_label.setText('회전(Rotation) 작업이 완료됐습니다.')
        QMessageBox.information(self, "처리 완료", f"{count}개의 회전(Rotation) 작업이 완료됐습니다.")

        self.progress_bar.setValue(0)

    def xmlRotation_all(self):
        for xml_file in self.original_xml_list:
            for i in self.num_rot:
                self.xmlRotation(xml_file, i)

    def rotate_vertices(self, vertices, angle, bbox_center, image_height, image_width):
        angle_rad = np.radians(- angle)
        x, y = vertices
        cx, cy = bbox_center
        new_x = (((x - cx) * np.cos(angle_rad)) - ((y - cy) * np.sin(angle_rad))) + cx
        new_y = (((x - cx) * np.sin(angle_rad)) + ((y - cy) * np.cos(angle_rad))) + cy
        if new_y < 0:
            new_y = 0
        elif new_y > image_height:
            new_y = image_height
        elif new_x > image_width:
            new_x = image_width
        elif new_x < 0:
            new_x = 0
        return new_x, new_y

    def maxNminDistance(self, rotated_vertices, bbox_center):
        max_x = max(rotated_vertices, key=lambda point: point[0])[0]
        max_y = max(rotated_vertices, key=lambda point: point[1])[1]
        dis_x = max_x - bbox_center[0]
        dis_y = max_y - bbox_center[1]
        return dis_x, dis_y

    def rotate_central_point(self, bbox_center, angle, image_center):
        x, y = bbox_center
        angle_rad = np.radians(360 - angle)
        translated_point = np.array([x - image_center[0], y - image_center[1]])
        rotated_x = (translated_point[0] * np.cos(angle_rad)) - (translated_point[1] * np.sin(angle_rad))
        rotated_y = (translated_point[0] * np.sin(angle_rad)) + (translated_point[1] * np.cos(angle_rad))
        new_x = rotated_x + image_center[0]
        new_y = rotated_y + image_center[1]
        return new_x, new_y

    def final_vertices(self, new_bbox_center, distance_from_center):
        new_xmin = new_bbox_center[0] - distance_from_center[0]
        new_ymin = new_bbox_center[1] - distance_from_center[1]
        new_xmax = new_bbox_center[0] + distance_from_center[0]
        new_ymax = new_bbox_center[1] + distance_from_center[1]
        return (int(new_xmin), int(new_ymin), int(new_xmax), int(new_ymax))

    def move_and_rotate_box(self, box_vertices, i, image_center, bbox_center, image_height, image_width):
        rotated_vertices = [self.rotate_vertices(vertex, i, bbox_center, image_height, image_width) for vertex in
                            box_vertices]
        distance_from_center = self.maxNminDistance(rotated_vertices, bbox_center)
        new_bbox_center = self.rotate_central_point(bbox_center, i, image_center)
        final_vertices_new_bbox = self.final_vertices(new_bbox_center, distance_from_center)
        return final_vertices_new_bbox

    def xmlRotation(self, xml_file, i):
        xml_path = os.path.join(self.folder_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_height = int(root.find('size/height').text)
        image_width = int(root.find('size/width').text)
        image_center = np.array([image_width / 2, image_height / 2])
        for fln in root.iter('filename'):
            original_filename = fln.text
            new_name = f"_RT{i}.jpg"
            modified_filename = original_filename.replace('.jpg', new_name)
            fln.text = modified_filename
        for path in root.iter('path'):
            original_path = path.text
            new_name = f"_RT{i}.jpg"
            modified_path = original_path.replace('.jpg', new_name)
            path.text = modified_path
        for obj in root.iter('object'):
            for bbox in obj.iter('bndbox'):
                xmin = bbox.find('xmin').text
                xmin = eval(xmin)
                ymin = bbox.find('ymin').text
                ymin = eval(ymin)
                xmax = bbox.find('xmax').text
                xmax = eval(xmax)
                ymax = bbox.find('ymax').text
                ymax = eval(ymax)
                box_vertices = np.array([[xmin, ymin], [xmax, ymin],
                                         [xmax, ymax], [xmin, ymax]
                                         ])
                bbox_center = np.mean(box_vertices, axis=0)
                result_vertices = self.move_and_rotate_box(box_vertices, i, image_center, bbox_center, image_height,
                                                      image_width)
                min_x = result_vertices[0]
                min_y = result_vertices[1]
                max_x = result_vertices[2]
                max_y = result_vertices[3]
                bbox.find('xmin').text = str(min_x)
                bbox.find('ymin').text = str(min_y)
                bbox.find('xmax').text = str(max_x)
                bbox.find('ymax').text = str(max_y)
        base_name, extension = os.path.splitext(os.path.basename(xml_file))
        new_xml_name = f"{base_name}_RT{i}.xml"
        output_xml_path = os.path.join(self.folder_path, self.folder_name_rt, new_xml_name)
        tree.write(output_xml_path)
        # 07.07
        shutil.copy(output_xml_path, self.annos_dir)
        # 07.07
    def xmlRotation_all(self):
        output_dir = os.path.join(self.folder_path, self.folder_name_rt)
        os.makedirs(output_dir, exist_ok=True)

        # 증강된 파일 존재 여부 체크
        already_augmented = False
        for xml_file in self.original_xml_list:
            base_name = os.path.basename(xml_file).split('.')[0]
            for angle in self.num_rot:
                output_xml_path = os.path.join(output_dir, f"{base_name}_RT{angle}.xml")
                if os.path.exists(output_xml_path):
                    already_augmented = True
                    break
            if already_augmented:
                break
        if already_augmented:
            self.progress_bar.setValue(0)
            return
        
        for xml_file in self.original_xml_list:
            for angle in self.num_rot:
                self.xmlRotation(xml_file, angle)
        self.clear_aug_label.setText('회전 (Rotation) 폴더가 준비됐습니다.')
############################## Color to Gray ###############################################
    def ColorToGray(self):
        os.makedirs(os.path.join(self.folder_path, self.folder_gray), exist_ok=True)
        os.makedirs(os.path.join(self.folder_path, self.folder_color), exist_ok=True)
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(self.folder_path, filename)
                img_color = cv2.imread(image_path)
                if img_color is not None:
                    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
                    output_path_gray = os.path.join(self.folder_path, self.folder_gray, filename)
                    output_path_color = os.path.join(self.folder_path, self.folder_color, filename)
                    cv2.imwrite(output_path_gray, img_gray)
                    cv2.imwrite(output_path_color, img_color)
        self.clear_aug_label.setText('흑백 파일이 생성됩니다.')

############################## BGR to RGB ###############################################
    def BGRtoRGB(self):
        os.makedirs(os.path.join(self.folder_path, self.output_folder_rgb), exist_ok=True)
        os.makedirs(os.path.join(self.folder_path, self.output_folder_bgr), exist_ok=True)
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(self.folder_path, filename)
                img_bgr = cv2.imread(image_path)
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    output_path_rgb = os.path.join(self.folder_path, self.output_folder_rgb, filename)
                    output_path_bgr = os.path.join(self.folder_path, self.output_folder_bgr, filename)
                    cv2.imwrite(output_path_rgb, img_rgb)
                    cv2.imwrite(output_path_bgr, img_bgr)
        self.clear_aug_label.setText('RGB 파일이 생성됩니다.')

############################## RGB to BGR ###############################################
    def RGBtoBGR(self):
        os.makedirs(os.path.join(self.folder_path, self.output_folder_rgb), exist_ok=True)
        os.makedirs(os.path.join(self.folder_path, self.output_folder_bgr), exist_ok=True)
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(self.folder_path, filename)
                img_rgb = cv2.imread(image_path)
                if img_rgb is not None:
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    output_path_rgb = os.path.join(self.folder_path, self.output_folder_rgb, filename)
                    output_path_bgr = os.path.join(self.folder_path, self.output_folder_bgr, filename)
                    cv2.imwrite(output_path_rgb, img_rgb)
                    cv2.imwrite(output_path_bgr, img_bgr)
        self.clear_aug_label.setText('BGR 파일이 생성됩니다.')

# ######################## XML 변경 (기존 XML 파일에 object 추가) _PJS #######################
    def transform_xml_boxes(self):
        folder_path = r"D:\AI_SVT_Training_mk\annotations\annos"
        mm_per_pixel = 0.03
        offset_mm = 0.15
        offset_px = int(offset_mm / mm_per_pixel)

        # 좌표 수정 함수 
        def modify_expand(bndbox, px):          # 확장
            bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) - px)
            bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) + px)
            bndbox.find('ymin').text = str(int(bndbox.find('ymin').text) - px)
            bndbox.find('ymax').text = str(int(bndbox.find('ymax').text) + px)

        def modify_shrink(bndbox, px):          # 축소
            bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) + px)
            bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) - px)
            bndbox.find('ymin').text = str(int(bndbox.find('ymin').text) + px)
            bndbox.find('ymax').text = str(int(bndbox.find('ymax').text) - px)

        def modify_left(bndbox, px):            # 좌로 이동
            bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) - px)
            bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) - px)

        def modify_right(bndbox, px):           # 우로 이동
            bndbox.find('xmin').text = str(int(bndbox.find('xmin').text) + px)
            bndbox.find('xmax').text = str(int(bndbox.find('xmax').text) + px)

        def modify_up(bndbox, px):           # 위로 이동_04.30추가
            bndbox.find('ymin').text = str(int(bndbox.find('ymin').text) - px)
            bndbox.find('ymax').text = str(int(bndbox.find('ymax').text) - px)

        def modify_down(bndbox, px):           # 아래로 이동_04.30추가
            bndbox.find('ymin').text = str(int(bndbox.find('ymin').text) + px)
            bndbox.find('ymax').text = str(int(bndbox.find('ymax').text) + px)

        # 함수 리스트
        transformations = [
            modify_expand,
            modify_left,
            modify_right,
            modify_shrink,
            modify_up,
            modify_down
        ]

         # ProgressBar
        xml_files = [file for file in os.listdir(folder_path) if file.endswith(".xml")]
        total_files = len(xml_files)
        self.progress_bar.setMaximum(total_files)   # 전체 개수 설정
        self.progress_bar.setValue(0)   # 초기화

        # 이미 증강된 XML 체크 (object 50개 이상이면 중복으로 간주)
        already_augmented = False
        for file in xml_files:
            xml_path = os.path.join(folder_path, file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                if len(root.findall('object')) >= 100:
                    already_augmented = True
                    break
            except Exception:
                continue
        if already_augmented:
            QMessageBox.information(self, "이미 증강됨", "이미 object가 추가된 XML이 존재합니다.")
            self.progress_bar.setValue(0)
            return

        # 전체 XML 처리
        count = 0
        for idx, file in enumerate(xml_files):
            xml_path = os.path.join(folder_path, file)

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                # 필터링 추가
                image_width = int(root.find("size/width").text)
                image_height = int(root.find("size/height").text)
                
                original_objects = root.findall('object')
                if not original_objects:
                    continue

                for obj in original_objects:
                    for transform_func in transformations:
                        new_obj = copy.deepcopy(obj)
                        bndbox = new_obj.find('bndbox')
                        transform_func(bndbox, offset_px)

                        # 유효성 검사 추가
                        xmin = int(bndbox.find('xmin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymin = int(bndbox.find('ymin').text)
                        ymax = int(bndbox.find('ymax').text)

                        if (xmin < 0 or xmax > image_width or ymin < 0 or ymax > image_height or
                            xmin >= xmax or ymin >= ymax):
                            continue

                        root.append(new_obj)

                tree.write(xml_path, encoding='utf-8')
                count += 1
            except Exception as e:
                print(f"오류 발생: {file} - {e}", flush=True)

            # ProgressBar 갱신
            self.progress_bar.setValue(idx + 1)
            QApplication.processEvents()    # 이벤트 루프 업뎃 (프리징 방지)

        QMessageBox.information(self, "처리 완료", f"{count}개의 XML파일이 수정되었습니다.")

        self.progress_bar.setValue(0)   # 완료 후 0으로 리셋

##################### Auto Label 조정(JPG, XML 따로따로 생성) _PJS ######################
    def adjust_bbox_xml(self, input_xml, output_xml, shift_x=0, shift_y=0, expand_x=0, expand_y=0):
        import shutil
        
        tree = ET.parse(input_xml)
        root = tree.getroot()

        new_filename = os.path.basename(output_xml).replace(".xml", ".jpg")

        # xml path ↠ annos 로 고정
        fixed_path = os.path.join("D:\\AI_SVT_Training_mk\\annotations\\annos", new_filename)

        # <filename> 태그 수정
        for fln in root.iter("filename"):
            fln.text = new_filename
        for path_tag in root.iter("path"):
            path_tag.text = fixed_path

        # <path> 태그 수정
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)    # 상, 하 추가
            ymax = int(bndbox.find('ymax').text)

            # 크기 확장/축소
            xmin -= expand_x
            xmax += expand_x
            ymin -= expand_y
            ymax += expand_y

            # 위치 이동
            xmin += shift_x
            xmax += shift_x
            ymin += shift_y
            ymax += shift_y

            xmin = max(0, xmin)
            xmax = max(xmin + 1, xmax)
            ymin = max(0, ymin)
            ymax = max(ymin + 1, ymax)

            bndbox.find('xmin').text = str(xmin)
            bndbox.find('xmax').text = str(xmax)
            bndbox.find('ymin').text = str(ymin)
            bndbox.find('ymax').text = str(ymax)

        # XML 저장
        tree.write(output_xml, encoding='utf-8')
        # 07.07
        shutil.copy(output_xml, self.annos_dir)

        # 이미지 복사
        base_name = os.path.basename(input_xml).replace(".xml", "")
        original_img_path = os.path.join(self.folder_path, f"{base_name}.jpg")

        if os.path.exists(original_img_path):
            new_img_path = output_xml.replace(".xml", ".jpg")
            shutil.copyfile(original_img_path, new_img_path)
            # 07.07
            shutil.copy(new_img_path, self.annos_dir)

    def process_xml_variants(self):
        input_dir = self.folder_path
        output_base = os.path.join(input_dir, "augmented_mm")
        os.makedirs(output_base, exist_ok=True)     # 07.07

        # 증강 파일명 패턴
        pattern = re.compile(r'_(HF|VF|HS\d+|VS\d+|clahe|arr|expand|shrink|left|right|up|down|RT\d+)\.', re.IGNORECASE)

        # 원본 이미지만 필터링
        original_img_list = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg') and not pattern.search(f)]

        # 조정 설정: 폴더 이름 ↠ (이동값, 크기값)
        subfolders = {
            "expand":  (0, 0, 3, 3),     # shift_x, shift_y, expand_x, expand_y
            "shrink":  (0, 0, -3, -3),
            "left":    (-3, 0, 0, 0),
            "right":   (3, 0, 0, 0),
            "up":      (0, -3, 0, 0),
            "down":    (0, 3, 0, 0)
        }
        # 07.07
        for sub in subfolders:
            os.makedirs(os.path.join(output_base, sub), exist_ok=True)

        # ProgressBar
        total_files = len(original_img_list) * len(subfolders)      # * len(subfolders) 07.07

        self.progress_bar.setMaximum(total_files)   # 전체 개수 설정
        self.progress_bar.setValue(0)   # 초기화

        # 각 형태별 폴더 생성
        for sub in subfolders:
            os.makedirs(os.path.join(output_base, sub), exist_ok=True)

        already_augmented = False
        for image_file in original_img_list:
            base_name = os.path.splitext(image_file)[0]
            for sub in subfolders:
                output_img_path = os.path.join(output_base, sub, f"{base_name}_{sub}.jpg")
                output_xml_path = os.path.join(output_base, sub, f"{base_name}_{sub}.xml")
                if os.path.exists(output_img_path) or os.path.exists(output_xml_path):
                    already_augmented = True
                    break
            if already_augmented:
                break
        if already_augmented:
            QMessageBox.information(self, "이미 증강됨", "이미 증강된 Auto Label 이미지 또는 XML이 존재합니다.\n증강을 다시 실행하지 않습니다.")
            self.progress_bar.setValue(0)
            return

        count = 0
        for idx, image_file in enumerate(original_img_list):
            # if image_file.lower().endswith(".jpg"):
            base_name = os.path.splitext(image_file)[0]
            xml_name = f"{base_name}.xml"
            xml_path = os.path.join(input_dir, xml_name)

            if not os.path.exists(xml_path):
                continue    # XML 없으면 스킵

            for sub, (sx, sy, ex, ey) in subfolders.items():
                output_xml_name = f"{base_name}_{sub}.xml"
                output_path = os.path.join(output_base, sub, output_xml_name)
                self.adjust_bbox_xml(xml_path, output_path,shift_x=sx, shift_y=sy, expand_x=ex, expand_y=ey)
                count += 1
        
                    # ProgressBar 갱신
            count += 1
            self.progress_bar.setValue(count)
            QApplication.processEvents()    # 이벤트 루프 업뎃 (프리징 방지)

        #07.08
        self.progress_bar.setMaximum(total_files)
        self.clear_aug_label.setText(f"총 {count}개의 파일이 저장 완료되었습니다")
        QMessageBox.information(self, "처리 완료", f"{count}개의 파일이 수정되었습니다.")
        self.progress_bar.setValue(0)   # 완료 후 0으로 리셋
        
    # 06.25 _annos 변경 건 자동갱신
    def on_annos_changed(self, path):
        # (1) 최신 이미지 개수 재계산
        try:
            num_images = len([
                f for f in os.listdir(self.annos_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
            ])
        except Exception:
            num_images = 0
        self.num_images = num_images

        # (2) EpochNo.txt 업데이트
        self.update_epochno_file()

        # (3) UI 갱신
        if hasattr(self, 'info_label'):
            mul = self.bg_mul.id(self.bg_mul.checkedButton())

    # 06.25 _EpochNo.txt 생성/덮어쓰기 로직 분리
    def update_epochno_file(self):
        """
        self.num_images 와 현재 선택된 mul 값을 이용해
        annotations/EpochNo.txt 를 쓰거나 덮어씁니다.
        """
        # 현재 mul 값 읽기 (10,20,30)
        if hasattr(self, 'bg_mul') and self.bg_mul.checkedButton():
            mul = self.bg_mul.id(self.bg_mul.checkedButton())
        else:
            mul = 20  # 기본값

        # 쓰기
        annotation_dir = r"D:\AI_SVT_Training_mk\annotations"
        os.makedirs(annotation_dir, exist_ok=True)
        epoch_file = os.path.join(annotation_dir, "EpochNo.txt")
        with open(epoch_file, "w", encoding="utf-8") as f:
            f.write(str(self.num_images * mul))


def main():
    safe_qt_init()   # Qt 안전 초기화 먼저

    app = QApplication(sys.argv)

    splash = SplashScreen()
    viewer = ClassAugChanger()

    def open_main_window():
        viewer.show()

    splash.start_animation(open_main_window)

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
