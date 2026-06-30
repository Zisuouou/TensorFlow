# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Creates and runs TF2 object detection models.

For local training/evaluation run:
PIPELINE_CONFIG_PATH=path/to/pipeline.config
MODEL_DIR=/tmp/model_outputs
NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python model_main_tf2.py -- \
  --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr
"""
from absl import flags
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append("C:/path/to/official/package")
# 06.23 
sys.path.append(r"D:\AI_SVT_Training_mk\object_detection")
sys.path.append(r"D:\AI_SVT_Training_mk\slim")

# 26.01.20
import time

import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2

FLAGS = flags.FLAGS

# # 06.23
# annos_dir = r"D:\AI_SVT_Training_mk\annotations\annos"
# try:
#     num_images = len([f for f in os.listdir(annos_dir)
#                       if os.path.isfile(os.path.join(annos_dir, f))
#                       and f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
# except:
#     num_images = 0
# iMaxTrainStep = min(max(num_images * 20, 10000), 100000)
# FLAGS.num_train_steps = iMaxTrainStep
# print(f"[OVERRIDE_NUM_TRAIN_STEPS] {FLAGS.num_train_steps}", flush=True)

flags.DEFINE_string('pipeline_config_path', 'configs/faster_rcnn_resnet50_v1_800x1333_batch1.config', 'Path to pipeline config '
                    'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_bool('eval_on_train_data', False, 'Enable evaluating on train '
                  'data (only supported in distributed training).')
flags.DEFINE_integer('sample_1_of_n_eval_examples', None, 'Will sample one of '
                     'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                     'one of every n train input examples for evaluation, '
                     'where n is provided. This is only used if '
                     '`eval_training_data` is True.')
flags.DEFINE_string(
    'model_dir', 'train_result', 'Path to output model directory '
                       'where event and checkpoint files will be written.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
    '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
    'writing resulting metrics to `model_dir`.')

flags.DEFINE_integer('eval_timeout', 3600, 'Number of seconds to wait for an'
                     'evaluation checkpoint before exiting.')

flags.DEFINE_bool('use_tpu', False, 'Whether the job is executing on a TPU.')
flags.DEFINE_string(
    'tpu_name',
    default=None,
    help='Name of the Cloud TPU for Cluster Resolvers.')
flags.DEFINE_integer(
    'num_workers', 1, 'When num_workers > 1, training uses '
    'MultiWorkerMirroredStrategy. When num_workers = 1 it uses '
    'MirroredStrategy.')
flags.DEFINE_integer(
    'checkpoint_every_n', 200, 'Integer defining how often we checkpoint.')  # jh: int(input('check_point 생성주기(200이상): '))
flags.DEFINE_boolean('record_summaries', True,
                     ('Whether or not to record summaries defined by the model'
                      ' or the training pipeline. This does not impact the'
                      ' summaries of the loss values which are always'
                      ' recorded.'))


FLAGS = flags.FLAGS

# jh 추가 2021_1215
#config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
#device_count = {'GPU': 1})
#Akerke 20241210 allow GPU full growth
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# config.gpu_options.allow_growth = True


session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
# jh 추가 끝

#필요한 만큼 메모리를 런타임에 할당하는 방법
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    #YSJ: XLA ON을 이용하여 Train Speed Up Type1(Working with Mixed Precision?) : Postion1
    # ENABLE_XLA=1 : XLA ON(True) , ENABLE_XLA=0 : XLA OFF(False), 기본값(=1)은 켜짐(True)_pjs
    tf.config.optimizer.set_jit(os.environ.get('ENABLE_XLA', '1').lower() not in ('0', 'false'))    # xla 활성화 or 비활성화
    print(f"[XLA] JIT enabled?: {tf.config.optimizer.get_jit()}")
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def main(unused_argv):
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')
  tf.config.set_soft_device_placement(True)
  # ── 여기에 추가 ── _06.23
  # (1) annos 폴더에서 이미지 개수 세기
  annos_dir = r"D:\AI_SVT_Training_mk\annotations\annos"
  try:
    num_images = len([
      f for f in os.listdir(annos_dir)
      if os.path.isfile(os.path.join(annos_dir, f))
         and f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
    ])
  except (OSError, FileNotFoundError):
    num_images = 0

  # (2-1) 06.30 추가
  annotation_dir = r"D:\AI_SVT_Training_mk\annotations"
  epoch_file = os.path.join(annotation_dir, "EpochNo.txt")

  # 06.30 _1: 파일 열기부터 파싱까지 한번에 처리
  try:
    with open(epoch_file, "r", encoding="utf-8") as f:
      raw = f.read().strip()
      raw_steps = int(raw)
  except (ValueError, OSError) as e:
    print(f"[WARNING] EpochNo.txt 읽기 실패 ({e}) → 기본값 사용", flush=True)
    raw_steps = num_images * 20  

  # 06.30 _2
  MIN_STEPS = 10000
  MAX_STEPS = 300000

  train_steps = min(max(raw_steps, MIN_STEPS), MAX_STEPS)
  # print(f"[TRAIN_STEPS] {train_steps}", flush=True)  # 추가_ 콘솔 출력용

  # (3) FLAGS.num_train_steps 덮어쓰기  (실제 적용된 학습 스텝 수)
  FLAGS.num_train_steps = train_steps
  # print(f"[OVERRIDE_NUM_TRAIN_STEPS] {FLAGS.num_train_steps}", flush=True)
  # ── 추가 끝 ──

  #YSJ: Turn On 'Mixed Precision' Train Speed Up Type2(Verified)
  # 기존인데 주석처리함 _09.01
  # from tensorflow.keras.mixed_precision import experimental as mixed_precision
  # policy = mixed_precision.Policy('mixed_float16')    # 혼합 정밀도 정책 (딥러닝 학습에서 권장)
  # mixed_precision.set_policy(policy)                  # float16 : 모든 연산과 변수 저장을 얘로 하는데 메모리 사용은 줄지만 수치가 불안정
  
  # 09.01 _추가 pjs
  # from tensorflow.keras.mixed_precision import experimental as mixed_precision
  from tensorflow.keras import mixed_precision    # 12.17
  # mixed_precision.set_policy(mixed_precision.Policy('float32'))   # float32 : 모든 연산과 변수를 float32로 저장 (기본값)
  policy = mixed_precision.Policy('mixed_float16')    # 혼합 정밀도 정책 : 빠르고, 메모리 절약, 정확도
  mixed_precision.set_global_policy(policy)
  print("[AMP] policy:", mixed_precision.global_policy().name)

  if FLAGS.checkpoint_dir:
    model_lib_v2.eval_continuously(
        pipeline_config_path=FLAGS.pipeline_config_path,
        model_dir=FLAGS.model_dir,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            FLAGS.sample_1_of_n_eval_on_train_examples),
        checkpoint_dir=FLAGS.checkpoint_dir,
        wait_interval=10, timeout=FLAGS.eval_timeout)
  else:
    if FLAGS.use_tpu:
      # TPU is automatically inferred if tpu_name is None and
      # we are running under cloud ai-platform.
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name)
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      strategy = tf.distribute.experimental.TPUStrategy(resolver)
    elif FLAGS.num_workers > 1:
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    else:
      strategy = tf.compat.v2.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    # 06.23 _PJS
    with strategy.scope():
      manager = model_lib_v2.train_loop(
          pipeline_config_path=FLAGS.pipeline_config_path,
          model_dir=FLAGS.model_dir,
          train_steps=FLAGS.num_train_steps,
          use_tpu=FLAGS.use_tpu,
          checkpoint_every_n=FLAGS.checkpoint_every_n,
          record_summaries=FLAGS.record_summaries
          )
    if manager is not None:
        final_checkpoint_index = (FLAGS.num_train_steps) // FLAGS.checkpoint_every_n
        manager.save(final_checkpoint_index)
        # print(f"[FINAL CHECKPOINT SAVED at ckpt {final_checkpoint_index}]", flush=True)
        # 정상 종료 시그널
        # print("[TRAIN_DONE]", flush=True)


if __name__ == '__main__':
  tf.compat.v1.app.run()
