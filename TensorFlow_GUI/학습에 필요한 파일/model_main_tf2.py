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


import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2

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
# flags.DEFINE_integer('num_train_steps', None, 'Number of training steps')   # 06.30

# jh 추가 2021_1215
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5))
#device_count = {'GPU': 1}

# config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
# jh 추가 끝

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

  # # (2) 동적 스텝 계산 및 상·하한 적용
  # iMaxTrainStep = min(max(num_images * 20, 10000), 100000)
  # # iMaxTrainStep = num_images * 200
  # # iMaxTrainStep = min(iMaxTrainStep, 100000)
  # # iMaxTrainStep = max(iMaxTrainStep, 10000)

  # # (3) FLAGS.num_train_steps 덮어쓰기
  FLAGS.num_train_steps = train_steps
  # print(f"[OVERRIDE_NUM_TRAIN_STEPS] {FLAGS.num_train_steps}", flush=True)
  # ── 추가 끝 ──
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
