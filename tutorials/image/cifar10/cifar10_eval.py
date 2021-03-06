# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10
import config


cifar10.NUM_CLASSES = config.NUM_CLASSES
#FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('eval_dir', './cifar10_eval',
#                           """Directory where to write event logs.""")
#tf.app.flags.DEFINE_string('eval_data', 'test',
#                           """Either 'test' or 'train_eval'.""")
#tf.app.flags.DEFINE_string('checkpoint_dir', './cifar10_train',
#                           """Directory where to read model checkpoints.""")
#tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
#                            """How often to run the eval.""")
#tf.app.flags.DEFINE_integer('num_examples', 10000,
#                            """Number of examples to run.""")
#tf.app.flags.DEFINE_boolean('run_once', False,
#                         """Whether to run eval only once.""")
parser = cifar10.parser

parser.add_argument('--eval_dir', type=str, default='/tmp/cifar10_eval',
                    help='Directory where to write event logs.')

parser.add_argument('--eval_data', type=str, default='test',
                    help='Either `test` or `train_eval`.')

parser.add_argument('--checkpoint_dir', type=str, default='/tmp/cifar10_train',
                    help='Directory where to read model checkpoints.')

parser.add_argument('--eval_interval_secs', type=int, default=60*5,
                    help='How often to run the eval.')

parser.add_argument('--num_examples', type=int, default=10000,
                    help='Number of examples to run.')

parser.add_argument('--run_once', type=bool, default=False,
                    help='Whether to run eval only once.')


#def eval_once(saver, summary_writer, top_k_op, summary_op):
def eval_once(saver, summary_writer, top_k_op, summary_op, labels, top_1_op): # test
  """Run Eval once.
  １回評価を実施する
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      # チェックポイントファイル（変数を保存するファイル？）から復元する
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    # キューの起動？を開始する
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions. # 正しい予測の数字をカウントする
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      predictions_1_op = [0 for i in range(cifar10.NUM_CLASSES)] # test
      inputs_1_op = [0 for i in range(cifar10.NUM_CLASSES)] # test
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1
        prediction, label = sess.run([top_1_op, labels]) # test
        #print(prediction) # test
        #print(label) # test
        # class part prediction count
        label_step = 0
        for l in label:
          inputs_1_op[l] += 1
          if l in prediction.indices[label_step]:
            predictions_1_op[l] += 1
         
          label_step += 1
      # Compute precision @ 1.
      # 予測を計算する
      precision = true_count / total_sample_count
      #print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      
      # part of label prediction
      names = sorted(config.label_names)
      for l in range(cifar10.NUM_CLASSES):
        print('%s: precision[%s:%s] @ 1 = %.3f' % (datetime.now(), l, names[l], predictions_1_op[l]/inputs_1_op[l]))
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  # ステップ番号ぶん、CIFAR-10を評価する
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    # CIFAR-10のための画像とラベルを取得
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    # 推論モデルからロジットの予測を計算したグラフを作る
    logits = cifar10.inference(images)

    # Calculate predictions.
    # 予測を計算する
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    top_1_op = tf.nn.top_k(logits, 1) # test

    # Restore the moving average version of the learned variables for eval.
    # evalのために学習した変数の移動平均のバージョンを復元する
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    # TensorFlowのサマリーの収集を基準にサマリーの操作を作る
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      #eval_once(saver, summary_writer, top_k_op, summary_op)
      eval_once(saver, summary_writer, top_k_op, summary_op, labels, top_1_op) # test
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  tf.app.run()
