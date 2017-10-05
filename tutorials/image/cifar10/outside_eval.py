import sys
import tensorflow as tf
 
#from tensorflow.models.image.cifar10 import cifar10
import cifar10
 
import config
FLAGS = tf.app.flags.FLAGS
 
cifar10.NUM_CLASSES = config.NUM_CLASSES
 
tf.app.flags.DEFINE_string('checkpoint_dir', './cifar10_train',
                           """Directory where to read model checkpoints.""")
 
def evaluate(filename):
  # filename:画像ファイルのパス
  with tf.Graph().as_default() as g:
    jpg = tf.read_file(filename)
    image = tf.image.decode_jpeg(jpg, channels = 3)
    image = tf.image.resize_images(image, [32,32]) 
    image = tf.image.resize_image_with_crop_or_pad(image, 24, 24) # cifar10は内部処理で32×32を24×24に切り出して利用している
    logits = cifar10.inference([image])
 
    top_k_op = tf.nn.top_k(logits,k=config.NUM_CLASSES)
    # ここのkの値もクラス数と一致させるようにしてください
 
    saver = tf.train.Saver()
 
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint file found')
      return
 
    tf.train.start_queue_runners(sess=sess)
    values, indices = sess.run(top_k_op)
    ratio = sess.run(tf.nn.softmax(values[0]))
    # 予想したラベルとそれぞれに対する確信度
    names = sorted(config.label_names)
    print('[' + ' '.join([ '%s:%s' % (l, names[l]) for l in indices[0] ]) + ']')
    print(indices[0])
    print(ratio)
def main(argv=None):
  evaluate(sys.argv[1])
 
if __name__ == '__main__':
  tf.app.run()
