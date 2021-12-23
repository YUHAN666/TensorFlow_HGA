import tensorflow as tf
import os


class TensorboardManager(object):

    def __init__(self, sess, tensorboard_dir, need_clear=False):
        self.session = sess
        self.tensorboard_dir = tensorboard_dir
        if need_clear:
            for _, _, files in os.walk(self.tensorboard_dir):
                for file in files:
                    os.remove(os.path.join(self.tensorboard_dir, file))
        # tensorboard --logdir=E:\CODES\TensorFlow_PZT\tensorboard
        self.writer = tf.summary.FileWriter(self.tensorboard_dir, self.session.graph)
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

    def add_summary(self, summary, step):
        self.writer.add_summary(summary, step)

    def close(self):
        self.writer.close()
