import tensorflow as tf
import numpy as np
class RunningMeanStd:
    def __init__(self, epsilon=1e-2, shape=()):
        self._sum = tf.get_variable(
            dtype=tf.float32,
            shape=shape,
            initializer=tf.constant_initializer(0.0),
            name="runningsum", trainable=False)
        self._sumsq = tf.get_variable(
            dtype=tf.float32,
            shape=shape,
            initializer=tf.constant_initializer(epsilon),
            name="runningsumsq", trainable=False)
        self._count = tf.get_variable(
            dtype=tf.float32,
            shape=(),
            initializer=tf.constant_initializer(epsilon),
            name="count", trainable=False)
        self.shape =shape

        self.mean = tf.to_float(self._sum / self._count)
        self.std = tf.sqrt(
            tf.maximum(
                tf.to_float(self._sumsq / self._count) - tf.square(self.mean), 1e-2
            )
        )
        self.newsum = tf.placeholder(shape=self.shape, dtype=tf.float32, name='sum')
        self.newsumsq = tf.placeholder(shape=self.shape, dtype=tf.float32, name='var')
        self.newcount = tf.placeholder(shape=[], dtype=tf.float32, name='count')
        self.update_sum = tf.assign_add(self._sum, self.newsum)
        self.update_sumsq = tf.assign_add(self._sumsq, self.newsumsq)
        self.update_count = tf.assign_add(self._count, self.newcount)

    def update(self, x):
        sess = tf.get_default_session()
        sess.run([self.update_sum, self.update_sumsq, self.update_count],
                 feed_dict={
                     self.newsum: np.sum(x, axis=0),
                     self.newsumsq: np.sum(np.square(x), axis=0),
                     self.newcount: len(x)
                 })

def test_runningmeanstd():
    means = [2.0, 1.0]
    stds = [1.0, 3.0]
    x = np.random.randn(1000, 3)*stds[0] + means[0]
    y = np.random.randn(1000, 3)*stds[1] + means[1]
    z = np.concatenate([x, y], axis=0)
    with tf.Session() as sess:
        rms = RunningMeanStd(epsilon=0.0, shape=[3])
        sess.run(tf.global_variables_initializer())
        rms.update(x)
        print(sess.run([rms.mean, rms.std]))
        rms.update(y)
        total_mean, total_std = sess.run([rms.mean, rms.std])
        print(total_mean, total_std)
        z_mean, z_std = np.mean(z, axis=0), np.std(z, axis=0)
        print(z_mean, z_std)
        assert np.allclose(total_mean, z_mean)
        assert np.allclose(total_std, z_std)