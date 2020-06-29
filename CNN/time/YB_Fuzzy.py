import tensorflow as tf
import numpy as np
import time


class FuzzyLayer(tf.keras.layers.Layer):

    #可變動參數fuzzy_size
    def __init__(self, output_dim, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = output_dim
        self.create_nine_grid()
        super(FuzzyLayer, self).__init__(**kwargs)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim
        })
        return config

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        init = tf.random_uniform_initializer(minval=0.1, maxval=0.9)
        self.sigma = self.add_weight(name='sigma', shape=(189,), initializer=init, trainable=True)

        super(FuzzyLayer, self).build(input_shape)  # 一定要在最后调用它

    def create_nine_grid(self):
        x, y = np.mgrid[-1:2, -1:2]
        x1, y1 = np.mgrid[-2:1, -2:1]
        x_al = np.append(x, x1, axis=0)
        y_al = np.append(y, y1, axis=0)
        x, y = np.mgrid[0:3, 0:3]
        x_al = np.append(x_al, x, axis=0)
        y_al = np.append(y_al, y, axis=0)
        x, y = np.mgrid[-2:1, -1:2]
        x_al = np.append(x_al, x, axis=0)
        y_al = np.append(y_al, y, axis=0)
        x, y = np.mgrid[-2:1, 0:3]
        x_al = np.append(x_al, x, axis=0)
        y_al = np.append(y_al, y, axis=0)
        x, y = np.mgrid[-1:2, 0:3]
        x_al = np.append(x_al, x, axis=0)
        y_al = np.append(y_al, y, axis=0)
        x, y = np.mgrid[-1:2, -2:1]
        x_al = np.append(x_al, x, axis=0)
        y_al = np.append(y_al, y, axis=0)
        x, y = np.mgrid[0:3, -1:2]
        x_al = np.append(x_al, x, axis=0)
        y_al = np.append(y_al, y, axis=0)
        x, y = np.mgrid[0:3, -2:1]
        x_al = np.append(x_al, x, axis=0)
        y_al = np.append(y_al, y, axis=0)
        x_al = x_al.reshape(9, 3, 3)
        y_al = y_al.reshape(9, 3, 3)
        x_al = tf.Variable(x_al)
        y_al = tf.Variable(y_al)
        self.x_y_ = x_al ** 2 + y_al ** 2
        self.x_y_ = tf.reshape(self.x_y_, [1, 9, 3, 3])
        self.x_y_all = tf.keras.backend.repeat_elements(self.x_y_, 21, 0)
        self.x_y_all = tf.dtypes.cast(self.x_y_all, tf.float32)

    def call(self, input):

        # print(input)

        # make Gaussian Filter

        all_x_y = tf.reshape(self.x_y_all, [189, 9])
        out = all_x_y / self.sigma[:, None] ** 2
        out_exp = tf.math.exp(-0.5 * out)
        out_sum = tf.reduce_sum(out_exp, 1)
        self.nor_out = out_exp / out_sum[:, None]
        self.nor_out = tf.reshape(self.nor_out, [189, 3, 3])

        x_input = tf.keras.backend.repeat_elements(input, 9, 1)
        output = x_input*self.nor_out
        # print(output)
        return output

    # def compute_output_shape(self, input_shape):
    #     print(input_shape[0], 3, 3, 3, self.output_dim)
    #
    #     return (input_shape[0], 189, 3, 3)
