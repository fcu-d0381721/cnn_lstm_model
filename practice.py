import numpy as np
import os
import cv2
# import matplotlib.pyplot as plt

# def readfile(path, label):
#     # label 是一個 boolean variable，代表需不需要回傳 y 值
#     image_dir = sorted(os.listdir(path))
#     x = np.zeros((len(image_dir), 3, 3, 3), dtype=np.uint8)
#     y = np.zeros((len(image_dir)), dtype=np.uint8)
#     for i, file in enumerate(image_dir):
#         img = cv2.imread(os.path.join(path, file))
#         x[i, :, :] = cv2.resize(img, (3, 3))
#         if label:
#           tmp = file.split("-")[3]
#           y[i] = int(tmp.split(".")[0])
#     if label:
#       return x, y
#     else:
#       return x
#
# #
# x,y = np.mgrid[0:3, -2:1]
# print(x)
# print('---------')
# print(y)
# sigma = 0.707
# gaussian_kernel = np.exp(-0.5*((x**2+y**2)/sigma**2))  #right
# print(gaussian_kernel)
# gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
# print(gaussian_kernel)
# # np.array()
# # u = np.zeros(shape=(3, 3, 1))
# print(gaussian_kernel.shape)
# gaussian_kernel = gaussian_kernel.reshape(3, 3, 1)
# workspace_dir = './train'
#
# X_train_Age1, y_train_Age1 = readfile(os.path.join(workspace_dir, "run/Age1"), True)
# X_train_Age1 = X_train_Age1[:, :, :, 0].reshape(4344, 3, 3, 1)
# norm = np.linalg.norm(X_train_Age1)
# X_train_Age1 = X_train_Age1 / norm
#
# o = X_train_Age1[0, :, :, :]
# print(o)
# print(o*gaussian_kernel)


# from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
import time

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.create_nine_grid()
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        init = tf.random_uniform_initializer(minval=0.1, maxval=0.9)
        self.sigma = self.add_weight(name='sigma', shape=(189, ), initializer=init, trainable=True)
        # self.sigma = self.add_weight(name='sigma', shape=input_shape, initializer=init, trainable=True)

        # make Gaussian Filter
        all_x_y = tf.reshape(self.x_y_all, [189, 9])
        out = all_x_y / self.sigma[:, None]**2
        out_exp = tf.math.exp(-0.5 * out)
        out_sum = tf.reduce_sum(out_exp, 1)
        self.nor_out = out_exp / out_sum[:, None]
        self.nor_out = tf.reshape(self.nor_out, [189, 3, 3, 1])
        print(self.nor_out)
        super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

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
        self.x_y_ = x_al**2 + y_al**2
        self.x_y_ = tf.reshape(self.x_y_, [1, 9, 3, 3])
        self.x_y_all = tf.keras.backend.repeat_elements(self.x_y_, 21, 0)
        self.x_y_all = tf.dtypes.cast(self.x_y_all, tf.float32)


    def call(self, x):
        y = tf.ones(shape=(1, 3, 3, 1))
        # print(y)
        # print('****')
        # print(K.expand_dims(y, axis=-1))
        # aligned_x = K.repeat_elements(y, 9, 0)
        aligned_mean = self.sigma
        # print(aligned_mean)
        # print(aligned_x)
        #
        # exp = np.exp(-0.5*((x**2+y**2)/aligned_mean**2))
        # xc = exp
        # print(K.flatten(xc))
        # return K.batch_flatten(xc)  # xc / K.maximum(sums, less)

d = MyLayer(1)
d.build(2)
d.call(1)