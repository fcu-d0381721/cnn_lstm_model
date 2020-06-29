from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from numpy import array
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error
from math import sqrt
import YBmodel_CNN_LSTM as ybcl

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 3, 3, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (3, 3))
        if label:
          tmp = file.split("-")[3]
          y[i] = int(tmp.split(".")[0])
    if label:
      return x, y
    else:
      return x


BATCH_SIZE = 3

workspace_dir = './train'

X_train_Age1, y_train_Age1 = readfile(os.path.join(workspace_dir, "run/Age1"), True)
X_train_Age1 = X_train_Age1[:, :, :, 0].reshape(4344, 3, 3, 1)
X_train_Age2 = readfile(os.path.join(workspace_dir, "run/Age2"), False)
X_train_Age2 = X_train_Age2[:, :, :, 0].reshape(4344, 3, 3, 1)
out = np.concatenate((X_train_Age1, X_train_Age2), axis=3)
X_train_Age3 = readfile(os.path.join(workspace_dir, "run/Age3"), False)
X_train_Age3 = X_train_Age3[:, :, :, 0].reshape(4344, 3, 3, 1)
out = np.concatenate((out, X_train_Age3), axis=3)
X_train_Age4 = readfile(os.path.join(workspace_dir, "run/Age4"), False)
X_train_Age4 = X_train_Age4[:, :, :, 0].reshape(4344, 3, 3, 1)
out = np.concatenate((out, X_train_Age4), axis=3)
X_train_Age5 = readfile(os.path.join(workspace_dir, "run/Age5"), False)
X_train_Age5 = X_train_Age5[:, :, :, 0].reshape(4344, 3, 3, 1)
out = np.concatenate((out, X_train_Age5), axis=3)
X_train_Age6 = readfile(os.path.join(workspace_dir, "run/Age6"), False)
X_train_Age6 = X_train_Age6[:, :, :, 0].reshape(4344, 3, 3, 1)
out = np.concatenate((out, X_train_Age6), axis=3)
X_train_total = readfile(os.path.join(workspace_dir, "run/Total"), False)
X_train_total = X_train_total[:, :, :, 0].reshape(4344, 3, 3, 1)
out = np.concatenate((out, X_train_total), axis=3)
# print(out.shape)

# from keras import backend as K
# tmp = out[0:3, :, :, :]
# # print(tmp)
# C_t = K.permute_dimensions(tmp, (0, 3, 1, 2))
# # print(C_t)
#
# C_t = K.reshape(C_t, (21, 3, 3))
# x_input = K.repeat_elements(C_t, 9, 0)
# # print(x_input)
# i = K.reshape(x_input, (189, 9))
# r = K.reshape(i, (3, 63, 9))
# u = K.permute_dimensions(r, (0, 2, 1))
# print(u)
# o = K.reshape(u, (3, 3, 3, 63))
# print(o)


# --------------------------- numpy
# tmp = out[0:3, :, :, :]
# tmp = np.transpose(tmp, (0, 3, 1, 2)).reshape(21, 3, 3, 1)
# tmp = tf.Variable(tmp)
# tmp = tf.keras.backend.repeat_elements(tmp, 9, 0)
# tmp = tf.dtypes.cast(tmp, tf.float32)
# # print(tmp)
# d = tf.reshape(tmp, [21, 3, 3, 9])
# f = np.transpose(d[0:7,:,:,:], (3, 0, 1, 2)).reshape(3, 3, 63)
# s = np.transpose(d[7:14,:,:,:], (3, 0, 1, 2)).reshape(3, 3, 63)
# t = np.transpose(d[14:21,:,:,:], (3, 0, 1, 2)).reshape(3, 3, 63)
# print(np.array([f, s, t]))
# # tmp = tmp.astype(tf.float32)
# # print(tmp)
# # tmp = tmp.reshape(21, 3, 3)
# # print(tmp)
# # tmpout = np.transpose(tmp)
# # print(tmpout.shape)



model = ybcl.yb_model(y_train_Age1, out, 3)
# 訓練
model.train(1, '', BATCH_SIZE)


# 預測
trainX, trainY, testX, testY = model.create_timestamp_X_Y()
yb = model.cnn_model(3, BATCH_SIZE)
yb.load_weights('nonfuzzy\weight\mse_model_0000300.h5')
yb.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.01))

yhat = yb.predict(testX, verbose=0)
# print(yhat)
ori_ = yhat * y_train_Age1.max(axis=0)
predicted_y = ori_[:, 0]
target_y = testY * y_train_Age1.max(axis=0)
out = tf.reduce_mean(target_y - predicted_y)
mse = mean_squared_error(target_y, predicted_y)
rmse = sqrt(mse)
mape = np.mean(np.abs((target_y - predicted_y) / target_y)) * 100
print('平均差: %f' % out)
print('MSE: %f' % mse)
print('RMSE: %f' % rmse)
print('MAPE: %f' % mape)