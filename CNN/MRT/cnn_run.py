from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from numpy import array
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import model_CNN_LSTM as ybcl

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
# workday = 1044 ; weekand = 416
# traindata = ['zhongciao_work_in', 'zhongciao_work_out', 'cityhall_work_in', 'cityhall_work_out', 'ganchain_work_in',
#              'ganchain_work_out', 'songgan_work_in', 'songgan_work_out', 'taipei_work_in', 'taipei_work_out',
#              'techbuil_work_in', 'techbuil_work_out', 'xiban_work_in', 'xiban_work_out']
# traindata = ['techbuil_work_in', 'techbuil_work_out', 'xiban_work_in', 'xiban_work_out']
# traindata = ['zhongciao_week_in', 'zhongciao_week_out', 'cityhall_week_in', 'cityhall_week_out', 'ganchain_week_in',
#              'ganchain_week_out', 'songgan_week_in', 'songgan_week_out', 'taipei_week_in', 'taipei_week_out',
#              'techbuil_week_in', 'techbuil_week_out', 'xiban_week_in', 'xiban_week_out']
traindata = ['zhongciao_work_in']
# for j in range(1, 15):
#     for i in traindata:
#         X_train_Age1, y_train_Age1 = readfile(os.path.join(workspace_dir, i + "/Age1"), True)
#         X_train_Age1 = X_train_Age1[:, :, :, 0].reshape(416, 3, 3, 1)
#         X_train_Age2 = readfile(os.path.join(workspace_dir, i + "/Age2"), False)
#         X_train_Age2 = X_train_Age2[:, :, :, 0].reshape(416, 3, 3, 1)
#         out = np.concatenate((X_train_Age1, X_train_Age2), axis=3)
#         X_train_Age3 = readfile(os.path.join(workspace_dir, i + "/Age3"), False)
#         X_train_Age3 = X_train_Age3[:, :, :, 0].reshape(416, 3, 3, 1)
#         out = np.concatenate((out, X_train_Age3), axis=3)
#         X_train_Age4 = readfile(os.path.join(workspace_dir, i + "/Age4"), False)
#         X_train_Age4 = X_train_Age4[:, :, :, 0].reshape(416, 3, 3, 1)
#         out = np.concatenate((out, X_train_Age4), axis=3)
#         X_train_Age5 = readfile(os.path.join(workspace_dir, i + "/Age5"), False)
#         X_train_Age5 = X_train_Age5[:, :, :, 0].reshape(416, 3, 3, 1)
#         out = np.concatenate((out, X_train_Age5), axis=3)
#         X_train_Age6 = readfile(os.path.join(workspace_dir, i + "/Age6"), False)
#         X_train_Age6 = X_train_Age6[:, :, :, 0].reshape(416, 3, 3, 1)
#         out = np.concatenate((out, X_train_Age6), axis=3)
#         X_train_total = readfile(os.path.join(workspace_dir, i + "/Total"), False)
#         X_train_total = X_train_total[:, :, :, 0].reshape(416, 3, 3, 1)
#         out = np.concatenate((out, X_train_total), axis=3)
#         model = ybcl.yb_model(y_train_Age1, out, 3)
#         # 訓練
#         model.train(1, '', BATCH_SIZE, i, j)
# 預測
for i in traindata:
    X_train_Age1, y_train_Age1 = readfile(os.path.join(workspace_dir, i + "/Age1"), True)
    X_train_Age1 = X_train_Age1[:, :, :, 0].reshape(1044, 3, 3, 1)
    X_train_Age2 = readfile(os.path.join(workspace_dir, i + "/Age2"), False)
    X_train_Age2 = X_train_Age2[:, :, :, 0].reshape(1044, 3, 3, 1)
    out = np.concatenate((X_train_Age1, X_train_Age2), axis=3)
    X_train_Age3 = readfile(os.path.join(workspace_dir, i + "/Age3"), False)
    X_train_Age3 = X_train_Age3[:, :, :, 0].reshape(1044, 3, 3, 1)
    out = np.concatenate((out, X_train_Age3), axis=3)
    X_train_Age4 = readfile(os.path.join(workspace_dir, i + "/Age4"), False)
    X_train_Age4 = X_train_Age4[:, :, :, 0].reshape(1044, 3, 3, 1)
    out = np.concatenate((out, X_train_Age4), axis=3)
    X_train_Age5 = readfile(os.path.join(workspace_dir, i + "/Age5"), False)
    X_train_Age5 = X_train_Age5[:, :, :, 0].reshape(1044, 3, 3, 1)
    out = np.concatenate((out, X_train_Age5), axis=3)
    X_train_Age6 = readfile(os.path.join(workspace_dir, i + "/Age6"), False)
    X_train_Age6 = X_train_Age6[:, :, :, 0].reshape(1044, 3, 3, 1)
    out = np.concatenate((out, X_train_Age6), axis=3)
    X_train_total = readfile(os.path.join(workspace_dir, i + "/Total"), False)
    X_train_total = X_train_total[:, :, :, 0].reshape(1044, 3, 3, 1)
    out = np.concatenate((out, X_train_total), axis=3)

model = ybcl.yb_model(y_train_Age1, out, 3)
trainX, trainY, testX, testY = model.create_timestamp_X_Y('zhongciao_work_in')
yb = model.cnn_model(3, BATCH_SIZE)
yb.load_weights('zhongciao_work_in\weight\weight15\mse_model_0002000.h5')
yb.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.01))

yhat = yb.predict(testX, verbose=0)

ori_ = yhat * y_train_Age1.max(axis=0)
predicted_y = ori_[:, 0]
target_y = testY * y_train_Age1.max(axis=0)
out = tf.reduce_mean(target_y - predicted_y)
mse = mean_squared_error(target_y, predicted_y)
# print(target_y, predicted_y)
plt.plot(target_y, color='r', label="target_y")
plt.plot(predicted_y, color='g', label="predicted_y")
plt.show()
rmse = sqrt(mse)
mape = np.mean(np.abs((target_y - predicted_y) / target_y)) * 100
print('平均差: %f' % out)
print('MSE: %f' % mse)
print('RMSE: %f' % rmse)
print('MAPE: %f' % mape)