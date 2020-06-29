from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from numpy import array
import numpy as np
import cv2
import matplotlib.pyplot as plt
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
# workday = 520 ; weekand = 204
# traindata = ['DonghuEveschoolday', 'DonghuMorschoolday', 'NeihuEveschoolday', 'NeihuEveworkday', 'NeihuMorschoolday', 'NeihuMorworkday', 'Songsanworkday', 'Zhongxiaoworkday']
# traindata = ['Songsanweekend', 'Zhongxiaoweekend']
traindata = ['Zhongxiaoworkday']
# for j in range(15):
#     for i in traindata:
#         X_train_Age1, y_train_Age1 = readfile(os.path.join(workspace_dir, i + "/Age1"), True)
#         X_train_Age1 = X_train_Age1[:, :, :, 0].reshape(520, 3, 3, 1)
#         X_train_Age2 = readfile(os.path.join(workspace_dir, i + "/Age2"), False)
#         X_train_Age2 = X_train_Age2[:, :, :, 0].reshape(520, 3, 3, 1)
#         out = np.concatenate((X_train_Age1, X_train_Age2), axis=3)
#         X_train_Age3 = readfile(os.path.join(workspace_dir, i + "/Age3"), False)
#         X_train_Age3 = X_train_Age3[:, :, :, 0].reshape(520, 3, 3, 1)
#         out = np.concatenate((out, X_train_Age3), axis=3)
#         X_train_Age4 = readfile(os.path.join(workspace_dir, i + "/Age4"), False)
#         X_train_Age4 = X_train_Age4[:, :, :, 0].reshape(520, 3, 3, 1)
#         out = np.concatenate((out, X_train_Age4), axis=3)
#         X_train_Age5 = readfile(os.path.join(workspace_dir, i + "/Age5"), False)
#         X_train_Age5 = X_train_Age5[:, :, :, 0].reshape(520, 3, 3, 1)
#         out = np.concatenate((out, X_train_Age5), axis=3)
#         X_train_Age6 = readfile(os.path.join(workspace_dir, i + "/Age6"), False)
#         X_train_Age6 = X_train_Age6[:, :, :, 0].reshape(520, 3, 3, 1)
#         out = np.concatenate((out, X_train_Age6), axis=3)
#         X_train_total = readfile(os.path.join(workspace_dir, i + "/Total"), False)
#         X_train_total = X_train_total[:, :, :, 0].reshape(520, 3, 3, 1)
#         out = np.concatenate((out, X_train_total), axis=3)
#         # print(out.shape)
#         # print(out)
#         # print('----')
#         # print(out[:, 2, 2, :])
#         model = ybcl.yb_model(y_train_Age1, out, 3)
#         # 訓練
#
#         model.train(1, '', BATCH_SIZE, i, j)
# 預測
for i in traindata:
    X_train_Age1, y_train_Age1 = readfile(os.path.join(workspace_dir, i + "/Age1"), True)
    X_train_Age1 = X_train_Age1[:, :, :, 0].reshape(520, 3, 3, 1)
    X_train_Age2 = readfile(os.path.join(workspace_dir, i + "/Age2"), False)
    X_train_Age2 = X_train_Age2[:, :, :, 0].reshape(520, 3, 3, 1)
    out = np.concatenate((X_train_Age1, X_train_Age2), axis=3)
    X_train_Age3 = readfile(os.path.join(workspace_dir, i + "/Age3"), False)
    X_train_Age3 = X_train_Age3[:, :, :, 0].reshape(520, 3, 3, 1)
    out = np.concatenate((out, X_train_Age3), axis=3)
    X_train_Age4 = readfile(os.path.join(workspace_dir, i + "/Age4"), False)
    X_train_Age4 = X_train_Age4[:, :, :, 0].reshape(520, 3, 3, 1)
    out = np.concatenate((out, X_train_Age4), axis=3)
    X_train_Age5 = readfile(os.path.join(workspace_dir, i + "/Age5"), False)
    X_train_Age5 = X_train_Age5[:, :, :, 0].reshape(520, 3, 3, 1)
    out = np.concatenate((out, X_train_Age5), axis=3)
    X_train_Age6 = readfile(os.path.join(workspace_dir, i + "/Age6"), False)
    X_train_Age6 = X_train_Age6[:, :, :, 0].reshape(520, 3, 3, 1)
    out = np.concatenate((out, X_train_Age6), axis=3)
    X_train_total = readfile(os.path.join(workspace_dir, i + "/Total"), False)
    X_train_total = X_train_total[:, :, :, 0].reshape(520, 3, 3, 1)
    out = np.concatenate((out, X_train_total), axis=3)

model = ybcl.yb_model(y_train_Age1, out, 3)
trainX, trainY, testX, testY = model.create_timestamp_X_Y('Zhongxiaoworkday')
yb = model.cnn_model(3, BATCH_SIZE)
yb.load_weights('Zhongxiaoworkday\weight\weight15\mse_model_0001200.h5')
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