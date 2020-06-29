import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
import YB_Fuzzy as f
from numpy import array
import numpy as np
import cv2

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


def split_sequence(sequence, n_steps):

    X, y = list(), list()
    for i in range(len(sequence[1])):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence[1]) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[0][i:end_ix, :, :, :], sequence[1][end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

json_file = open(r'test\architecture\cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json,custom_objects={'FuzzyLayer': f.FuzzyLayer})
model.load_weights('test\weight\mse_model_0000300.h5')
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.01))

test_stamp = 48
workspace_dir = './train'
# checkpoint_dir = 'training_checkpoint'
# weight_filepath = './training_weight/'

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

norm = np.linalg.norm(out)
X_train_Age1 = out / norm

y_train_Age2 = y_train_Age1 / y_train_Age1.max(axis=0)
X_train_Age1 = X_train_Age1.astype(np.float32)
X, y = split_sequence([X_train_Age1, y_train_Age2], 3)

train_X = X[0:(len(X) - test_stamp), :, :, :]
test_X = X[-test_stamp::]
train_Y = y[0:(len(X) - test_stamp)]
test_Y = y[-test_stamp::]

print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)


yhat = model.predict(test_X, verbose=0)
# print(yhat)
ori_ = yhat * y_train_Age1.max(axis=0)
predicted_y = ori_[:, 0]
target_y = y_train_Age1[-test_stamp::]
out = tf.reduce_mean(target_y - predicted_y)
mse = mean_squared_error(target_y, predicted_y)
rmse = sqrt(mse)
mape = np.mean(np.abs((target_y - predicted_y) / target_y)) * 100
print('平均差: %f' % out)
print('MSE: %f' % mse)
print('RMSE: %f' % rmse)
print('MAPE: %f' % mape)