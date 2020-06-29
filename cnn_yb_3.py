from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
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
    return np.array(X), np.array(y)


def cnn_model(batchSize):
    # gauss_kernel = gaussian_kernel(9, 5, 1)
    # gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

    model = tf.keras.Sequential()
    # model.add(layers.Input(shape=(3, 64, 64, 7), batch_size=batchSize))  # , batch_input_shape=(batchSize, 1, )))
    model.add(layers.TimeDistributed(layers.Conv2D(14, (2, 2), padding='same', strides=2), input_shape=(3, 3, 3, 7),
                                     batch_size=batchSize))
    # model.add(layers.TimeDistributed(layers.AveragePooling2D(pool_size=(2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(28, (2, 2))))
    model.add(layers.TimeDistributed(layers.BatchNormalization()))
    model.add(layers.TimeDistributed(layers.Activation("relu")))
    # model.add(layers.TimeDistributed(layers.AveragePooling2D(pool_size=(2, 2))))

    # model.add(layers.TimeDistributed(layers.Conv2D(56, (3, 3))))
    # model.add(layers.TimeDistributed(layers.BatchNormalization()))
    # model.add(layers.TimeDistributed(layers.Activation("relu")))

    model.add(layers.TimeDistributed(layers.GlobalAveragePooling2D()))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.LSTM(100, stateful=True, return_sequences=True))
    model.add(layers.LSTM(25, stateful=True))
    model.add(layers.Dense(1))

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.01))

    return model

BATCH_SIZE = 3
test_stamp = 48
# scheduler = 1e-3

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

y_train_Age1 = y_train_Age1 / y_train_Age1.max(axis=0)
X_train_Age1 = X_train_Age1.astype(np.float32)
X, y = split_sequence([X_train_Age1, y_train_Age1], 3)

train_X = X[0:(len(X) - test_stamp), :, :, :]
test_X = X[-test_stamp::]
train_Y = y[0:(len(X) - test_stamp)]
test_Y = y[-test_stamp::]

print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)

logdir = os.path.join('run_model_31\logs\scalars', datetime.now().strftime("%Y%m%d-%H%M%S"))
architecture_filepath = os.path.join(r'run_model_31\architecture', 'cnn.json')
weight_filepath = os.path.join('run_model_31\weight', 'mse_model_{epoch:07d}.h5')

def scheduler(epoch):
  if epoch < 100:
    return 0.001
  elif epoch < 200:
    return 0.0001
  elif epoch < 300:
    return 0.00001
  elif epoch < 400:
    return 0.000001

tbCallBack = TensorBoard(log_dir=logdir,
                         histogram_freq=0,
                         write_graph=True,
                         write_grads=True,
                         write_images=True,
                         update_freq='epoch',
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

checkpoint = ModelCheckpoint(weight_filepath, monitor='loss', verbose=2, save_weights_only=True, period=10)

callbacks_list = [tbCallBack, checkpoint, scheduler]

model = cnn_model(BATCH_SIZE)


def train(initial, architecture_filepath, train_X, train_Y, test_X, test_Y, callbacks_list, model):
    if initial == 1:
        training_history = model.fit(
            train_X,  # input
            train_Y,  # output
            batch_size=3,
            verbose=0,  # Suppress chatty output; use Tensorboard instead
            epochs=300,
            validation_data=(test_X, test_Y),
            callbacks=callbacks_list
        )
    else:
        model.load_weights('run_model_31\weight\mse_model_0000050.h5')
        training_history = model.fit(
            train_X,  # input
            train_Y,  # output
            batch_size=3,
            verbose=0,  # Suppress chatty output; use Tensorboard instead
            epochs=150,
            validation_data=(test_X, test_Y),
            callbacks=callbacks_list,
            initial_epoch=50
        )
    model_json = model.to_json()
    with open(architecture_filepath, "w") as json_file:
        json_file.write(model_json)

train(1, architecture_filepath, train_X, train_Y, test_X, test_Y, callbacks_list, model)