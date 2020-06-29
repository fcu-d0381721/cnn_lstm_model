from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.backend import permute_dimensions
from tensorflow.keras.layers import Reshape, Permute
from numpy import array
import numpy as np
import cv2
import time
import YB_Fuzzy as f

class yb_model(object):
    def __init__(self, ybdata, peopledata, look_back):
        self.ybdata = ybdata
        self.peopledata = peopledata
        self.look_back = look_back

    def create_dataset(self, sequence, look_back):
        X, y = list(), list()
        for i in range(len(sequence[1])):
            # find the end of this pattern
            end_ix = i + look_back
            # check if we are beyond the sequence
            if end_ix > len(sequence[1]) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[0][i:end_ix, :, :, :], sequence[1][end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    # 切割資料為85:15等分
    def split_train_test(self):
        norm = np.linalg.norm(self.peopledata)
        self.peopledata = self.peopledata / norm
        ybdata = self.ybdata / self.ybdata.max(axis=0)
        ybdata = ybdata.astype(np.float32)

        train_size = 48
        train_P, test_P = self.peopledata[0:(len(ybdata) - train_size), :, :, :], self.peopledata[-train_size::]
        train_YB, test_YB = ybdata[0:(len(ybdata) - train_size)], ybdata[-train_size::]
        return train_P, test_P, train_YB, test_YB

    # 將資料整理成timestamp格式
    def create_timestamp_X_Y(self):
        train_P, test_P, train_YB, test_YB = self.split_train_test()
        look_back = self.look_back
        trainX, trainY = self.create_dataset([train_P, train_YB], look_back)
        testX, testY = self.create_dataset([test_P, test_YB], look_back)
        return trainX, trainY, testX, testY

    def cnn_model(self, timestamp, batchSize):

        model = tf.keras.Sequential()
        model.add(layers.LSTM(100, return_sequences=True))  # 錯誤在這裡
        model.add(layers.LSTM(25))
        model.add(layers.Dense(1))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.01))

        return model

    def train(self, initial, load_weights, batchsize):
        look_back = self.look_back
        trainX, trainY, testX, testY = self.create_timestamp_X_Y()
        model = self.cnn_model(look_back, batchsize)
        # start = time.time()
        logdir = os.path.join('test\logs\scalars', datetime.now().strftime("%Y%m%d-%H%M%S"))
        architecture_filepath = os.path.join(r'test\architecture', 'cnn.json')
        weight_filepath = os.path.join('test\weight', 'mse_model_{epoch:07d}.h5')

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

        if initial == 1:
            training_history = model.fit(
                trainX,  # input
                trainY,  # output
                batch_size=batchsize,
                verbose=0,  # Suppress chatty output; use Tensorboard instead
                epochs=300,
                validation_data=(testX, testY),
                callbacks=callbacks_list
            )
        else:
            model.load_weights('test\weight\mse_model_' + load_weights + '.h5')
            training_history = model.fit(
                trainX,  # input
                trainY,  # output
                batch_size=batchsize,
                verbose=0,  # Suppress chatty output; use Tensorboard instead
                epochs=150,
                validation_data=(testX, testY),
                callbacks=callbacks_list,
                initial_epoch=50
            )
        model_json = model.to_json()
        with open(architecture_filepath, "w") as json_file:
            json_file.write(model_json)


