import pandas as pd
import numpy as np
import time
import fuzzy_layer as f
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.constraints import max_norm


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))



class pm25_model(object):
    def __init__(self, wavelete_data, pm25_data, value_setting):
        self.pm25_data = pm25_data
        self.wavelete = wavelete_data
        self.delete = set([])
        self.column = self.wavelete.columns  # 原始欄位
        self.value_setting = value_setting
        self.look_back = 12

    def create_dataset(self, dataset, select_feature, look_back=12):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back*2-1):
            shift = i+look_back
            a = dataset[i:shift, :select_feature]
            dataX.append(a)
            dataY.append(dataset[(1 + shift):(1 + shift + 1), select_feature])
        return np.array(dataX), np.array(dataY)

    def normalize(self, train):
        train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
        return train_norm

    def denormalize(self, train):
        Y = pd.read_csv(self.pm25_data, index_col=0).values
        denorm = train.apply(lambda x: x*(np.max(Y)-np.min(Y))+np.mean(Y))
        return denorm

    #模糊層神經元的名稱
    def build_fz_name(self, column):
        columns = []
        for fuz in column:
            columns.append(fuz+'_fz1')
            columns.append(fuz+'_fz2')
            columns.append(fuz+'_fz3')
        return columns

    # 不重要欄位
    def delt_unimportant(self, unimport):
        delt = []
        for i in range(len(unimport)):
            delt.append(unimport[i][:-4])
        return delt

     #篩選出在於訓練過程中3個模糊有其中2個以上為0的特徵
    def select_unimportant(self, delt):
        delete = pd.value_counts(delt)[pd.value_counts(delt) > 1].index
        return delete

    def input_data(self):
        self.wavelete = self.wavelete[sorted(set(self.column) - set(self.delete))]  # 篩選出要訓練的特徵
        self.column = self.wavelete.columns
        select_feature = self.wavelete.shape[1] # 重要特徵數量
        Y = np.array(pd.read_csv(self.pm25_data, index_col=0))
        return select_feature, Y

    def split_train_test(self):
        select_feature, Y = self.input_data()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        wavelete_value = scaler.fit_transform(self.wavelete)
        dataset = np.hstack((wavelete_value, Y.reshape(len(Y), 1)))
        train_size = int(len(dataset) * 0.85)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
        return train, test

    def X_Y(self):
        select_feature,Y = self.input_data()
        train, test = self.split_train_test()
        look_back = self.look_back
        trainX, trainY = self.create_dataset(train, select_feature, look_back)
        testX, testY = self.create_dataset(test, select_feature, look_back)
        return trainX, trainY, testX, testY

    def model(self):
        select_feature, Y = self.input_data()
        look_back = self.look_back
        trainX, trainY, testX, testY = self.X_Y()
        start = time.time()
        model = Sequential()
        model.add(Reshape((select_feature*look_back,), input_shape=(look_back, select_feature)))
        model.add(f.FuzzyLayer(fuzzy_size=3, input_dim=select_feature))
        model.add(Reshape((look_back, select_feature*3)))
        model.add(LSTM(48, input_shape=(self.look_back, select_feature*3),
                       return_sequences=True, kernel_constraint=max_norm(3),
                       recurrent_constraint=max_norm(3), bias_constraint=max_norm(3)))
        model.add(Dropout(0.3))
        model.add(LSTM(48, kernel_constraint=max_norm(3), recurrent_constraint=max_norm(3), bias_constraint=max_norm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(12))
        model.add(Dense(1))

        model.compile(loss='mean_absolute_error', optimizer='adam')
        trainX = trainX.reshape(trainX.shape[0], look_back, select_feature)
        model.fit(trainX, trainY, epochs=15,  verbose=2, validation_split=0.15)
        end = time.time()
        training_time = end-start
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        trainScore = math.sqrt(mean_squared_error(trainPredict, trainY))
        print('Train core: %.2f RMSE' % (trainScore))
        W, b = np.array(model.layers[1].get_weights())
        testScore = math.sqrt(mean_squared_error(testPredict, testY))
        print('Test Score: %.2f RMSE' % (testScore))
        loss = model.history.history['loss']
        val_loss = model.history.history['val_loss']
        get_fuzzy_layer_output = K.function([model.layers[0].input], [model.layers[1].output])
        layer_output = get_fuzzy_layer_output([testX])[0]
        feature = np.hstack(layer_output.reshape(len(testX)*look_back, select_feature*3, 1))
        return feature, training_time, loss, val_loss, trainScore, testScore, trainPredict, testPredict

    def feature_select(self, feature):
        feature_std = []
        for i in range(len(feature)):
            feature_std.append(feature[i].std())
        feature_std = np.array(feature_std)
        unique = np.array(self.build_fz_name(self.column))
        fe = dict(zip(unique, feature_std))
        unimport = [i for i in fe.keys() if fe[i] < self.value_setting]
        return fe, unimport

    def important(self, fe, unimport):
        self.delete = set(self.select_unimportant(self.delt_unimportant(unimport)))
        important = sorted(set(self.column) - set(self.delete))
        return self.delete, important