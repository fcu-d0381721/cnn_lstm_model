import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import time
import os
from math import sqrt
from datetime import datetime
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

def get_selcet_clean_data(data, threshold, c):
    data1 = pd.read_csv('./data/select_data/' + data + '_' + threshold + '_' + str(select_hour[c][0]) + '.csv', encoding='utf-8')
    data2 = pd.read_csv('./data/select_data/' + data + '_' + threshold + '_' + str(select_hour[c][1]) + '.csv', encoding='utf-8')
    data3 = pd.read_csv('./data/select_data/' + data + '_' + threshold + '_' + str(select_hour[c][2]) + '.csv', encoding='utf-8')
    data1 = data1.append(data2).reset_index(drop=True)
    data1 = data1.append(data3).reset_index(drop=True)
    del data1['Unnamed: 0']
    data1['count'] = data1['0']
    data1 = data1[['Time', 'Latitude', 'Longitude', 'count']]
    data1 = data1.sort_values(by='Time')

    data1["Time"] = pd.to_datetime(data1["Time"])
    data1["month"] = data1["Time"].dt.month  # 11
    data1["day"] = data1["Time"].dt.day  # 12
    data1["hour"] = data1["Time"].dt.hour  # 13
    data1['Week'] = data1['Time'].dt.weekday  # 14

    return data1

def get_all_clean_data(data, c):
    data1 = pd.read_csv('./data/all_data/' + data + '_' + str(select_hour[c][0]) + '.csv', encoding='utf-8')
    data2 = pd.read_csv('./data/all_data/' + data + '_' + str(select_hour[c][1]) + '.csv', encoding='utf-8')
    data3 = pd.read_csv('./data/all_data/' + data + '_' + str(select_hour[c][2]) + '.csv', encoding='utf-8')
    data1 = data1.append(data2).reset_index(drop=True)
    data1 = data1.append(data3).reset_index(drop=True)
    del data1['Unnamed: 0']
    data1 = data1.sort_values(by='Time')

    return data1

def get_other_data(data, threshold, c):
    data1 = pd.read_csv('./data/other_data/' + data + '_' + threshold + '_' + str(select_hour[c][0]) + '_other.csv',
                        encoding='utf-8')
    data2 = pd.read_csv('./data/other_data/' + data + '_' + threshold + '_' + str(select_hour[c][1]) + '_other.csv',
                        encoding='utf-8')
    data3 = pd.read_csv('./data/other_data/' + data + '_' + threshold + '_' + str(select_hour[c][2]) + '_other.csv',
                        encoding='utf-8')
    data1 = data1.append(data2).reset_index(drop=True)
    data1 = data1.append(data3).reset_index(drop=True)
    del data1['Unnamed: 0']
    data1['count'] = data1['0']
    data1 = data1[['Time', 'Latitude', 'Longitude', 'count']]
    data1 = data1.sort_values(by='Time')

    data1["Time"] = pd.to_datetime(data1["Time"])
    data1["month"] = data1["Time"].dt.month  # 11
    data1["day"] = data1["Time"].dt.day  # 12
    data1["hour"] = data1["Time"].dt.hour  # 13
    data1['Week'] = data1['Time'].dt.weekday  # 14

    return data1

def split_train_test(data):

    data[1] = data[1] / data[1].max(axis=0)
    data[1] = data[1].astype(np.float32)

    train_size = int(len(data[1]) * 0.6)
    test_size = int(len(data[1]) * 0.2)
    valid_size = int(len(data[1]) * 0.2)

    train_X, test_X, valid_X = data[0][0:train_size], data[0][train_size:train_size + test_size], \
                                        data[0][-valid_size:]
    train_Y, test_Y, valid_Y = data[1][0:train_size], data[1][train_size:train_size + test_size], \
                               data[1][-valid_size:]
    shape = train_X.shape[1]
    train_X_shape = train_X.shape[0]
    test_X_shape = test_X.shape[0]
    valid_X_shape = valid_X.shape[0]
    train_X = train_X.reshape(train_X_shape, 1, shape)
    test_X = test_X.reshape(test_X_shape, 1, shape)
    valid_X = valid_X.reshape(valid_X_shape, 1, shape)
    return train_X, test_X, valid_X, train_Y, test_Y, valid_Y

def LSTM_model(look_back, feature):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(128, input_shape=(look_back, feature), return_sequences=True))
    model.add(layers.LSTM(32, return_sequences=True))
    model.add(layers.LSTM(8))
    model.add(layers.Dense(1))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    # model.summary()
    return model

def train(model, initial, load_weights, batchsize, trainX, trainY, testX, testY, t, n):
    logdir = os.path.join('logs\scalars', datetime.now().strftime("%Y%m%d-%H%M%S"))
    architecture_filepath = os.path.join(r'architecture', 'lstm.json')
    weight_filepath = os.path.join('weight\w' + t + '_' + n, 'mse_model_{epoch:07d}.h5')
    def scheduler(epoch):
        if epoch < 800:
            return 0.001
        elif epoch < 1200:
            return 0.0001
        elif epoch < 1600:
            return 0.00001
        elif epoch < 2100:
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
            epochs=2000,
            validation_data=(testX, testY),
            callbacks=callbacks_list
        )
    else:
        model.load_weights('weight\mse_model_' + load_weights + '.h5')
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

traindata = ['DonghuEveschoolday', 'DonghuMorschoolday', 'NeihuEveschoolday', 'NeihuEveworkday', 'NeihuMorschoolday',
             'NeihuMorworkday', 'Songsanworkday', 'Zhongxiaoworkday', 'Songsanweekend', 'Zhongxiaoweekend']
times = ['借車次數', '借車次數', '借車次數', '借車次數', '借車次數', '借車次數', '借車次數', '還車次數', '借車次數', '還車次數']
select_hour = [[14, 15, 16], [4, 5, 6], [15, 16, 17], [14, 15, 16], [4, 5, 6], [6, 7, 8], [13, 14, 15], [13, 14, 15], [13, 14, 15], [13, 14, 15]]


Threshold_Folder = ['06', '065', '07', '075']
c = 0
month = [1, 2, 3, 4, 5, 6]
day = [31, 28, 31, 30, 31, 30]
n = ['select', 'other', 'original']
name = []


for t in traindata:
    for th in Threshold_Folder:
        data = get_selcet_clean_data(t, th, c)
        all_data = get_all_clean_data(t, c)
        other_data = get_other_data(t, th, c)
        yb_data = pd.read_csv('./data/yb_data/' + t + '.csv', encoding='utf-8')
        yb_data = yb_data[times[c]].to_numpy()
        data_container = [data, other_data, all_data]
        condition = 0
        for dc in data_container:
            dd = 0
            flag = True
            traindata = np.array([])

            for m in month:
                for d in range(1, day[dd] + 1):

                    con = dc['month'] == m
                    con1 = dc['day'] == d
                    if condition <= 1:
                        count = dc[con & con1].reset_index(drop=True)['count']
                        if len(dc[con & con1]) > 0:
                            if flag:
                                shape = count.to_numpy().shape[0]
                                flag = False
                            traindata = np.append(traindata, count.to_numpy())
                    else:
                        count = dc[con & con1].reset_index(drop=True)[['Age1', 'Age2', 'Age3', 'Age4', 'Age5', 'Age6', 'Total']]


                        if len(dc[con & con1]) > 0:
                            count = count.to_numpy().reshape(1, count.to_numpy().shape[0] * count.to_numpy().shape[1])
                            if flag:
                                shape = count.shape[1]
                                flag = False
                            traindata = np.append(traindata, count)
                dd += 1
            num = traindata.shape[0] / shape
            traindata = traindata.reshape(int(num), shape)
            print(traindata.shape)
            train_X, test_X, valid_X, train_Y, test_Y, valid_Y = split_train_test([traindata, yb_data])

            model = LSTM_model(1, train_X.shape[2])
            # train(model, 1, 1, 2, train_X, train_Y, test_X, test_Y, t, n[condition])
            model.load_weights('weight\w' + t + '_' + n[condition] + '\mse_model_0002000.h5')
            predicted_yb = model.predict(valid_X)
            predicted_yb_count = predicted_yb * yb_data.max(axis=0)
            valid_Y = valid_Y * yb_data.max(axis=0)
            plt.plot(valid_Y, color='red', label='Real yb')
            plt.plot(predicted_yb_count, color='blue', label='Predicted yb')
            plt.title(t + ' ' + n[condition] + ' ' + th + ' yb Prediction')
            plt.xlabel('Time')
            plt.ylabel('count')
            plt.legend()
            # plt.show()
            plt.savefig('./weight/w' + t + '_' + n[condition] + '/' + t + '_' + n[condition] + '_' + th + '.png')
            plt.close()
            mse = mean_squared_error(valid_Y, predicted_yb_count)
            mae = mean_absolute_error(valid_Y, predicted_yb_count)
            rmse = sqrt(mse)
            mape = np.mean(np.abs((valid_Y - predicted_yb_count) / valid_Y)) * 100
            name.append([t + '_' + n[condition] + '_' + th, mse, rmse, mae, mape])
            print('MSE: %f' % mse)
            print('RMSE: %f' % rmse)
            print('MAE: %f' % mae)
            print('MAPE: %f' % mape)

            condition += 1
        # time.sleep(100)
        print('---- down for this Threshold ' + th + ' ! ----')
    print('---- down for this ' + t + ' ! ----')
    c += 1
df = pd.DataFrame(name, columns=['Name', 'MSE', 'RMSE', 'MAE', 'MAPE'], dtype=float)
df.to_csv('all_model_output.csv')
# 0.85-> 92 , 0.75->105, 0.65 -> 120 0.6->125 0.7->114
