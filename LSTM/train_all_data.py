import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import time
import os
from math import sqrt
from datetime import datetime
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

def split_time_week(data):

    data["Time"] = pd.to_datetime(data["Time"])
    data["month"] = data["Time"].dt.month  # 11
    data["day"] = data["Time"].dt.day  # 12
    data["hour"] = data["Time"].dt.hour  # 13
    data['Week'] = data['Time'].dt.weekday  # 14

    con = data['hour'] == 13
    con1 = data['hour'] == 14
    con2 = data['hour'] == 15
    con3 = data['hour'] == 16
    tmp = data[con | con1 | con2 | con3].reset_index(drop=True)

    # sunday = tmp['Week'] == 6
    # saturday = tmp['Week'] == 5
    # nonsunday = tmp['Week'] != 6
    # nonsaturday = tmp['Week'] != 5
    # weekand = tmp[sunday | saturday].reset_index(drop=True)  # 假日
    # workday = tmp[nonsunday & nonsaturday].reset_index(drop=True)  # 平日

    return tmp

def get_clean_data():
    yb = pd.read_csv('../DATA/金山市民路口.csv', encoding='utf-8')
    del yb['Unnamed: 0']
    yb['Time'] = yb['時間']
    yb = yb[['Time', '借車次數', '還車次數']]

    thdf = pd.read_csv('athdf.csv', encoding='utf-8')
    fodf = pd.read_csv('afodf.csv', encoding='utf-8')
    fidf = pd.read_csv('afidf.csv', encoding='utf-8')
    del thdf['Unnamed: 0']
    del fodf['Unnamed: 0']
    del fidf['Unnamed: 0']


    thdf = thdf[['Time','Longitude','Latitude','Age1','Age2','Age3','Age4','Age5','Age6','Total']]
    fodf = fodf[['Time','Longitude','Latitude','Age1','Age2','Age3','Age4','Age5','Age6','Total']]
    fidf = fidf[['Time','Longitude','Latitude','Age1','Age2','Age3','Age4','Age5','Age6','Total']]
    thdf = pd.merge(thdf, yb, on='Time')
    fodf = pd.merge(fodf, yb, on='Time')
    fidf = pd.merge(fidf, yb, on='Time')

    total = thdf.append(fodf).reset_index(drop=True)
    total = total.append(fidf).reset_index(drop=True)
    total = split_time_week(total)
    yb = split_time_week(yb)

    month = total['month'].unique()
    day = total['day'].unique()

    flag = True
    t = []
    for m in month:
        for d in day:
            con = yb['month'] == m
            con1 = yb['day'] == d
            con2 = total['month'] == m
            con3 = total['day'] == d
            if len(total[con2 & con3]) > 0:
                if flag:
                    t = yb[con & con1].reset_index(drop=True)
                    flag = False
                else:
                    te = yb[con & con1].reset_index(drop=True)
                    t = t.append(te).reset_index(drop=True)
    t = t.sort_values(by='Time')

    return total, t

def split_train_test(data):

    data[1] = data[1] / data[1].max(axis=0)
    data[1] = data[1].astype(np.float32)

    train_size = 30
    train_P, test_P = data[0][0:(len(data[1]) - train_size), :], data[0][-train_size::]
    train_YB, test_YB = data[1][0:(len(data[1]) - train_size)], data[1][-train_size::]
    return train_P, test_P, train_YB, test_YB

def LSTM_model(look_back):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(128, input_shape=(look_back, 189), return_sequences=True))
    model.add(layers.LSTM(32, return_sequences=True))
    model.add(layers.LSTM(8))
    model.add(layers.Dense(1))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    model.summary()
    return model

def train(model, initial, load_weights, batchsize, trainX, trainY, testX, testY):
    logdir = os.path.join('logs\scalars', datetime.now().strftime("%Y%m%d-%H%M%S"))
    architecture_filepath = os.path.join(r'architecture', 'lstm.json')
    weight_filepath = os.path.join('weight', 'mse_model_{epoch:07d}.h5')
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

month = [1, 2, 3, 4, 5, 6]
day = [31, 28, 31, 30, 31, 30]
hour = [13, 14, 15, 16]
tm = [0, 1, 2, 3, 4, False]
total, yb = get_clean_data()

totala = total.to_numpy()
tmpp = totala[:, 3:10]
norm = np.linalg.norm(tmpp)
tmpp = tmpp / norm
df = pd.DataFrame(data=tmpp, columns=['Age1', 'Age2', 'Age3', 'Age4', 'Age5', 'Age6', 'Total'])
total['Age1'] = df['Age1']
total['Age2'] = df['Age2']
total['Age3'] = df['Age3']
total['Age4'] = df['Age4']
total['Age5'] = df['Age5']
total['Age6'] = df['Age6']
total['Total'] = df['Total']

a = np.array([])
f = np.array([])
y = []
for m in month:
    for d in range(1, day[m-1]+1):
        count = 0
        for h in hour:
            con = total['month'] == m
            con1 = total['day'] == d
            con2 = total['hour'] == h
            e = total[con & con1 & con2]

            if h == 16:
                ybcon = yb['month'] == m
                ybcon1 = yb['day'] == d
                ybcon2 = yb['hour'] == h
                ybe = yb[ybcon & ybcon1 & ybcon2]
                if len(ybe) > 0:
                    tm[4] = ybe['借車次數'].to_numpy()[0]
                    y.append(tm[4])

            if len(e) > 0:
                a = np.append(a, total[con & con1 & con2][['Age1']].to_numpy()[:, 0])
                a = np.append(a, total[con & con1 & con2][['Age2']].to_numpy()[:, 0])
                a = np.append(a, total[con & con1 & con2][['Age3']].to_numpy()[:, 0])
                a = np.append(a, total[con & con1 & con2][['Age4']].to_numpy()[:, 0])
                a = np.append(a, total[con & con1 & con2][['Age5']].to_numpy()[:, 0])
                a = np.append(a, total[con & con1 & con2][['Age6']].to_numpy()[:, 0])
                a = np.append(a, total[con & con1 & con2][['Total']].to_numpy()[:, 0])
                tm[5] = True
        if tm[5]:
            r = np.array([tm[0], tm[1], tm[2]])
            tm[5] = False

a = a.reshape(130, 189)

y = np.array(y)
# print(a.shape)
train_P, test_P, train_YB, test_YB = split_train_test([a, y])


train_P = train_P.reshape(100, 1, 189)
# print(train_P.shape)
test_P = test_P.reshape(30, 1, 189)
train_P = np.asarray(train_P).astype(np.float32)
test_P = np.asarray(test_P).astype(np.float32)
train_YB = np.asarray(train_YB).astype(np.float32)
test_YB = np.asarray(test_YB).astype(np.float32)
model = LSTM_model(1)
train(model, 1, 1, 5, train_P, train_YB, test_P, test_YB)

predicted_yb = model.predict(test_P)
predicted_yb_count = predicted_yb * y.max(axis=0)
test_YB = test_YB * y.max(axis=0)
plt.plot(test_YB, color='red', label='Real yb')
plt.plot(predicted_yb_count, color='blue', label='Predicted yb')
plt.title('yb Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

mse = mean_squared_error(test_YB, predicted_yb_count)
rmse = sqrt(mse)
mape = np.mean(np.abs((test_YB - predicted_yb_count) / test_YB)) * 100

print('MSE: %f' % mse)
print('RMSE: %f' % rmse)
print('MAPE: %f' % mape)