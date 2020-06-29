from sklearn.svm import SVR
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

warnings.filterwarnings("ignore")

def get_selcet_clean_data(data, threshold, c):
    data1 = pd.read_csv('./data/' + data + '_' + threshold + '_' + str(select_hour[c][0]) + '.csv', encoding='utf-8')
    data2 = pd.read_csv('./data/' + data + '_' + threshold + '_' + str(select_hour[c][1]) + '.csv', encoding='utf-8')
    data3 = pd.read_csv('./data/' + data + '_' + threshold + '_' + str(select_hour[c][2]) + '.csv', encoding='utf-8')
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

def set_timestamp_data(data, c):
    data1 = pd.read_csv('./data/all_data/' + data + '_' + str(select_hour[c][0]) + '.csv', encoding='utf-8')
    data2 = pd.read_csv('./data/all_data/' + data + '_' + str(select_hour[c][1]) + '.csv', encoding='utf-8')
    data3 = pd.read_csv('./data/all_data/' + data + '_' + str(select_hour[c][2]) + '.csv', encoding='utf-8')
    data1 = data1.append(data2).reset_index(drop=True)
    data1 = data1.append(data3).reset_index(drop=True)
    del data1['Unnamed: 0']
    data1 = data1.sort_values(by='Time')

    return data1

def split_train_test(data):

    data[1] = data[1] / data[1].max(axis=0)
    data[1] = data[1].astype(np.float32)

    train_size = int(len(data[1]) * 0.7)
    test_size = int(len(data[1]) * 0.3)


    train_X, test_X = data[0][0:train_size], data[0][train_size:train_size + test_size]
    train_Y, test_Y = data[1][0:train_size], data[1][train_size:train_size + test_size]
    shape = train_X.shape[1]
    train_X_shape = train_X.shape[0]
    test_X_shape = test_X.shape[0]

    # train_X = train_X.reshape(train_X_shape, 1, shape)
    # test_X = test_X.reshape(test_X_shape, 1, shape)

    return train_X, test_X, train_Y, test_Y


# traindata = ['DonghuEveschoolday', 'DonghuMorschoolday', 'NeihuEveschoolday', 'NeihuEveworkday', 'NeihuMorschoolday',
#              'NeihuMorworkday', 'Songsanworkday', 'Zhongxiaoworkday']
# times = ['借車次數', '借車次數', '借車次數', '借車次數', '借車次數', '借車次數', '借車次數', '還車次數']
# select_hour = [[14, 15, 16], [4, 5, 6], [15, 16, 17], [14, 15, 16], [4, 5, 6], [6, 7, 8], [13, 14, 15], [13, 14, 15]]
traindata = ['DonghuEveschoolday', 'DonghuMorschoolday', 'NeihuEveschoolday', 'NeihuEveworkday', 'NeihuMorschoolday',
             'NeihuMorworkday', 'Songsanworkday', 'Zhongxiaoworkday', 'Songsanweekend', 'Zhongxiaoweekend']
times = ['借車次數', '借車次數', '借車次數', '借車次數', '借車次數', '借車次數', '借車次數', '還車次數', '借車次數', '還車次數']
select_hour = [[14, 15, 16], [4, 5, 6], [15, 16, 17], [14, 15, 16], [4, 5, 6], [6, 7, 8], [13, 14, 15], [13, 14, 15], [13, 14, 15], [13, 14, 15]]


Threshold_Folder = ['06', '065', '07', '075', '08', '085']
c = 0
month = [1, 2, 3, 4, 5, 6]
day = [31, 28, 31, 30, 31, 30]
n = ['select', 'original']
name = []


for t in traindata:
    for th in Threshold_Folder:
        data = get_selcet_clean_data(t, th, c)
        all_data = get_all_clean_data(t, c)
        yb_data = pd.read_csv('./data/yb_data/' + t + '.csv', encoding='utf-8')
        yb_data = yb_data[times[c]].to_numpy()
        data_container = [data, all_data]
        condition = 0
        for dc in data_container:
            dd = 0
            flag = True
            traindata = np.array([])

            for m in month:
                for d in range(1, day[dd] + 1):

                    con = dc['month'] == m
                    con1 = dc['day'] == d
                    if condition == 0:
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
            train_X, test_X, train_Y, test_Y = split_train_test([traindata, yb_data])
            SVR_Model = SVR()
            rgr = SVR_Model.fit(train_X, train_Y)
            X_test_predict = rgr.predict(train_X)
            predicted_yb_count = X_test_predict * yb_data.max(axis=0)
            train_Y = train_Y * yb_data.max(axis=0)
            plt.plot(train_Y, color='red', label='Real yb')
            plt.plot(predicted_yb_count, color='blue', label='Predicted yb')
            plt.title(t + ' ' + n[condition] + ' ' + th + ' train' + ' yb Prediction')
            plt.xlabel('Time')
            plt.ylabel('count')
            plt.legend()
            # plt.show()
            plt.savefig('./svrimage/' + t + '_' + n[condition] + '_' + th + '_train' + '.png')
            plt.close()
            mse = mean_squared_error(train_Y, predicted_yb_count)
            mae = mean_absolute_error(train_Y, predicted_yb_count)
            rmse = sqrt(mse)
            mape = np.mean(np.abs((train_Y - predicted_yb_count) / train_Y)) * 100
            name.append([t + '_' + n[condition] + '_' + th + '_train', mse, rmse, mae, mape])
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
df.to_csv('all_svr_output_train.csv')
# 0.85-> 92 , 0.75->105, 0.65 -> 120 0.6->125 0.7->114


