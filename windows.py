import pandas as pd
import numpy as np
import cv2
import math
from sklearn.preprocessing import MinMaxScaler
import time

x_step = 500
y_step = 1000
col = ['Age1', 'Age2', 'Age3', 'Age4', 'Age5', 'Age6', 'Total']
flag = True
for m in range(1, 7):
    if flag:
        people = pd.read_csv('./DATA/20180' + str(m) + '.csv', encoding='utf-8')
        flag = False
    else:
        people = people.append(pd.read_csv('./DATA/20180' + str(m) + '.csv', encoding='utf-8')).reset_index(drop=True)


for c in col:
    print('down')
    print(c)
    y = people.groupby(['Time', 'Latitude', 'Longitude'])[c].sum()
    y = y.unstack(level=0)
    y = y.fillna(0)
    print(y)

    df = pd.read_csv('人流站點.csv', encoding='utf-8')
    del df['field_1']
    df = df.sort_values(by=['lat', 'lon']).reset_index(drop=True)
    # print(y.iloc[:, :])
    scaler = MinMaxScaler()
    print(scaler.fit(y.iloc[:, :]))
    X = scaler.transform(y.iloc[:, :])
    # print(X)

    img = np.zeros([x_step + 1, y_step + 1, 1])

    x = list()
    y = list()
    max_x = 330000 - 280000
    max_y = 2800000 - 2700000

    windows = max_x / x_step

    for value in df.X:
        value = value - 280000
        x.append(float(value) / ((float(max_x/x_step))))

    for value in df.Y:
        value = value - 2700000 - 25000
        y.append(float(value) / ((float(max_y/y_step))))

    month = ['1', '2', '3', '4', '5', '6']
    day = [31, 28, 31, 30, 31, 30]
    # month = ['4']
    # day = [30]

    count = 0
    for m in range(len(month)):
        for d in range(1, day[m]+1):
            for h in range(24):
                for index in range(len(x)):
                    try:
                        img_x = round(x[index])
                        img_y = round(y[index])

                        img[img_x][img_y][0] = 1
                        img[img_x-2:img_x+2, img_y-2:img_y+2, 0] = X[index, count]
                    except:
                        count -= 1
                        img[img_x - 2:img_x + 2, img_y - 2:img_y + 2, 0] = X[index, count]
                count += 1
                cv2.imwrite('./CNN/train/' + c + '/' + month[m] + '-' + str(d) + '-' + str(h) + '.png', img*255)