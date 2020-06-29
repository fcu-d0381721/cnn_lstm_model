import pandas as pd
import cv2
import os
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")


def split_time_week(data, time1, time2, time3):

    data["Time"] = pd.to_datetime(data["Time"])
    data["month"] = data["Time"].dt.month  # 11
    data["day"] = data["Time"].dt.day  # 12
    data["hour"] = data["Time"].dt.hour  # 13
    data['Week'] = data['Time'].dt.weekday  # 14

    con = data['hour'] == time1
    con1 = data['hour'] == time2
    con2 = data['hour'] == time3
    con3 = data['hour'] == time3 + 1
    tmp = data[con | con1 | con2 | con3].reset_index(drop=True)

    sunday = tmp['Week'] == 6
    saturday = tmp['Week'] == 5
    nonsunday = tmp['Week'] != 6
    nonsaturday = tmp['Week'] != 5
    weekand = tmp[sunday | saturday].reset_index(drop=True)  # 假日
    workday = tmp[nonsunday & nonsaturday].reset_index(drop=True)  # 平日

    return weekand, workday

def split_data(workday, time1, time2, time3):
    first_time_columns = workday[workday['hour'] == time1].reset_index(drop=True)
    second_time_columns = workday[workday['hour'] == time2].reset_index(drop=True)
    third_time_columns = workday[workday['hour'] == time3].reset_index(drop=True)

    first_time_columns = first_time_columns.sort_values(by=['Time']).reset_index(drop=True)
    second_time_columns = second_time_columns.sort_values(by=['Time']).reset_index(drop=True)
    third_time_columns = third_time_columns.sort_values(by=['Time']).reset_index(drop=True)
    return first_time_columns, second_time_columns, third_time_columns

traindata = ['DonghuEveschoolday', 'DonghuMorschoolday', 'NeihuEveschoolday', 'NeihuEveworkday', 'NeihuMorschoolday',
             'NeihuMorworkday', 'Songsanworkday', 'Zhongxiaoworkday']
traindata_name = ['東湖站', '東湖站', '港墘站', '港墘站', '港墘站', '港墘站', '松山文創', '忠孝新生']
select_hour = [[14, 15, 16], [4, 5, 6], [15, 16, 17], [14, 15, 16], [4, 5, 6], [6, 7, 8], [13, 14, 15], [13, 14, 15]]
# select_hour = [[13, 14, 15], [13, 14, 15]]
# traindata_name = ['松山文創', '忠孝新生']
# traindata = ['Songsanweekend', 'Zhongxiaoweekend']

date = ['201801', '201802', '201803', '201804', '201805', '201806']
c = 0

for t in traindata_name:

    flag = True
    temp = []
    for d in date:
        if flag:
            ori_data = pd.read_csv('../DATA/雜/' + t + '/' + d + '.csv', encoding='utf-8')
            temp = ori_data
            flag = False
        else:
            ori_data = pd.read_csv('../DATA/雜/' + t + '/' + d + '.csv', encoding='utf-8')
            temp = temp.append(ori_data).reset_index(drop=True)

    del temp['Unnamed: 0']

    weekand, workday = split_time_week(temp, select_hour[c][0], select_hour[c][1], select_hour[c][2])
    first_time_columns, second_time_columns, third_time_columns = split_data(workday, select_hour[c][0],
                                                                             select_hour[c][1], select_hour[c][2])
    # first_time_columns, second_time_columns, third_time_columns = split_data(weekand, select_hour[c][0],
    #                                                                          select_hour[c][1], select_hour[c][2])
    first_time_columns.to_csv('./data/all_data/' + traindata[c] + '_' + str(select_hour[c][0]) + '.csv')
    second_time_columns.to_csv('./data/all_data/' + traindata[c] + '_' + str(select_hour[c][1]) + '.csv')
    third_time_columns.to_csv('./data/all_data/' + traindata[c] + '_' + str(select_hour[c][2]) + '.csv')
    print('---- down for this ' + traindata[c] + ' ! ----')
    c += 1