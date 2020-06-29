import pandas as pd
import cv2
import os
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        if label:
          y[i] = int(file.split(".")[0])
    if label:
      return y

def split_time_week(data, time1, time2, time3):

    data["Time"] = pd.to_datetime(data["Time"])
    data["month"] = data["Time"].dt.month  # 11
    data["day"] = data["Time"].dt.day  # 12
    data["hour"] = data["Time"].dt.hour  # 13
    data['Week'] = data['Time'].dt.weekday  # 14

    con = data['hour'] == time1
    con1 = data['hour'] == time2
    con2 = data['hour'] == time3
    # con3 = data['hour'] == time3 + 1
    # tmp = data[con | con1 | con2 | con3].reset_index(drop=True)
    tmp = data[con | con1 | con2].reset_index(drop=True)

    sunday = tmp['Week'] == 6
    saturday = tmp['Week'] == 5
    nonsunday = tmp['Week'] != 6
    nonsaturday = tmp['Week'] != 5
    weekand = tmp[sunday | saturday].reset_index(drop=True)  # 假日
    workday = tmp[nonsunday & nonsaturday].reset_index(drop=True)  # 平日

    return weekand, workday

def select_feature(filename, data, c):
    first_time_columns = pd.Series()
    second_time_columns = pd.Series()
    third_time_columns = pd.Series()
    first_time_data = pd.Series()
    second_time_data = pd.Series()
    third_time_data = pd.Series()
    for i in filename:
        hr_plus = int(i / 63)  # 商數來判斷哪小時/63
        total = select_hour[c][0] + hr_plus  # 商數來判斷哪小時/63  有加數字代表往下幾個小時
        h = data['hour'] == total
        lngg = data['Longitude'] == lng_lat[c][i % 9][0]  # 餘數來判斷是哪格%9
        latt = data['Latitude'] == lng_lat[c][i % 9][1]  # 餘數來判斷是哪格%9
        if total == select_hour[c][0]:
            if len(data[h & lngg & latt]) > 0:
                colum = data[h & lngg & latt].reset_index(drop=True).iloc[:, 0:4]  # 前面的東西 like time lng lat
                needata = data[h & lngg & latt].reset_index(drop=True).iloc[:, int(i / 9) + 4]  # 選定的因子 商數來判斷是哪個因子/9
                first_time_columns = first_time_columns.append(colum, ignore_index=True)
                first_time_data = first_time_data.append(needata, ignore_index=True)

        elif total == select_hour[c][1]:
            if len(data[h & lngg & latt]) > 0:
                colum = data[h & lngg & latt].reset_index(drop=True).iloc[:, 0:4]
                needata = data[h & lngg & latt].reset_index(drop=True).iloc[:, int(i / 9) - 3]
                second_time_columns = second_time_columns.append(colum, ignore_index=True)
                second_time_data = second_time_data.append(needata, ignore_index=True)

        elif total == select_hour[c][2]:
            if len(data[h & lngg & latt]) > 0:
                colum = data[h & lngg & latt].reset_index(drop=True).iloc[:, 0:4]
                needata = data[h & lngg & latt].reset_index(drop=True).iloc[:, int(i / 9) - 9]
                third_time_columns = third_time_columns.append(colum, ignore_index=True)
                third_time_data = third_time_data.append(needata, ignore_index=True)
    first_time_columns = first_time_columns.append(second_time_columns, ignore_index=True)
    first_time_columns = first_time_columns.append(third_time_columns, ignore_index=True)
    first_time_data = first_time_data.append(second_time_data, ignore_index=True)
    first_time_data = first_time_data.append(third_time_data, ignore_index=True)
    first_time_columns[0] = first_time_data
    first_time_columns = first_time_columns.sort_values(by=['Time']).reset_index(drop=True)

    return first_time_columns

workspace_dir = '../CNN/time'
# traindata = ['DonghuEveschoolday', 'DonghuMorschoolday', 'NeihuEveschoolday', 'NeihuEveworkday', 'NeihuMorschoolday',
#              'NeihuMorworkday', 'Songsanworkday', 'Zhongxiaoworkday']
# traindata_name = ['東湖站', '東湖站', '港墘站', '港墘站', '港墘站', '港墘站', '松山文創', '忠孝新生']
# lng_lat = [[[121.615, 25.07], [121.62, 25.065], [121.61, 25.075], [121.615, 25.065], [121.61, 25.065], [121.61, 25.07],
#            [121.62, 25.07], [121.615, 25.075], [121.62, 25.075]], [[121.615, 25.07], [121.62, 25.065], [121.61, 25.075], [121.615, 25.065], [121.61, 25.065], [121.61, 25.07],
#             [121.62, 25.07], [121.615, 25.075], [121.62, 25.075]], [[121.575, 25.08], [121.58, 25.075], [121.57, 25.085], [121.575, 25.075], [121.57, 25.075], [121.57, 25.08],
#             [121.58, 25.08], [121.575, 25.085], [121.58, 25.085]], [[121.575, 25.08], [121.58, 25.075], [121.57, 25.085], [121.575, 25.075], [121.57, 25.075], [121.57, 25.08],
#             [121.58, 25.08], [121.575, 25.085], [121.58, 25.085]], [[121.575, 25.08], [121.58, 25.075], [121.57, 25.085], [121.575, 25.075], [121.57, 25.075], [121.57, 25.08],
#            [121.58, 25.08], [121.575, 25.085], [121.58, 25.085]], [[121.575, 25.08], [121.58, 25.075], [121.57, 25.085], [121.575, 25.075], [121.57, 25.075], [121.57, 25.08],
#            [121.58, 25.08], [121.575, 25.085], [121.58, 25.085]], [[121.565, 25.045], [121.57, 25.04], [121.56, 25.05], [121.565, 25.04], [121.56, 25.04], [121.56, 25.045],
#            [121.57, 25.045], [121.565, 25.05], [121.57, 25.05]], [[121.53, 25.045], [121.535, 25.04], [121.525, 25.05], [121.53, 25.04], [121.525, 25.04], [121.525, 25.045],
#            [121.535, 25.045], [121.53, 25.05], [121.535, 25.05]]]
# select_hour = [[14, 15, 16], [4, 5, 6], [15, 16, 17], [14, 15, 16], [4, 5, 6], [6, 7, 8], [13, 14, 15], [13, 14, 15]]
traindata = ['Songsanweekend', 'Zhongxiaoweekend']
traindata_name = ['松山文創', '忠孝新生']
lng_lat = [[[121.565, 25.045], [121.57, 25.04], [121.56, 25.05], [121.565, 25.04], [121.56, 25.04], [121.56, 25.045],
           [121.57, 25.045], [121.565, 25.05], [121.57, 25.05]], [[121.53, 25.045], [121.535, 25.04], [121.525, 25.05], [121.53, 25.04], [121.525, 25.04], [121.525, 25.045],
           [121.535, 25.045], [121.53, 25.05], [121.535, 25.05]]]
select_hour = [[13, 14, 15], [13, 14, 15]]
Threshold_Folder = ['06', '065', '07', '075', '08']
date = ['201801', '201802', '201803', '201804', '201805', '201806']
c = 0

for t in traindata:
    for th in Threshold_Folder:
        y_train_Age1 = readfile(os.path.join(workspace_dir, t + "/_excess_" + th), True)
        # y_train_Age1 = sorted(y_train_Age1)
        a = np.arange(0, 190).tolist()
        y_train_Age1 = sorted(list(set(a) - set(y_train_Age1)))

        flag = True
        temp = []
        for d in date:
            if flag:
                ori_data = pd.read_csv('../DATA/雜/' + traindata_name[c] + '/' + d + '.csv', encoding='utf-8')
                temp = ori_data
                flag = False
            else:
                ori_data = pd.read_csv('../DATA/雜/' + traindata_name[c] + '/' + d + '.csv', encoding='utf-8')
                temp = temp.append(ori_data).reset_index(drop=True)

        del temp['Unnamed: 0']

        weekand, workday = split_time_week(temp, select_hour[c][0], select_hour[c][1], select_hour[c][2])

        first_time_columns = select_feature(y_train_Age1, weekand, c)

        first_time_columns.to_csv('./newdata/other_data/' + t + '_' + th + '_' + str(select_hour[c][0]) + '_other.csv')

        print('---- down for this Threshold ' + th + ' ! ----')
    print('---- down for this ' + t + ' ! ----')
    c += 1