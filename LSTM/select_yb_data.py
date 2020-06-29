import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
import time
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

def split_time_week(data, time1):

    data["時間"] = pd.to_datetime(data["時間"])
    data["month"] = data["時間"].dt.month  # 11
    data["day"] = data["時間"].dt.day  # 12
    data["hour"] = data["時間"].dt.hour  # 13
    data['Week'] = data['時間'].dt.weekday  # 14

    con = data['hour'] == time1

    tmp = data[con].reset_index(drop=True)

    sunday = tmp['Week'] == 6
    saturday = tmp['Week'] == 5
    nonsunday = tmp['Week'] != 6
    nonsaturday = tmp['Week'] != 5
    weekand = tmp[sunday | saturday].reset_index(drop=True)  # 假日
    workday = tmp[nonsunday & nonsaturday].reset_index(drop=True)  # 平日

    return weekand, workday

# traindata = ['DonghuEveschoolday', 'DonghuMorschoolday', 'NeihuEveschoolday', 'NeihuEveworkday', 'NeihuMorschoolday',
#              'NeihuMorworkday', 'Songsanworkday', 'Zhongxiaoworkday']
# traindata_name = ['東湖', '東湖', '港墘站', '港墘站', '港墘站', '港墘站', '松山文創', '忠孝新生']
# yb_name = ['捷運東湖站', '捷運東湖站', '捷運港墘站(2號出口)', '捷運港墘站(2號出口)', '捷運港墘站(2號出口)', '捷運港墘站(2號出口)',
#            '松山高中', '金山市民路口']
# times = ['借車次數', '借車次數', '借車次數', '借車次數', '借車次數', '借車次數', '借車次數', '還車次數']
traindata = ['Songsanweekend', 'Zhongxiaoweekend']
traindata_name = ['松山文創', '忠孝新生']
yb_name = ['松山高中', '金山市民路口']
times = ['借車次數', '還車次數']
select_hour = [16, 16]
c = 0

for y in yb_name:
    data = pd.read_csv('../DATA/' + y + '.csv', encoding='utf-8')
    del data['Unnamed: 0']
    weekand, workday = split_time_week(data, select_hour[c])
    # workday = workday[['時間', '借車場站', times[c]]]
    # first_time_columns = workday.sort_values(by=['時間']).reset_index(drop=True)
    weekand = weekand[['時間', '借車場站', times[c]]]
    first_time_columns = weekand.sort_values(by=['時間']).reset_index(drop=True)
    first_time_columns.to_csv('./data/yb_data/' + traindata[c] + '.csv')
    print('---- down for this ' + traindata[c] + ' ! ----')
    c += 1