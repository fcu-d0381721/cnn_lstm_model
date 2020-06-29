import pandas as pd
import numpy as np
import cv2
import time
import os
import warnings
warnings.filterwarnings("ignore")

"""
將抓取完的9宮格
輸出成圖片
依照時間'星期

"""

x_step = 3
y_step = 3
col = ['Age1', 'Age2', 'Age3', 'Age4', 'Age5', 'Age6', 'Total']
# point = ['北車站', '大直站', '市府站', '港墘站', '田徑場']
# point_en = ['taipei', 'dachi', 'cityhall', 'gangchen', 'run']
# site = ['市民太原路口', '捷運大直站(3號出口)', '捷運市政府站(3號出口)', '捷運港墘站(2號出口)',  '臺北田徑場']
point = ['板橋']
site = ['板橋']
dataname = [['banqiao_work_in', 'banqiao_week_in']]
# dataname = [['zhongciao_work_out', 'zhongciao_week_out'], ['cityhall_work_out', 'cityhall_week_out'], ['ganchain_work_out', 'ganchain_week_out'], ['songgan_work_out', 'songgan_week_out'], ['taipei_work_out', 'taipei_week_out'], ['banqiao_work_out', 'banqiao_week_out'], ['techbuil_work_out', 'techbuil_week_out'], ['xiban_work_out', 'xiban_week_out']]

day = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
Y = [0, 1, 2]
X = [2, 1, 0]
# pe = 0
e = 0

def split_week_work(rental):
    sunday = rental['Week'] == 6
    saturday = rental['Week'] == 5
    nonsunday = rental['Week'] != 6
    nonsaturday = rental['Week'] != 5
    yweekand = rental[sunday | saturday].reset_index(drop=True)  # 假日
    yworkday = rental[nonsunday & nonsaturday].reset_index(drop=True)  # 平日
    return yweekand, yworkday

def get_assign_time(data, name, yweekand, yworkday, a, b, c, d):
    TwoPM1 = yweekand[name] == a
    ThreePM1 = yweekand[name] == b
    FourPM1 = yweekand[name] == c
    FivePM1 = yweekand[name] == d

    TwoPM = yworkday[name] == a
    ThreePM = yworkday[name] == b
    FourPM = yworkday[name] == c
    FivePM = yworkday[name] == d
    yEveweekand = yweekand[TwoPM1 | ThreePM1 | FourPM1 | FivePM1].reset_index(drop=True)  # 假日中的14,15,16,17點
    yEveworkday = yworkday[TwoPM | ThreePM | FourPM | FivePM].reset_index(drop=True)  # 平日中的14,15,16,17點
    return yEveweekand, yEveworkday

for p in point:

    print(p)

    flag = True

    # rental = pd.read_csv('./DATA/捷運/2018' + site[pe] + '進站.csv', encoding='utf-8')  #yb
    rental = pd.read_csv('./DATA/捷運/2018' + p + '進站.csv', encoding='utf-8')

    rental["日期"] = pd.to_datetime(rental["日期"])
    # rental["hour"] = rental["時間"].dt.hour  # 13
    rental['Week'] = rental['日期'].dt.weekday

    yweekand, yworkday = split_week_work(rental)
    yEveweekand, yEveworkday = get_assign_time(rental, '時段', yweekand, yworkday, 6, 7, 8, 9)
    # print(yEveweekand, yEveworkday)

    yEveweekand = yEveweekand[['人次']]
    yEveworkday = yEveworkday[['人次']]
    ybd = [yEveworkday, yEveweekand]
    # renp = rental.to_numpy()

    for m in range(1, 13):
        if flag:
            people = pd.read_csv('./DATA/' + p + '/20180' + str(m) + '.csv', encoding='utf-8')
            flag = False
        else:
            if m < 10:
                people = people.append(pd.read_csv('./DATA/' + p + '/20180' + str(m) + '.csv', encoding='utf-8')).reset_index(drop=True)
            else:
                people = people.append(
                    pd.read_csv('./DATA/' + p + '/2018' + str(m) + '.csv', encoding='utf-8')).reset_index(drop=True)
    del people['Unnamed: 0']

    people["Time"] = pd.to_datetime(people["Time"])
    people["month"] = people["Time"].dt.month  # 11
    people["day"] = people["Time"].dt.day  # 12
    people["hour"] = people["Time"].dt.hour  # 13
    people['Week'] = people['Time'].dt.weekday  # 14
    print(people)
    #
    time.sleep(100)
    weekand, workday = split_week_work(people)
    Eveweekand, Eveworkday = get_assign_time(people, 'hour', weekand, workday, 6, 7, 8, 9)
    data = [Eveworkday, Eveweekand]

    yb = 0
    for i in data:
        s = 0
        ybdata = ybd[yb].to_numpy()
        # print(ybdata)
        date = i['day'].unique()
        hour = i['hour'].unique()
        print(date, hour)
        time.sleep(1)
        print(i)
        i = i.to_numpy()

        for c in range(4, 11):
            print("**********" + col[s])
            _range = np.max(abs(i[:, c]))
            i[:, c] = i[:, c] / _range
            evd = 0
            for m in range(13):
                for d in date:
                    for h in hour:
                        count = 0
                        print(i)
                        temp = i[np.where((i[:, 11] == m + 1) & (i[:, 12] == d) & (i[:, 13] == h)), c]
                        lng_lat = i[np.where((i[:, 11] == m + 1) & (i[:, 12] == d) & (i[:, 13] == h)), 2: 4]
                        print(temp)
                        if temp.size != 0:

                            img = np.zeros([x_step, y_step, 1])

                            for y in Y:
                                for x in X:
                                    img[x, y, 0] = temp[0, count]
                                    count += 1
                            print(str(dataname[e][yb]))
                            print(str(col[s]))
                            print(str(m + 1), str(d), str(h))
                            print(str(ybdata[evd, 0]))
                            print('-----------------------')
                            time.sleep(1)
                            cv2.imwrite('./CNN/MRT/train/' + str(dataname[e][yb]) + '/' + str(col[s]) + '/' + str(m + 1) + '-' + str(d) + '-' + str(h) + '-' + str(ybdata[evd, 0]) + '.png', img * 255)
                            evd += 1
            s += 1
        yb += 1
    # pe += 1
    e += 1