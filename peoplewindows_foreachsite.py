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
依照時間'地點

"""

x_step = 3
y_step = 3
col = ['Age1', 'Age2', 'Age3', 'Age4', 'Age5', 'Age6', 'Total']
# point = ['北車站', '大直站', '市府站', '港墘站', '田徑場']
# point_en = ['taipei', 'dachi', 'cityhall', 'gangchen', 'run']
# site = ['市民太原路口', '捷運大直站(3號出口)', '捷運市政府站(3號出口)', '捷運港墘站(2號出口)',  '臺北田徑場']
point = ['松山文創']
point_en = ['donghu']
site = ['松山高中']


# , '大直站', '市府站', '港墘站', '田徑場']
day = [31, 28, 31, 30, 31, 30]
Y = [0, 1, 2]
X = [2, 1, 0]
pe = 0
#
for p in point:

    print(point_en[pe])
    flag = True

    rental = pd.read_csv('./DATA/' + site[pe] + '.csv', encoding='utf-8')
    rental["時間"] = pd.to_datetime(rental["時間"])
    rental["hour"] = rental["時間"].dt.hour  # 13
    rental['Week'] = rental['時間'].dt.weekday

    nonsunday = rental['Week'] != 6
    nonsaturday = rental['Week'] != 5
    yworkday = rental[nonsunday & nonsaturday].reset_index(drop=True)  # 平日
    print(len(yworkday))
    time.sleep(1000)
    FourAM = rental['hour'] == 4
    FiveAM = rental['hour'] == 5
    SixAM = rental['hour'] == 6
    SevenAM = rental['hour'] == 7
    # EightAM = rental['hour'] == 8
    # NineAM = rental['hour'] == 9
    TwoPM = rental['hour'] == 14
    ThreePM = rental['hour'] == 15
    FourPM = rental['hour'] == 16
    FivePM = rental['hour'] == 17
    # SixPM = rental['hour'] == 18

    # yNeihuMorworkday = yworkday[SixAM | SevenAM | EightAM | NineAM].reset_index(drop=True)  # 平日中的6,7,8,9點
    # yNeihuEveworkday = yworkday[ThreePM | FourPM | FivePM | SixPM].reset_index(drop=True)  # 平日中的15,16,17,18點
    yNeihuMorschoolday = yworkday[FourAM | FiveAM | SixAM | SevenAM].reset_index(drop=True)  # 平日中的4,5,6,7點
    yNeihuEveschoolday = yworkday[TwoPM | ThreePM | FourPM | FivePM].reset_index(drop=True)  # 平日中的14,15,16,17點

    # yNeihuMorworkday = yNeihuMorworkday[['借車次數']]
    # yNeihuEveworkday = yNeihuEveworkday[['借車次數']]
    yNeihuMorschoolday = yNeihuMorschoolday[['借車次數']]
    yNeihuEveschoolday = yNeihuEveschoolday[['借車次數']]
    ybd = [yNeihuMorschoolday, yNeihuEveschoolday]
    # renp = rental.to_numpy()

    for m in range(1, 7):
        if flag:
            people = pd.read_csv('./DATA/' + p + '/20180' + str(m) + '.csv', encoding='utf-8')
            flag = False
        else:
            people = people.append(
                pd.read_csv('./DATA/' + p + '/20180' + str(m) + '.csv', encoding='utf-8')).reset_index(drop=True)

    del people['Unnamed: 0']

    people["Time"] = pd.to_datetime(people["Time"])
    people["month"] = people["Time"].dt.month  # 11
    people["day"] = people["Time"].dt.day  # 12
    people["hour"] = people["Time"].dt.hour  # 13
    people['Week'] = people['Time'].dt.weekday  # 14


    nonsunday = people['Week'] != 6
    nonsaturday = people['Week'] != 5
    workday = people[nonsunday & nonsaturday].reset_index(drop=True)  # 平日

    FourAM = people['hour'] == 4
    FiveAM = people['hour'] == 5
    SixAM = people['hour'] == 6
    SevenAM = people['hour'] == 7
    # EightAM = people['hour'] == 8
    # NineAM = people['hour'] == 9
    TwoPM = people['hour'] == 14
    ThreePM = people['hour'] == 15
    FourPM = people['hour'] == 16
    FivePM = people['hour'] == 17
    # SixPM = people['hour'] == 18
    # NeihuMorworkday = workday[SixAM | SevenAM | EightAM | NineAM].reset_index(drop=True)  # 平日中的6,7,8,9點
    # NeihuEveworkday = workday[ThreePM | FourPM | FivePM | SixPM].reset_index(drop=True)  # 平日中的15,16,17,18點
    NeihuMorschoolday = workday[FourAM | FiveAM | SixAM | SevenAM].reset_index(drop=True)  # 平日中的4,5,6,7點
    NeihuEveschoolday = workday[TwoPM | ThreePM | FourPM | FivePM].reset_index(drop=True)  # 平日中的14,15,16,17點

    data = [NeihuMorschoolday, NeihuEveschoolday]
    dataname = ['DonghuMorschoolday', 'DonghuEveschoolday']
    Morhour = [4, 5, 6, 7]
    Evehour = [14, 15, 16, 17]
    yb = 0
    for i in data:
        s = 0
        ybdata = ybd[yb].to_numpy()
        # print(ybdata)
        date = i['day'].unique()
        hour = i['hour'].unique()
        i = i.to_numpy()

        for c in range(4, 11):
            print("**********" + col[s])
            _range = np.max(abs(i[:, c]))
            i[:, c] = i[:, c] / _range
            evd = 0
            for m in range(6):
                for d in date:
                    for h in hour:
                        count = 0
                        temp = i[np.where((i[:, 11] == m + 1) & (i[:, 12] == d) & (i[:, 13] == h)), c]
                        lng_lat = i[np.where((i[:, 11] == m + 1) & (i[:, 12] == d) & (i[:, 13] == h)), 2: 4]
                        if temp.size != 0:

                            img = np.zeros([x_step, y_step, 1])

                            for y in Y:
                                for x in X:
                                    img[x, y, 0] = temp[0, count]
                                    count += 1

                            cv2.imwrite('./CNN/time/train/' + str(dataname[yb]) + '/' + str(col[s]) + '/' + str(m + 1) + '-' + str(d) + '-' + str(h) + '-' + str(ybdata[evd, 0]) + '.png', img * 255)
                            evd += 1

            s += 1
        yb += 1
    pe += 1