import pandas as pd
import numpy as np
import cv2
import time
import os
"""
將抓取完的9宮格
輸出成圖片

"""

x_step = 3
y_step = 3
col = ['Age1', 'Age2', 'Age3', 'Age4', 'Age5', 'Age6', 'Total']
# point = ['北車站', '大直站', '市府站', '港墘站', '田徑場']
# point_en = ['taipei', 'dachi', 'cityhall', 'gangchen', 'run']
# site = ['市民太原路口', '捷運大直站(3號出口)', '捷運市政府站(3號出口)', '捷運港墘站(2號出口)',  '臺北田徑場']
point = ['港墘站']
point_en = ['gangchen']
site = ['捷運港墘站(2號出口)']

    # , '大直站', '市府站', '港墘站', '田徑場']
day = [31, 28, 31, 30, 31, 30]
Y = [0, 1, 2]
X = [2, 1, 0]
pe = 0


#
for p in point:
    s = 0
    print(point_en[pe])
    flag = True

    rental = pd.read_csv('./DATA/' + site[pe] + '.csv', encoding='utf-8')
    rental = rental[['借車次數']]
    renp = rental.to_numpy()

    for m in range(1, 7):
        if flag:
            people = pd.read_csv('./DATA/' + p + '/20180' + str(m) + '.csv', encoding='utf-8')
            flag = False
        else:
            people = people.append(pd.read_csv('./DATA/' + p + '/20180' + str(m) + '.csv', encoding='utf-8')).reset_index(drop=True)


    del people['Unnamed: 0']
    people["Time"] = pd.to_datetime(people["Time"])
    people["month"] = people["Time"].dt.month  # 11
    people["day"] = people["Time"].dt.day  # 12
    people["hour"] = people["Time"].dt.hour  # 13

    people = people.to_numpy()

    for c in range(4, 11):
        print("**********" + col[s])
        _range = np.max(abs(people[:, c]))
        people[:, c] = people[:, c] / _range
        evd = 0
        for m in range(6):
            for d in range(1, day[m]+1):
                for h in range(24):
                    count = 0
                    temp = people[np.where((people[:, 11] == m + 1) & (people[:, 12] == d) & (people[:, 13] == h)), c]
                    lng_lat = people[np.where((people[:, 11] == m + 1) & (people[:, 12] == d) & (people[:, 13] == h)), 2: 4]

                    img = np.zeros([x_step, y_step, 1])

                    for y in Y:
                        for x in X:
                            img[x, y, 0] = temp[0, count]
                            # img[x - 100:x + 100, y - 100:y + 100, 0] = temp[0, count]
                            count += 1
                    # cv2.imshow("img", img)
                    # cv2.waitKey(0)
                    cv2.imwrite('./CNN/train/' + str(point_en[pe]) + '/' + str(col[s]) + '/' + str(m+1) + '-' + str(d) + '-' + str(h) + '-' + str(renp[evd, 0]) + '.png', img*255)
                    evd += 1
                    # print('down')
        s += 1
    pe += 1