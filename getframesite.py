import pandas as pd
import numpy as np
import time
"""
缺值--以上下取平均補進
並只取目標租借站的人流網格，以便之後作圖

"""
# point = ['北車站', '大直站', '市府站', '港墘站', '田徑場']
point = ['板橋']
M = ['201801', '201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812']
# lng = [121.515, 121.545, 121.565, 121.575, 121.55]
# lat = [25.05, 25.08, 25.04, 25.08, 25.05]
lng = [121.465]
lat = [25.015]
# count = [-0.005, 0, 0.005]
count = [-0.005, 0, 0.005]
inde = 0


def dt_to_np(num, data):
    if num == 12:
        y = np.arange('2018-' + str(num) + '-01', '2019-01-01', dtype='datetime64[h]')
    elif num > 9:
        y = np.arange('2018-' + str(num) + '-01', '2018-' + str(num + 1) + '-01', dtype='datetime64[h]')
    elif num + 1 == 10:
        y = np.arange('2018-0' + str(num) + '-01', '2018-' + str(num + 1) + '-01', dtype='datetime64[h]')
    else:
        y = np.arange('2018-0' + str(num) + '-01', '2018-0' + str(num + 1) + '-01', dtype='datetime64[h]')
    data = data.sort_values(by=['Time']).reset_index(drop=True)
    data = data.to_numpy()
    data[:, 0] = np.array(data[:, 0])
    nps = np.array(data[:, 0], dtype='datetime64')

    for i in y:
        if i not in nps:
            # print(i)
            a = data[np.where(i - 1 == nps), 4:]
            b = data[np.where(i + 1 == nps), 4:]
            # print(a, b)
            wanna_data = np.around(np.mean([a, b], 0).astype(float))[0, 0, :]
            wanna_data = wanna_data.astype(object)
            wanna_data = np.insert(wanna_data, 0, data[np.where(i - 1 == nps), 3], axis=0)
            wanna_data = np.insert(wanna_data, 0, data[np.where(i - 1 == nps), 2], axis=0)
            wanna_data = np.insert(wanna_data, 0, data[np.where(i - 1 == nps), 1], axis=0)
            wanna_data = np.insert(wanna_data, 0, i, axis=0)

            data = np.append(data, [wanna_data], axis=0)

    df = pd.DataFrame(data, columns=['Time', 'TownID', 'Longitude', 'Latitude', 'Age1', 'Age2', 'Age3', 'Age4', 'Age5', 'Age6', 'Total'])
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values(by=['Time']).reset_index(drop=True)
    return df


for p in point:
    num = 1
    print(p)
    for m in M:
        flag = True
        # clear_point_data = []
        df = pd.read_csv('./DATA/人流/' + m + '.csv', encoding='utf-8')
        del df['Unnamed: 0']
        # df1 = pd.read_csv('./DATA/東湖/' + m + '.csv', encoding='utf-8')
        # del df1['Unnamed: 0']
        # df = df.append(df1).reset_index(drop=True)
        # print(df)
        for i in count:
            for j in count:
                # print(round(lng[inde] + i, 3), round(lat[inde] + j, 3))
                con = df['Longitude'] == round(lng[inde] + i, 3)
                con1 = df['Latitude'] == round(lat[inde] + j, 3)
                temp = df[con & con1].reset_index(drop=True)
                print(temp)
                time.sleep(100)
                # print
                # print(i, j)
                if len(temp) != 0:
                    if flag:
                        clear_point_data = dt_to_np(num, temp)
                        flag = False
                    else:
                        t = dt_to_np(num, temp)
                        clear_point_data = clear_point_data.append(t).reset_index(drop=True)
        print(len(clear_point_data))
        clear_point_data.to_csv('./DATA/' + p + '/' + m + '.csv')
        num += 1
    inde += 1