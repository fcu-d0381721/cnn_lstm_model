import pandas as pd
import time
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import idw

date = ['2018-01-31', '2018-02-28', '2018-03-31', '2018-04-30', '2018-05-31', '2018-06-30']
position = ['台北', '大直', '內湖', '信義', '松山', '永和']

rain_pos = pd.read_csv("./增加平均距離.csv", encoding='utf-8')
con = np.asarray(rain_pos[['經度', '緯度']])
rain = []
cal = []
hour = []
tempruture = []
for i in date:
    temp = []
    for j in position:
        df = pd.read_csv('./' + j + '/' + i + '.csv', encoding='utf8')
        del df['Unnamed: 0']
        temp.append(df)
    for size in range(len(temp[0])):
        for n in range(6):
            if temp[n].iloc[size:size + 1, 3].values[0] == '...' or temp[n].iloc[size:size + 1, 3].values[0] == 'X' or temp[n].iloc[size:size + 1, 3].values[0] == '/':
                tempruture.append(float(20.0))
                cal.append(temp[n].iloc[size:size + 1, 17].values[0])
                hour.append(temp[n].iloc[size:size + 1, 0].values[0])
            else:
                cal.append(temp[n].iloc[size:size + 1, 17].values[0])
                tempruture.append(float(temp[n].iloc[size:size + 1, 3].values[0]))
                hour.append(temp[n].iloc[size:size + 1, 0].values[0])
            # if temp[n].iloc[size:size + 1, 10].values[0] == 'T' or temp[n].iloc[size:size + 1, 10].values[0] == 'X':
            #     rain.append(0.0)
            #     cal.append(temp[n].iloc[size:size + 1, 17].values[0])
            #     hour.append(temp[n].iloc[size:size + 1, 0].values[0])
            # else:
            #     cal.append(temp[n].iloc[size:size + 1, 17].values[0])
            #     rain.append(float(temp[n].iloc[size:size + 1, 10].values[0]))
            #     hour.append(temp[n].iloc[size:size + 1, 0].values[0])
# print(len(rain))
# my_dict = {i:rain.count(i) for i in rain}
# # del my_dict[0.5]
# # del my_dict[0.0]
# print(my_dict)
# plt.bar(range(len(my_dict)), my_dict.values(), align="center")
# plt.xticks(range(len(my_dict)), list(my_dict.keys()))
# plt.show()

#
amin, amax = min(tempruture), max(tempruture)
for i, val in enumerate(tempruture):
    tempruture[i] = (val-amin) / (amax-amin)

tempruture = np.asarray(tempruture)

for i in range(0, len(tempruture), 6):
    rain1 = np.asarray(tempruture[i:i+6])
    print(rain1)
    print(cal[i])
    print(hour[i])
    idw_tree = idw.tree(con[:, 0:2], rain1)

    spacing = np.linspace(121.45, 121.6, 30)
    spacing1 = np.linspace(24.95, 25.2, 210)
    X2 = np.meshgrid(spacing, spacing1)
    grid_shape = X2[0].shape
    X2 = np.reshape(X2, (2, -1)).T
    z2 = idw_tree(X2)
    norm = colors.Normalize(vmin=0, vmax=1)
    plt.contourf(spacing, spacing1, z2.reshape(grid_shape),norm=norm)
    # plt.show()
    plt.savefig('./temp/' + str(cal[i]) + '_' + str(hour[i]) + '.png')

