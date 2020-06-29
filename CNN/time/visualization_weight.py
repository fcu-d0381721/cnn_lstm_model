import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import YBmodel_CNN_LSTM as ybcl
import pandas as pd
import time


def isGroup(obj):
    if isinstance(obj, h5py.Group):
        return True
    return False

def isDataset(obj):
    if isinstance(obj, h5py.Dataset):
        return True
    return False

def getDatasetfromGroup(datasets, obj):
    if isGroup(obj):
        for key in obj:
            x = obj[key]
            getDatasetfromGroup(datasets, x)
    else:
        datasets.append(obj)

def getWeightsForLayer(layerName, fileName):
    weights = []
    with h5py.File(fileName, mode='r') as f:
        for key in f:
            if layerName in key:
                obj = f[key]
                datasets = []
                getDatasetfromGroup(datasets, obj)

                for dataset in datasets:
                    w = np.array(dataset)
                    weights.append(w)
    return weights

def create_grid():
    x, y = np.mgrid[-1:2, -1:2]
    x1, y1 = np.mgrid[-2:1, -2:1]
    x_al = np.append(x, x1, axis=0)
    y_al = np.append(y, y1, axis=0)
    x, y = np.mgrid[0:3, 0:3]
    x_al = np.append(x_al, x, axis=0)
    y_al = np.append(y_al, y, axis=0)
    x, y = np.mgrid[-2:1, -1:2]
    x_al = np.append(x_al, x, axis=0)
    y_al = np.append(y_al, y, axis=0)
    x, y = np.mgrid[-2:1, 0:3]
    x_al = np.append(x_al, x, axis=0)
    y_al = np.append(y_al, y, axis=0)
    x, y = np.mgrid[-1:2, 0:3]
    x_al = np.append(x_al, x, axis=0)
    y_al = np.append(y_al, y, axis=0)
    x, y = np.mgrid[-1:2, -2:1]
    x_al = np.append(x_al, x, axis=0)
    y_al = np.append(y_al, y, axis=0)
    x, y = np.mgrid[0:3, -1:2]
    x_al = np.append(x_al, x, axis=0)
    y_al = np.append(y_al, y, axis=0)
    x, y = np.mgrid[0:3, -2:1]
    x_al = np.append(x_al, x, axis=0)
    y_al = np.append(y_al, y, axis=0)
    x_al = x_al.reshape(9, 3, 3)
    y_al = y_al.reshape(9, 3, 3)
    x_al = tf.Variable(x_al)
    y_al = tf.Variable(y_al)
    x_y_ = x_al ** 2 + y_al ** 2
    x_y_ = tf.reshape(x_y_, [1, 9, 3, 3])
    x_y_all = tf.keras.backend.repeat_elements(x_y_, 21, 0)
    x_y_all = tf.dtypes.cast(x_y_all, tf.float32)

    return x_y_all

traindata = ['DonghuEveschoolday', 'DonghuMorschoolday', 'NeihuEveschoolday', 'NeihuEveworkday', 'NeihuMorschoolday', 'NeihuMorworkday', 'Songsanworkday', 'Zhongxiaoworkday', 'Songsanweekend', 'Zhongxiaoweekend']
Threshold = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
Threshold_Folder = ['06', '065', '07', '075', '08', '085']
c = 0
for th in Threshold:
    for t in traindata:
        sum = []
        x_y_all = create_grid()
        for i in range(15):  # 讀取每一個weight值 並加總除以次數
            weights = getWeightsForLayer("fuzzy_layer", "./" + t + "/weight/weight" + str(i+1) + "/mse_model_0001200.h5")
            if len(sum) == 0:
                sum = weights[0]
            else:
                sum = sum + weights[0]
        sum = sum/15

        model = ybcl.yb_model('', '', 3)
        # 以下開始做二維高斯函數 先將產生出來的網格乘上加總相除過後的標準差(exp)在做標準化
        all_x_y = tf.reshape(x_y_all, [189, 9])
        out = all_x_y / sum[:, None] ** 2
        out_exp = tf.math.exp(-0.5 * out)
        out_sum = tf.reduce_sum(out_exp, 1)
        nor_out = out_exp / out_sum[:, None]  # 做標準化的地方
        nor_out = tf.reshape(nor_out, [189, 3, 3])

        arr = [[1, 1], [2, 2], [0, 0], [2, 1], [2, 0], [1, 0], [1, 2], [0, 1], [0, 2]]
        for i in range(189):
            out = nor_out[i, :, :]
            if out[arr[i % 9]] >= th:
                plt.imshow(out, cmap=plt.get_cmap('jet'), interpolation='nearest', vmin=0, vmax=1)
                plt.colorbar()
                plt.savefig('./' + t + '/excess_' + Threshold_Folder[c] + '/' + str(i) + '.png')
                plt.clf()
        print('---- down for this ' + t + ' ! ----')
    print('---- down for this Threshold ' + Threshold_Folder[c] + ' ! ----')
    c += 1
