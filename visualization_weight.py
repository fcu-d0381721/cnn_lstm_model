import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
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

weights = getWeightsForLayer("fuzzy_layer", "./time/DonghuEveschoolday/weight/weight1/mse_model_0001200.h5")
print(weights[0].shape)

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
print(x_y_all.shape)
#
all_x_y = tf.reshape(x_y_all, [189, 9])
out = all_x_y / weights[0][:, None] ** 2
print(weights[0].shape)
time.sleep(100)
out_exp = tf.math.exp(-0.5 * out)
out_sum = tf.reduce_sum(out_exp, 1)
nor_out = out_exp / out_sum[:, None]
nor_out = tf.reshape(nor_out, [189, 3, 3])
print(nor_out)



import numpy as np
import matplotlib.pyplot as plt




#
# for i in range(189):
#     plt.imshow(nor_out[i, :, :], cmap=plt.get_cmap('jet'), interpolation='nearest', vmin=0, vmax=4)
#     plt.colorbar()
#     # plt.show()
#     plt.savefig('./image300/' + str(i) + '.png')
#     plt.clf()