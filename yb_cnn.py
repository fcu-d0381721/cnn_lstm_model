import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import os
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from numpy import array
import pandas as pd
import numpy as np
import cv2


def split_sequence(sequence, n_steps):

    X, y = list(), list()
    for i in range(len(sequence[1])):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence[1]) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[0][i:end_ix, :, :, :], sequence[1][end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def cnn_model(cellNumber, batchSize):

    # gauss_kernel = gaussian_kernel(9, 5, 1)
    # gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(3, 64, 64, 7), batch_size=batchSize))  # , batch_input_shape=(batchSize, 1, )))
    model.add(layers.TimeDistributed(layers.Conv2D(2, (2, 2), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPool2D(pool_size=(2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(2, (2, 2), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPool2D(pool_size=(2, 2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.LSTM(cellNumber, stateful=True))
    model.add(layers.Dense(1))
    model.summary()
    return model


def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 64, 64, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (64, 64))
        if label:
          tmp = file.split("-")[3]
          y[i] = int(tmp.split(".")[0])
    if label:
      return x, y
    else:
      return x


def loss(predicted_y, target_y):
    predicted_y = tf.dtypes.cast(predicted_y, tf.float64)
    return tf.keras.losses.mean_squared_error(target_y, predicted_y)


def gaussian_kernel(size: int, mean: float, std: float):
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


@tf.function
def train_step(input, ground_truth):
    with tf.GradientTape() as tape:
        model_outputs = model(input, training=True)
        model_loss = loss(model_outputs, ground_truth)

    gradients_of_model = tape.gradient(model_loss, model.trainable_variables)
    model_optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))

    train_loss(model_loss)
    train_accuracy(model_outputs, ground_truth)


def train(dataset, test_data, epochs):
    for epoch in range(epochs):
        start = time.time()
        count = 1
        for data_batch in dataset:
            train_step(data_batch[0], data_batch[1])
            if count % 72 == 0:
                model.reset_states()
                count += 1

        for data_batch in test_data:
            model.reset_states()
            test_step(data_batch[0], data_batch[1])

        # Save the model every 15 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        template = 'Epoch {}, Time: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              time.time() - start,
                              train_loss.result(),
                              train_accuracy.result(),
                              test_loss.result(),
                              test_accuracy.result()))

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


@tf.function
def test_step(input, ground_truth):
    model_outputs = model(input, training=False)
    model_loss = loss(model_outputs, ground_truth)

    test_loss(model_loss)
    test_accuracy(model_outputs, ground_truth)


def test_return_prediction(test_dataset):
    prediction = np.array([])
    for input, ground_truth in test_dataset:
        model_outputs = model(input, training=False)
        prediction = np.append(prediction, model_outputs.numpy())
    return prediction


if __name__ == '__main__':

    site = '市民太原路口'
    model = cnn_model(100, 1)
    model_optimizer = tf.keras.optimizers.Adam(1e-5)

    checkpoint_dir = 'training_checkpoint'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(model_optimizer=model_optimizer, model=model)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')

    test_stamp = 48

    workspace_dir = './train'
    X_train_Age1, y_train_Age1 = readfile(os.path.join(workspace_dir, "taipei/Age1"), True)
    X_train_Age1 = X_train_Age1[:, :, :, 0].reshape(4344, 64, 64, 1)

    X_train_Age2 = readfile(os.path.join(workspace_dir, "taipei/Age2"), False)
    X_train_Age2 = X_train_Age2[:, :, :, 0].reshape(4344, 64, 64, 1)
    out = np.concatenate((X_train_Age1, X_train_Age2), axis=3)

    X_train_Age3 = readfile(os.path.join(workspace_dir, "taipei/Age3"), False)
    X_train_Age3 = X_train_Age3[:, :, :, 0].reshape(4344, 64, 64, 1)
    out = np.concatenate((out, X_train_Age3), axis=3)

    X_train_Age4 = readfile(os.path.join(workspace_dir, "taipei/Age4"), False)
    X_train_Age4 = X_train_Age4[:, :, :, 0].reshape(4344, 64, 64, 1)
    out = np.concatenate((out, X_train_Age4), axis=3)

    X_train_Age5 = readfile(os.path.join(workspace_dir, "taipei/Age5"), False)
    X_train_Age5 = X_train_Age5[:, :, :, 0].reshape(4344, 64, 64, 1)
    out = np.concatenate((out, X_train_Age5), axis=3)

    X_train_Age6 = readfile(os.path.join(workspace_dir, "taipei/Age6"), False)
    X_train_Age6 = X_train_Age6[:, :, :, 0].reshape(4344, 64, 64, 1)
    out = np.concatenate((out, X_train_Age6), axis=3)

    X_train_total = readfile(os.path.join(workspace_dir, "taipei/Total"), False)
    X_train_total = X_train_total[:, :, :, 0].reshape(4344, 64, 64, 1)
    out = np.concatenate((out, X_train_total), axis=3)
    print(out.shape)
    norm = np.linalg.norm(out)
    X_train_Age1 = out / norm

    y_train_Age1 = y_train_Age1 / y_train_Age1.max(axis=0)
    X_train_Age1 = X_train_Age1.astype(np.float32)
    X, y = split_sequence([X_train_Age1, y_train_Age1], 3)

    train_X = X[0:(len(X) - test_stamp), :, :, :]
    test_X = X[-test_stamp::]
    train_Y = y[0:(len(X) - test_stamp)]
    test_Y = y[-test_stamp::]

    train_model_inputs = tf.Variable(train_X)
    train_outputs = tf.Variable(train_Y)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_model_inputs, train_outputs))
    train_batched_dataset = train_dataset.batch(1)
    test_model_inputs = tf.Variable(test_X)
    test_outputs = tf.Variable(test_Y)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_model_inputs, test_outputs))
    test_batched_dataset = test_dataset.batch(1)

    train(train_batched_dataset, test_batched_dataset, 1000)
    #
    # ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    #
    # if ckpt_manager.latest_checkpoint:
    #     checkpoint.restore(ckpt_manager.latest_checkpoint)
    #
    #     print('Latest checkpoint restored!!', ckpt_manager.latest_checkpoint)
    #
    #     model = checkpoint.model
    #
    #     first = True
    #     for inputs, other_input, ground_truth in test_batched_dataset:
    #         model_outputs = model(inputs, training=False)
    #
    #         if first:
    #             mid_outputs = model_outputs.numpy()
    #             first = False
    #         else:
    #             mid_outputs = np.append(mid_outputs, model_outputs.numpy(),axis=0)
    #
    #     mid_outputs = np.swapaxes(mid_outputs, 0, 1)
    #     fig = plt.figure(figsize=(20,20))

    # summarize the data
    # for i in range(len(X)):
    #     print(X[i], y[i])
    #
    # model.fit(X, y, epochs=500, verbose=0)
    #
    # x_input = array([[1019.3, 15.9, 0., 4.], [1018.9, 16., 0.,  6.], [1018.8, 16.1, 0., 8.]])
    # x_input = x_input.reshape((1, 3, 4))
    # yhat = model.predict(x_input, verbose=0)
    # print(yhat)