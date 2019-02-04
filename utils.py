
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import Model
import cv2
from os import path
from PIL import Image
from collections import defaultdict
import pandas as pd
import tensorflow as tf

SZ = 128
KERNEL_SZ = 32
IM_SZ = 512
data_path = 'E:\\all\\'
train_path = path.join(data_path, 'train')
test_path = path.join(data_path, 'test')
data = pd.read_csv(path.join(data_path, 'train.csv'))


def pretty_print(txt, arr):
    print(txt)
    for idx, value in enumerate(arr):
        print('{0}: {1}'.format(idx, value))


def my_loss(y_true, y_pred):
    res = 0
    ind = 0
    tf.nn.relu(y_pred)
    for i in range(len(y_true)):
        if y_pred[i] > 0:
            res += y_true[i] - y_pred[i]
            ind += 1
    return res ** 2 / ind

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def f1(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def load_dataset():
    train_dataset = defaultdict(list)
    for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
        labels = np.array([int(label) for label in labels])
        for label in labels:
            train_dataset[label].append({
                'path': path.join(train_path, name),
                'labels': labels})
    return train_dataset


def pretty_print(txt, arr):
    print(txt)
    for idx, value in enumerate(arr):
        print('{0}: {1}'.format(idx, value))

def show_image(path):
    image_blue_ch = np.array(Image.open(path + '_blue.png'))
    image_red_ch = np.array(Image.open(path + '_red.png'))
    # image_yellow_ch = np.array(Image.open(path + '_yellow.png'))[x:x+SZ, y:y+SZ]
    image_green_ch = np.array(Image.open(path + '_green.png'))
    # cv2.imshow("Image", image_blue_ch)
    # cv2.waitKey(0)

    image = np.stack((
        image_red_ch,
        image_green_ch,
        image_blue_ch), -1)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

def visualize_layer(layer, model, img_to_visualize):

    layer = model.get_layer(layer)

    model = Model(input=model.input, output=layer.output)
    img = img_to_visualize[np.newaxis]
    to_show = model.predict(img)

    for i in range(0, 5):
        img = np.squeeze(to_show, axis=0)
        #img = img[:, :, i + 10]
        img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(np.uint8)
        cv2.imshow('wnd', img)
        cv2.waitKey(0)


    '''inputs = [K.learning_phase()] + model.inputs

_convout1_f = K.function(inputs, [layer.output])

def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print('Shape of conv:', convolutions.shape)

    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12, 8))
    for i in range(len(convolutions)):
        ax = fig.add_subplot(n, n, i + 1)
        ax.imshow(convolutions[i], cmap='gray')'''
