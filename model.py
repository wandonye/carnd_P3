import os

import numpy as np
import pandas as pd
import cv2

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.pooling import MaxPooling2D

def loadData(data_dir):
    header = ['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed']
    driving_log = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'),
                              header=0, names=header, comment='#')
    driving_log['left'] = data_dir + os.path.sep + driving_log['left'].str.strip()
    driving_log['center'] = data_dir + os.path.sep + driving_log['center'].str.strip()
    driving_log['right'] = data_dir + os.path.sep + driving_log['right'].str.strip()

    print('Total images found: ', len(driving_log))

    return driving_log

def preprocess_image(image, resize_shape, color_transformation=cv2.COLOR_BGR2RGB):
    bottom = int(.84 * image.shape[0])
    top = int(.31 * image.shape[0])
    image = image[top:bottom,:]
    image = cv2.resize(image, resize_shape)
    image = cv2.cvtColor(image, color_transformation)
    return image

def loadLabeledIMG(driving_log, resize_shape):
    imgs = []
    labels = []
    for row in driving_log.iterrows():
        img = cv2.imread(row[1]['center'])
        imgs.append(preprocess_image(img,resize_shape))
        labels.append(float(row[1]['steering']))
    return np.array(imgs), np.array(labels)

def Nvidia(input_shape=(200,66,3)):
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, name="normalization", input_shape=input_shape))

    model.add(Convolution2D(24, 5, 5, name="conv1", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, name="conv2", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, name="conv3", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, name="conv4", border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, name="conv5", border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Flatten())

    model.add(Dense(100, name="hidden1", init='he_normal'))
    model.add(ELU())
    model.add(Dense(50, name="hidden2", init='he_normal'))
    model.add(ELU())
    model.add(Dense(10, name="hidden3", init='he_normal'))
    model.add(ELU())

    model.add(Dense(1, name="steering_angle", activation="linear"))

    return model

if __name__ == '__main__':
    driving_log = loadData('data')
    input_shape = (66, 200, 3)
    resize_shape = (200, 66)
    X, y = loadLabeledIMG(driving_log, resize_shape)
    model = Nvidia(input_shape)
    output_shape = (-1,) + input_shape

    optimizer = Adam(1e-4)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    model.fit(X,y, validation_split=0.2, shuffle=True)

    model.save('model.h5')
