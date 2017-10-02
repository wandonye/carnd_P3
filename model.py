import os

import numpy as np
import pandas as pd
import cv2


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

def loadIMG(driving_log, resize_shape):
    imgs = []
    for img_file in driving_log['center'].items():
        print(img_file)
        img = cv2.imread(img_file)
        imgs.append(preprocess_image(img,resize_shape))
    return imgs

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
    loadIMG(driving_log, resize_shape)
