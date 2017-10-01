import os

import numpy as np
import pandas as pd


def loadData(data_dir):
    header = ['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed']
    driving_log = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'),
                              header=0, names=header, comment='#')
    driving_log['left'] = data_dir + os.path.sep + driving_log['left'].str.strip()
    driving_log['center'] = data_dir + os.path.sep + driving_log['center'].str.strip()
    driving_log['right'] = data_dir + os.path.sep + driving_log['right'].str.strip()

    print('Total images found: ', len(driving_log))

    return driving_log

def Nvidia(input_shape=(200,66,3)):
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, name="image_normalization", input_shape=input_shape))

    model.add(Convolution2D(24, 5, 5, name="convolution_1", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, name="convolution_2", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, name="convolution_3", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, name="convolution_4", border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, name="convolution_5", border_mode="valid", init='he_normal'))
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
    dataset = loadData('data')
    print(dataset[:5])
