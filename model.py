import os

import numpy as np
import pandas as pd
import cv2
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D, Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.pooling import MaxPooling2D

def loadData(data_dir):
    # This function load image paths and steering angles from the driving log
    header = ['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed']
    driving_log = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'),
                              header=0, names=header, comment='#')
    driving_log['left'] = data_dir + os.path.sep + driving_log['left'].str.strip()
    driving_log['center'] = data_dir + os.path.sep + driving_log['center'].str.strip()
    driving_log['right'] = data_dir + os.path.sep + driving_log['right'].str.strip()

    print('Total images found: ', len(driving_log))
    img_path = []
    steering = []
    for index, row in driving_log.iterrows():
        d = float(row['steering'])
        img_path.append(row['left'])
        steering.append(d+0.2)
        img_path.append(row['center'])
        steering.append(d)
        img_path.append(row['right'])
        steering.append(d-0.2)

    return img_path, steering

def Nvidia():
    # Modified NVidia model
    model = Sequential()
    # Crop the original image to remove the sky and hood.
    model.add(Cropping2D(cropping=((75,25), (0,0)), input_shape=(160,320,3)))
    # Normaize the input values so that they lie in (-0.5, 0.5)
    model.add(Lambda(lambda x: x / 255.0 - 0.5, name="normalization"))

    model.add(Conv2D(24, 5, 5, subsample=(2,2), init="he_normal", border_mode="valid", name="conv1"))
    model.add(ELU())
    model.add(Conv2D(36, 3, 3, init="he_normal", border_mode="valid", name="conv2"))
    model.add(ELU())
    model.add(Conv2D(48, 3, 3, init="he_normal", border_mode="valid", name="conv3"))
    model.add(ELU())
    model.add(Conv2D(64, 5, 5, subsample=(2,2), init="he_normal", border_mode="valid", name="conv4"))
    model.add(ELU())
    model.add(Conv2D(64, 5, 5, subsample=(2,2), init="he_normal", border_mode="valid", name="conv5"))
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

def RollCam(img,deg):
    # This function perform perspective tranformation to img
    # The resulting image roughly equivalent to shift the camera
    # by the value of deg
    # The rsulting image is expected to have steering angle equals to
    # the steering angle of the original image (img) - deg

    if deg==0: return dst
    d = deg*150
    pts1 = (np.float32([[d,d/40],[d,159-d/40],[319,0],[319,159]]) if d>0
            else np.float32([[0,0],[0,159],[319+d,-d/40],[319+d,159+d/40]]))
    pts2 = np.float32([[0,0],[0,159],[319,0],[319,159]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(320,160))
    return dst

def generator(X,y,batch_size=32):
    num_samples = len(X)
    while 1:
        X,y = shuffle(X,y)
        for offset in range(0, num_samples, batch_size):
            batch_X = X[offset:offset+batch_size]
            batch_y = y[offset:offset+batch_size]

            images = []
            angles = []
            for i in range(len(batch_X)):
                originalImage = cv2.cvtColor(cv2.imread(batch_X[i]),
                                             cv2.COLOR_BGR2RGB)
                # The trick below is very important to balance the data
                # generate a random pertubation angle
                adjust = random.uniform(-0.5,0.5)
                # adjust the original image by the angle generated above
                # i.e. tilt the camera to the right/left
                image = RollCam(originalImage,adjust)
                images.append(image)
                # compensate the steering angle
                # When the cam is rolled to the left, then the steering angle
                # should be compensated to the right
                steering = batch_y[i]-adjust
                angles.append(steering)
                # Flipping, this will balance left and right steering
                images.append(cv2.flip(image,1))
                angles.append(steering*-1.0)

            inputs = np.array(images)
            outputs = np.array(angles)
            yield shuffle(inputs, outputs)

if __name__ == '__main__':
    img_list, angles = loadData('data')
    X_train, X_valid, y_train, y_valid = train_test_split(img_list, angles, test_size=0.2)

    train_generator = generator(X_train,y_train, batch_size=32)
    validation_generator = generator(X_valid,y_valid, batch_size=32)

    model = Nvidia()

    optimizer = Adam(1e-4)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(X_train), validation_data=validation_generator, \
                 nb_val_samples=len(X_valid), nb_epoch=3, verbose=1)

    model.save('model.h5')
