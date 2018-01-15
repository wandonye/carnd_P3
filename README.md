# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

---

**Snapshot**

* The video recorded from autonomous driving

![driving video](https://github.com/wandonye/carnd_P3/blob/master/run1.gif "Track 1")

* The model file is stored in [model.h5](https://github.com/wandonye/carnd_P3/blob/master/model.h5)

* To run the model, create an enviroment with [Udacity's starter kit for Self-Driving car Nano Degree](https://github.com/udacity/CarND-Term1-Starter-Kit)

* Results are summerized in [writeup.md](https://github.com/wandonye/carnd_P3/blob/master/writeup.md)

---

**Highlight**

* I didn't collect any new data with the simulator due to my bad driving game skill. Instead, I used computer vision method to generate new data. In particular I used perspective transformation to pertubate the camera direction for each image, and compensated the recorded steering angle accordingly. This balanced the training data, and is crucial for my model training.
