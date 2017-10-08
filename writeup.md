# **Behavioral Cloning**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

## The Result

![driving video](https://github.com/wandonye/carnd_P3/blob/master/run1.gif "Track 1")

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md (this file) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a modification of NVidia's network (model.py lines 40-69).


#### 2. Attempts to reduce overfitting in the model
I augmented the training data with perspective transformation (see the next section)

I trained the model for only 3 epoch.

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line  123-126). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 130).

#### 4. Appropriate training data
All my training data were augmentation of the provided data. I didn't drive with the simulator due to my terrible gaming skill.

The original training data were extremely unbalanced. Most frames were little to no steering. Even with the help of right and left camera, the steering angle will be mainly -0.2, 0, 0.2 (0.2 was my choice of compensation of steering for the camera position). A balanced set of data should have steering angle distributed evenly in a reasonable range.

I used perspective transformation to create a balanced data set. For details about my method, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

For this project, model design is not as important as training data preparation.

I started with LeNet and familiarized myself with the workflow and jumped directly to Nvidia model since it has been proven to be a working model. My modification to the model is purely experimental: I changed the 2nd and 3rd conv layers in to 3x3, and 4th, 5th conv layers to 5x5. I didn't use dropout in order to save training time. With the help of diversified training data, I was able to avoid overfitting without dropout.

#### 2. Final Model Architecture

The final model architecture (model.py lines 40-69) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					|
|:------------------------:|:---------------------------------------------:|
| Input         		| 320x160x3 RGB image   							|
| Cropping2D        | Crop out top 75 pixels and bottom 25 pixels for each column 	|
| Normalization        | So that values lie in [-0.5, 0.5]|
| 24 x Conv 5x5     	| 2x2 stride, valid padding|
| RELU					|												|
| 36 x Conv 3x3     	| 1x1 stride, valid padding|
| RELU					|												|
| 48 x Conv 3x3     	| 1x1 stride, valid padding, outputs 12x12x16 	|
| RELU					|												|
| 64 x Conv 5x5     	| 2x2 stride, valid padding|
| RELU					|												|
| 64 x Conv 5x5     	| 2x2 stride, valid padding|
| RELU					|												|
| Flatten		|  |
| Fully connected		| 100 neurons		|
| RELU					|												|
| Fully connected		| 50 neurons					|
| RELU					|												|
| Fully connected		| 10 neurons					|
|	RELU			|												|
|	Output			| linear activation function |

#### 3. Creation of the Training Set & Training Process

All my training data are from the provided [sample data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). Nothing recorded from the simulator myself.

I first loaded all the steering angles and visualized the distribution:

![steering angle distribution](https://github.com/wandonye/carnd_P3/blob/master/steering_distribution.png)

Obviously the labels are too concentrated around 0. This means in order to give a model with low mean squared error on the validation data, the model just needs to predict more values close to 0.

Then I took the left camera and right camera into consideration. For the left camera, I adjusted the steering by 0.2 to the right, i.e. steering += 0.2. For right camera, I adjusted the steering by 0.2 to the left, i.e. steering -= 0.2. Thus the new distribution of steering angle is like

![steering angle distribution with two extra cams](https://github.com/wandonye/carnd_P3/blob/master/steering_distr_3cam.png)

Then I used perspective transformation to evenly distribute the three major steering direction to any direction between -0.7 and 0.7.

The key observation motivate my method is the following (Note that the 2nd and 3rd image are generated with perspective transformation):

![rolling camera to the left and right](https://github.com/wandonye/carnd_P3/blob/master/roll_cam_demo.png =300x "The 2nd and 3rd image are generated with perspective transformation")

It's not hard to see that the second image needs more steering to the right, and the third image needs more steering to the left.

The way I got the second image from the first image is demonstrated by the diagram below.

![rolling camera and perspective](https://github.com/wandonye/carnd_P3/blob/master/roll_cam_perspective.png)


When the camera pans to the left, its image only contains the red cone region. Thus the points A, B, C', D' will be at the four corners of the newly captured image at position O'. So if I apply a perspective transformation which maps (A, B, C', D') to (A, B, C, D) to the old image captured at position O, what I get will be the image taken by the camera at O'. This analysis assumed all objects are on the same plane ABCD. Otherwise, there will be blocking region change as the camera moves. But for the purpose of augmentation of training data, this doesn't matter since we mainly care about the camera's (car's) position relative to the lane.

This augmentation function is implemented in RollCam(img, deg) (model.py line 71-86). If the input image has a steering angle `a`, and if we set `deg=d`, then the output image will require a steering angle of `a-d`.

I created a generator with the above RollCam function. For each original image, I panned the camera by a random number between -0.5 to 0.5. The resulting image and its flipped version together with their adjusted steering angles were sent to train the model.

This effectively even out the 8036x3=24108 images whose steering angles were mostly -0.2, 0 and 0.2 to any number between -0.7 to 0.7.

To separate training set and validation set, I used `train_test_split` from scikit-learn. This function always shuffles the input data first, so I don't have to shuffle before split.

The generator was applied to both the training data and the testing data.

I used an Adam optimizer so that manually training the learning rate wasn't necessary. I trained the model for 3 epochs, since mean squared error won't drop much after that.

The resulting model drives safely without touching the lane borders. Because its steering angle is seldomly 0, frequent steering influenced the driving speed a little bit. I believe this can be solved by appending a smoothing function to the output of the neural network.
