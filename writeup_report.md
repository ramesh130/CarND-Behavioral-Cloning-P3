#**Behavioral Cloning** 

##Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* BehavorCloning.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The BehavorCloning.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model architecture is based on the [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf) on end-to-end learning. In this paper, a CNN is implemented with 5 convolutional layers, followed by 1 flattened layers, and then by 3 fully connected layers. At the end of the network is a single neuron which generates the steering angle. My CNN follows the similar architecture, except it contains 4 convolutional layers and 4 fully connected layers.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. A dropout layer is added after every convolution and fully connected layer. The layer eliminates a percentage of the output value to help the algorithm learn a more robust model.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use one of the known well performing model like LeNet, GoogLenet etc. Then I came across the model presented by Nvidia and decide to use it as they have already spent lot of time analyzing the architecture and came up with a well performing model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To fix this, I must enrich my dataset. Due to time limitation, I was not able to do so.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road for some distance.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.
```
Layer (type)                     Output Shape          Param       Connected to                     
====================================================================================================
cn1 (Convolution2D)              (None, 9, 39, 24)     672         convolution2d_input_3[0][0]      
____________________________________________________________________________________________________
dropout_17 (Dropout)             (None, 9, 39, 24)     0           cn1[0][0]                        
____________________________________________________________________________________________________
activation_17 (Activation)       (None, 9, 39, 24)     0           dropout_17[0][0]                 
____________________________________________________________________________________________________
cn2 (Convolution2D)              (None, 4, 19, 12)     2604        activation_17[0][0]              
____________________________________________________________________________________________________
dropout_18 (Dropout)             (None, 4, 19, 12)     0           cn2[0][0]                        
____________________________________________________________________________________________________
activation_18 (Activation)       (None, 4, 19, 12)     0           dropout_18[0][0]                 
____________________________________________________________________________________________________
cn3 (Convolution2D)              (None, 2, 17, 8)      872         activation_18[0][0]              
____________________________________________________________________________________________________
dropout_19 (Dropout)             (None, 2, 17, 8)      0           cn3[0][0]                        
____________________________________________________________________________________________________
activation_19 (Activation)       (None, 2, 17, 8)      0           dropout_19[0][0]                 
____________________________________________________________________________________________________
cn4 (Convolution2D)              (None, 1, 16, 4)      132         activation_19[0][0]              
____________________________________________________________________________________________________
dropout_20 (Dropout)             (None, 1, 16, 4)      0           cn4[0][0]                        
____________________________________________________________________________________________________
activation_20 (Activation)       (None, 1, 16, 4)      0           dropout_20[0][0]                 
____________________________________________________________________________________________________
flatten (Flatten)                (None, 64)            0           activation_20[0][0]              
____________________________________________________________________________________________________
fc1 (Dense)                      (None, 20)            1300        flatten[0][0]                    
____________________________________________________________________________________________________
dropout_21 (Dropout)             (None, 20)            0           fc1[0][0]                        
____________________________________________________________________________________________________
activation_21 (Activation)       (None, 20)            0           dropout_21[0][0]                 
____________________________________________________________________________________________________
fc2 (Dense)                      (None, 20)            420         activation_21[0][0]              
____________________________________________________________________________________________________
dropout_22 (Dropout)             (None, 20)            0           fc2[0][0]                        
____________________________________________________________________________________________________
activation_22 (Activation)       (None, 20)            0           dropout_22[0][0]                 
____________________________________________________________________________________________________
fc3 (Dense)                      (None, 20)            420         activation_22[0][0]              
____________________________________________________________________________________________________
dropout_23 (Dropout)             (None, 20)            0           fc3[0][0]                        
____________________________________________________________________________________________________
activation_23 (Activation)       (None, 20)            0           dropout_23[0][0]                 
____________________________________________________________________________________________________
fc4 (Dense)                      (None, 20)            420         activation_23[0][0]              
____________________________________________________________________________________________________
dropout_24 (Dropout)             (None, 20)            0           fc4[0][0]                        
____________________________________________________________________________________________________
activation_24 (Activation)       (None, 20)            0           dropout_24[0][0]                 
____________________________________________________________________________________________________
fc5 (Dense)                      (None, 1)             21          activation_24[0][0]              
====================================================================================================
Total params: 6861
```

####3. Creation of the Training Set & Training Process

I used the data set provided by Udacity as I found it diffult to get good data set buy running the simulator myself.

I used `train_test_split()` method to divide the data set into tarining and test set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
