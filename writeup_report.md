# **Behavioral Cloning** 

## Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
[//]: # (Image References)

[image2]: ./examples/center_2016_12_01_13_33_10_173.jpg "Centre camera"
[image3]: ./examples/left_2016_12_01_13_40_11_077.jpg "Left camera"
[image4]: ./examples/right_2016_12_01_13_31_13_686.jpg "Right camera"
[image22]: ./examples/center_2017_03_14_20_42_53_900.jpg "Centre camera"
[image33]: ./examples/left_2017_03_14_20_42_53_900.jpg "Left camera"
[image44]: ./examples/right_2017_03_14_20_42_53_900.jpg "Right camera"
[image5]: ./examples/center_2017_03_14_20_42_55_719.jpg "Offroad Centre camera"
[image6]: ./examples/left_2017_03_14_20_42_55_719.jpg "Offroad Left camera"
[image7]: ./examples/right_2017_03_14_20_42_55_824.jpg "Offroad Right camera"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* BehavorCloning.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The BehavorCloning.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
Line 167 - 305
The simple neural network with 3 convolution layers and 2 fully connected layers is used. The first layer is a convolution layer of kernel size 1x1 and a depth of 3 and the goal of this layer is so the model can figure out the best color space. Following this the model uses 3 convolution layers each followed by RELU activation and a maxpool layer of size (2x2). The first convolution layer has a kernel size of 3x3, stride of 2x2 and a depth of 32. The second convolution layer has a kernel size of 3x3, stride of 2x2 and a depth of 64. The third convolution layer has a kernel size of 3x3, stride of 1x1 and a depth of 128.
After this the output is flattened. Dropout of 50% is applied and then there are 2 dense layers of 128 neurons. The final layer is an output layer of 1 neuron. All the layers are followed by RELU activation to introduce non-linearity.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. A dropout layer is added after every convolution and fully connected layer. The layer eliminates a percentage of the output value to help the algorithm learn a more robust model.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning
Line 305 - 309
Within the neural network the following parameters were tuned:
1. Neural network structure — Number of convolution, max pool and dense layers
2. Learning rate in the optimizer
3. No. Of epoches — I found a value b/w 5–8 worked best. All the intermediate models were saved using Keras checkpoint (model.py lines 329–333) and tested in the simulator
4. Training samples per epoch — I found 20k to be the optimal number

I found that validation loss was not a very good indicator of the quality of the model and the true test was performance in the simulator. However models with very high validation loss performed poorly. But within different epochs, models with higher validation loss could have better performance.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use one of the known well performing model like LeNet, GoogLenet etc. However I went ahead and used a simpler model to reduce the training time. The model may not be the best but works reasonably well.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road for some distance.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
zeropadding2d_1 (ZeroPadding2D)  (None, 162, 322, 3)   0           zeropadding2d_input_1[0][0]      
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 82, 322, 3)    0           zeropadding2d_1[0][0]            
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 64, 64, 3)     0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
lambda_2 (Lambda)                (None, 64, 64, 3)     0           lambda_1[0][0]                   
____________________________________________________________________________________________________
color_conv (Convolution2D)       (None, 64, 64, 3)     12          lambda_2[0][0]                   
____________________________________________________________________________________________________
conv1 (Convolution2D)            (None, 32, 32, 32)    896         color_conv[0][0]                 
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 32, 32, 32)    0           conv1[0][0]                      
____________________________________________________________________________________________________
pool1 (MaxPooling2D)             (None, 31, 31, 32)    0           activation_1[0][0]               
____________________________________________________________________________________________________
conv2 (Convolution2D)            (None, 16, 16, 64)    18496       pool1[0][0]                      
____________________________________________________________________________________________________
relu2 (Activation)               (None, 16, 16, 64)    0           conv2[0][0]                      
____________________________________________________________________________________________________
pool2 (MaxPooling2D)             (None, 8, 8, 64)      0           relu2[0][0]                      
____________________________________________________________________________________________________
conv3 (Convolution2D)            (None, 8, 8, 128)     73856       pool2[0][0]                      
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 8, 8, 128)     0           conv3[0][0]                      
____________________________________________________________________________________________________
pool3 (MaxPooling2D)             (None, 4, 4, 128)     0           activation_2[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2048)          0           pool3[0][0]                      
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 2048)          0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense1 (Dense)                   (None, 128)           262272      dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 128)           0           dense1[0][0]                     
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 128)           0           activation_3[0][0]               
____________________________________________________________________________________________________
dense2 (Dense)                   (None, 128)           16512       dropout_2[0][0]                  
____________________________________________________________________________________________________
output (Dense)                   (None, 1)             129         dense2[0][0]                     
====================================================================================================
Total params: 372,173
Trainable params: 372,173
Non-trainable params: 0
```

#### 3. Creation of the Training Set & Training Process

I used the data set provided by Udacity as well as data I generated to augment the data set.
Here is an example image of center lane driving:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image22]
![alt text][image33]
![alt text][image44]

I also used images of vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover in case it goes off the road. Below are images for data when the car veers offroad-

![alt text][image5]
![alt text][image6]
![alt text][image7]

I used `train_test_split()` method to divide the data set into tarining and test set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
