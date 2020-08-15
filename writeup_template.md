# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/image1.png "Model Visualization"
[image2]: ./writeup/image2.jpg "Center Camera Image"
[image3]: ./writeup/image3.jpg "Recovery Image"
[image4]: ./writeup/image4.jpg "Recovery Image"
[image5]: ./writeup/image5.jpg "Recovery Image"
[image6]: ./writeup/image6.jpg "Track 2 Image"
[image7]: ./writeup/image7.jpg "Track 2 Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is the same model as the [model described in this paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) which consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 95-99).

The model includes RELU layers to introduce nonlinearity (code line 95-99), and the data is normalized in the model using a Keras lambda layer (code line 90) and a cropping layer (code line 91).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 102, 104, 106).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 111). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with learning rate manually tuned to 1e-4. It was the best parameter for reducing training losses and validation losses.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving reversely, using track 1 and track 2.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I don't think there was much involvement for finding the right architecture of the model except adding pre-processing layer (normalization and cropping) and some dropout to reduce overfitting. The overall strategy for deriving a model architecture was from the lecture video with some research on finding the model and construct the right layers, which were different from lecture.

And of course, to start the iteration process, I first started with a very simple regression model and then moved on to LeNet.

At the beginning, I was aware that these are not going to be the final solution. They are used to explore the pipeline and understand the problem. Those model couldn't solve the problem since the images are much more complicated and needed a few layers of convolution to capture all the essential features of the images.

Simple regression model didn't work well and wasn't able to turn properly (always turning right).

LeNet model with default data didn't work well and stuck at bridge when there's no clear road lanes in the image.

Some other solution I had was without dropout, the model ran with 3 epochs works a bit better since it hadn't overfit. With 10 epochs, it went off-track near the end where the right side of the route is dirt and there's a large runoff area. This has actually been the major problem I have seen throughout the project.

I tried to collect more data around that part of the track but it wasn't improve too much. And it also takes a long time to validate since that part of the track is near the end.

One of the problem I realized was that due to VM, my training data was pretty poor (this was compared with the provided training data) and my model performed much better.

So I decided to download the simulator onto my local machine and generate training data there.

I have 2 laps of perfectly centered driving. 1 lap of recovery. 1 lap of track 2 centered driving.

And indeed, this data made the model able to finish track 1.

While training, I saw the validation loss went down until 3rd epoch and went up again all the way till 10.

So I decide to add a few dropout layer in the fully connected layer to combat this. The end result was validation loss doesn't go up much and stay roughly the same while training loss went down epoch by epoch.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 88-108) consisted of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 and RELU as activation function. This is followed by a few layers of fully connected layer with dropouts.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from these cases. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

![alt text][image6]
![alt text][image7]

In my code, I have augmented images but the final model didn't use these because it wasn't helping much as I found out.

After the collection process, I had 4439 number of data points. I then preprocessed this data by normalizing and cropping.


I put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by training loss was still going down but validation loss has been going up and down for a few epochs. I used an adam optimizer with a lower learning rate of 1e-4 since I found that a higher learning rate stopped reducing training loss after a few epochs.

And finally, here's the link to the [video](https://youtu.be/XOBLuwXn8lY) that shows the vehicle was able to navigate track 1 successfully.
