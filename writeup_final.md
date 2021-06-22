# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals of this project are the following:
* Load the data set of the German Traffic Signs (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1a]: ./examples/stats_before.jpg "Visualization"
[image1b]: ./examples/stats_after.jpg "Visualization"
[image2]: ./examples/gray_scaled.jpg "Grayscaling"
[image3]: ./examples/noises.jpg "Random Noise"
[image4]: ./own_pics/9-no_passing.jpg "Traffic Sign 1"
[image5]: ./own_pics/2-Speed_Limit-50.jpg "Traffic Sign 2"
[image6]: ./own_pics/13-Yield.jpg "Traffic Sign 3"
[image7]: ./own_pics/32-end_of_all_speed_limit.jpg "Traffic Sign 4"
[image8]: ./own_pics/25-roadwork.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/StyxXx1337/udacity-sdc-challenge3/blob/origin/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python standard library `len`-Function to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `[32, 32, 3]`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1a]

### Design and Test a Model Architecture

#### 1. Processing the Data for Model Input

##### 1.1 Create additional Data to ensure proper Training on all classes
I used 4 ways to create new data for the categories which had lower than `1500` images.
- Gaussian Blurring
- Salt and Pepper Noise
- Speckle Pattern
- Poisson Image

Here are examples of the noisy pictures:
![alt text][image3]

I created random noise of all the 4 possible noises, until the amount of pictures has reached the minimum limit.
![alt text][image1b]

##### 1.2 Gray scaling the pictures
As a next step, I decided to convert the images to grayscale because the color of the signs don't add additional information and add additional complexity for the neural network.

Here is an example of a traffic sign image before and after gray scaling.

![alt text][image2]

##### 1.3 Normalizing the pictures
As a last step, I normalized the image data because the neural network is working better with data that is around 0 because it speeds up the learning of the neural network.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| [1] Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 28x28x6 				|
| [2] Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x12 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 24x24x12 				|
| [3] Convolution 4x4     	| 1x1 stride, valid padding, outputs 21x21x24 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 21x21x24 				|
| [4] Convolution 4x4     	| 1x1 stride, valid padding, outputs 18x18x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 9x9x48 				|
| [5] Convolution 2x2     	| 1x1 stride, valid padding, outputs 8x8x96 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x96 				|
| Flatten |           input = 4x4x96 = 1536        |
| Fully connected		|       output = 768						|
| Fully connected		|       output = 384						|
| Logits		|       output = 43						|




#### 3. Training the Model

To train the model, I used a `Batch Size` of 64 and train for an `Epochs` of 25.
As `Optimizer` I use the Adam optimizer as it is one of the most popular optimizer and has a good speed and precision.
As hyperparameters I tried various learning rates between `0.001` to `0.0001` and ended up with `0.0004` since it seems to be working best for my 5-Convolutional LeNet Architecture.

#### 4. Solution approach

I started with the `LeNet` architecture and tried to tune the parameters as good as possible to achieve the target, but couldn't.
Then I tried adding additional layers with various `Filter sizes` `Pooling` and `dropout` combinations.
From what I could test the LeNet Architecture with 5 convolution layers worked the best, so I chose this.
Next I was fine tuning the `Filters` and `Pooling` Variables.

My final model results were:
* training set accuracy of `99,7%`
* validation set accuracy of `95.8%`
* test set accuracy of `94.5%`




### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set

Here are the results of the prediction:

| Image			                              |     Prediction        					    |
|:---------------------------------------:|:-----------------------------------:|
| No Passing     			                    | No Passing     				              |
| Speed limit (50km/h) 		                | _Speed limit (30km/h)_			        |
| Yield	      		                        | Yield				 				                |
| End of all speed and passing limits			| End of all speed and passing limits	|
| Road Work			                          | Road Work     						          |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The one miss prediction has as second highest prediction the correct one and also the predicted result is also very close to the actual prediction.

#### 3. Prediction of traffic signs taken from the internet.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

##### For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| .99     				      | No Passing   										              |
| .00         			    | No passing for vehicles over 3.5 metric tons  |
| .00					          | Priority road										              |
| .00	      			      | Yield	  					 				                    |
| .00				            | Vehicles over 3.5 metric tons prohibited  		|


##### For the second image the model missed the prediction of 50km/h and was predicting 30km/h speed limit. Given the size of the image and the resolution, the letters are difficult to classify, so I believe with a higher resolution the model could be able to give better predictions. What you can see in the predictions is, that the type was correctly classified `Speed Limit` but not the value.

| Probability         	|     Prediction	        					      |
|:---------------------:|:---------------------------------------:|
| .63         			    | Speed limit (30km/h)  									|
| .33     				      | Speed limit (50km/h) 										|
| .01					          | Speed limit (80km/h)                    |
| .01      			        | Speed limit (20km/h)	  					 			|
| .01				            | Speed limit (70km/h)	     							|

##### For the Third image the prediction has also a high level of confidence of 99%. The shape of the `Yield` sign is very unique, so that why the prediction is very sure about the rule.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------:|
| .99         			    | Yield   									        |
| .00     				      | Ahead only   										  |
| .00				            | Priority road											|
| .00	      			      | Turn right ahead				 				  |
| .00				            | No vehicles                       |

##### For the Fourth image is also very sure on the prediction of the `End of passing limits` as 98%.

| Probability         	|     Prediction	        					                 |
|:---------------------:|:--------------------------------------------------:|
| .98         			    | End of all speed and passing limits                |
| .01     				      | End of speed limit (80km/h)			                   |
| .00					          | End of no passing           			                 |
| .00	      			      | Priority road	  					 				                 |
| .00				            | End of no passing by vehicles over 3.5 metric tons |

##### For the Fifth image is also clearly predicted as the `road work`-sign with more than 99%.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------:|
| .99         			    | Road work  									      |
| .00     				      | Dangerous curve to the right 			|
| .00					          | Bumpy road									      |
| .00	      			      | General caution					 				  |
| .00				            | Keep right    							      |
