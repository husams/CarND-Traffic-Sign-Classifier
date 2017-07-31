# Traffic Sign Recognition

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/visualization.png "Visualization"
[image2]: ./images/grayscale.png "Grayscaling"
[image3]: ./images/random_rotation.png "Random Noise"
[image4]: ./images/image1.png "Traffic Sign 1"
[image5]: ./images/image2.png "Traffic Sign 2"
[image6]: ./images/image3.png "Traffic Sign 3"
[image7]: ./images/image4.png "Traffic Sign 4"
[image8]: ./images/image5.png "Traffic Sign 5"
[image9]: ./images/probabilities.png "Top Five probabilities"
[image10]: ./images/layer1.png "Layer 1"
[image11]: ./images/layer2.png "Layer 2"
[image12]: ./images/layer3.png "Layer 3"
[image13]: ./images/layer4.png "Layer 4"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

## Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

## Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distbuted across 43 classes

![alt text][image1]

We can see from the chart above that we have skewed data, which means that some the labels will have more impact on the traning then other

## Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because it allow the allow the nwtowrk to learn filter which detect the edges, however that could lead to a problem for some of the signs with same shape but has different color. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it helps with gradient.

I decided to generate additional data because skewed data which most likly havwe inpact on the traning

To add more data to the the data set, I used rotation with range of random rangles (-15,15) because the new genrated image which belong to the correct class/label.   

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the fnew images is rotated little bit the left. 


## Final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x128	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128 			    	|
| Fully connected		| 1024 Units     								|
| RELU					|												|
| Fully connected		| 43 Units      								|
| Softmax				|           									|
 


## 3. Training

To train the model, I used Adm optimizer with 10 Epochs, 32 sample batch size, 0.00001 learning rate and 0.3 Keep probability for the dropout. 


My final model results were:
* training set accuracy of 0.99999
* validation set accuracy of 0.97937
* test set accuracy of 0.960

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture was LeNet-5 since it work well with grayscale hand written digits, so it should be able to handle grayscale trafic signs images.
* What were some problems with the initial architecture?
The problem with initial architecture was the maximum validation accuracy I could get is 86-87% with just the provaided data and around 92% with augmented data.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I first increased number of layers by adding convolution layer and max pooling which result increase of  training accuracy to 98% and validation was round  95$ which suggested overfitting so I added dropout, as  result the  validation accuracy increased to around 96%.
next I wanted to see if I can increase the validation accuracy so I reduced number of pooling layer to preserve some of the information and as result validation accuracy increased.  
* Which parameters were tuned? How were they adjusted and why?
I tunned number of batch and I found the best result with 32 samples per batch, I also tunned keep probability and I found 0.2-0.3 to be good number.
I found with current model 10 Epochs to be a good numner, anything above that resultted with noise.
As for the learning rate I start with 0.01, 0.001, 0.0001 and I found 0.0001 to be good fit.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
1. Samll learning rate prevented the model for overshooting
2. Dropout with 0.2-0.3 keep probability helped with overfiting
3. Small batch size increased accuracy (Looks like avraging over small batch of data yield better result).
4. Genersting fake samples  to balance the data seems to have good impact.
5. Increase number of  Convolution layers to 4 with larger number of filters (16,32,64,128) and only 2 max pooling helped with accuracy, my guess is preserving most of the image sstrcture helped the netwrk to learn more about it. 


## Test a Model on New Images

# Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

THe model was able to ptoduct 5 ot 5 images correctly.


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      	| Priority road									| 
| Turn right ahead     	| Turn right ahead 								|
| No Entry				| No Entry										|
| Speed limit (70km/h)	| Speed limit (70km/h)			 				|
| Speed limit (30km/h)	| Speed limit (30km/h)      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96%

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Priority road sign (probability of 1.0), and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					        | 
|:---------------------:|:-----------------------------------------------------:| 
| 1.00000000e+00        | Priority road  								        | 
| 6.56601329e-18     	| Roundabout mandatory 							        |
| 2.43700147e-18		| End of no passing by vehicles over 3.5 metric tons	|
| 2.02630381e-18	    | Right-of-way at the next intersection			        |
| 5.71072312e-19		| End of no passing    							        |


For the second image the model is relatively sure that this is a Turn right ahead sign (probability of 1.0), and the image does contain a Turn right ahead. The top five soft max probabilities were

| Probability         	|     Prediction	        					        | 
|:---------------------:|:-----------------------------------------------------:| 
| 1.00000000e+00        | Turn right ahead  			    			        | 
| 7.35587480e-10     	| Ahead only 							                |
| 2.47404347e-10		| Go straight or left	                                |
| 6.47218529e-11	    | Road narrows on the right			                    |
| 8.11815112e-12		| Traffic signals                                       |


For the third image the model is relatively sure that this is a No Entry sign (probability of 1.0), and the image does contain a No Entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					        | 
|:---------------------:|:-----------------------------------------------------:| 
| 1.00000000e+00        | No Entry  			    			                | 
| 1.45890844e-09     	| Stop 							                        |
| 7.80807981e-14		| Keep right	                                        |
| 7.80807981e-14	    | Turn left ahead			                            |
| 1.10506798e-15		| Turn right ahead                                      |

For the forth image the model is relatively sure that this is a Speed limit (70km/h) sign (probability of 0.99999881), and the image does contain aSpeed limit (70km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					        | 
|:---------------------:|:-----------------------------------------------------:| 
| 9.99999881e-01        | Speed limit (70km/h)	    			                | 
| 1.30532356e-07     	| Speed limit (30km/h)							        |
| 1.05665277e-09		| Speed limit (20km/h)	                                |
| 3.81812637e-11	    | Speed limit (80km/h)			                        |
| 2.63529216e-11		| Traffic signals                                       |

For the fifth image the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 1.0), and the image does contain aSpeed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					        | 
|:---------------------:|:-----------------------------------------------------:| 
| 1.00000000e+00        | Speed limit (70km/h)	    			                | 
| 2.19309619e-08     	| Speed limit (80km/h)							        |
| 2.08303264e-09		| Speed limit (50km/h)	                                |
| 5.07546609e-11	    | Speed limit (20km/h)			                        |
| 1.52807156e-13		| Speed limit (70km/h)                                  |

And here some Visualization for the prediction

![alt text][image9]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

We can see from the images below the network manage to learn filter which allow it to react to different edges, however it seems that filter 40 in layer 3 didn't activate which could suggets that we have dead neuron.

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]




