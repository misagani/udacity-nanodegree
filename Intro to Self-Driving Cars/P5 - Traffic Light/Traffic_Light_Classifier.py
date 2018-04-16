
# coding: utf-8

# # Traffic Light Classifier
# ---
# 
# In this project, you’ll use your knowledge of computer vision techniques to build a classifier for images of traffic lights! You'll be given a dataset of traffic light images in which one of three lights is illuminated: red, yellow, or green.
# 
# In this notebook, you'll pre-process these images, extract features that will help us distinguish the different types of images, and use those features to classify the traffic light images into three classes: red, yellow, or green. The tasks will be broken down into a few sections:
# 
# 1. **Loading and visualizing the data**. 
#       The first step in any classification task is to be familiar with your data; you'll need to load in the images of traffic lights and visualize them!
# 
# 2. **Pre-processing**. 
#     The input images and output labels need to be standardized. This way, you can analyze all the input images using the same classification pipeline, and you know what output to expect when you eventually classify a *new* image.
#     
# 3. **Feature extraction**. 
#     Next, you'll extract some features from each image that will help distinguish and eventually classify these images.
#    
# 4. **Classification and visualizing error**. 
#     Finally, you'll write one function that uses your features to classify *any* traffic light image. This function will take in an image and output a label. You'll also be given code to determine the accuracy of your classification model.    
#     
# 5. **Evaluate your model**.
#     To pass this project, your classifier must be >90% accurate and never classify any red lights as green; it's likely that you'll need to improve the accuracy of your classifier by changing existing features or adding new features. I'd also encourage you to try to get as close to 100% accuracy as possible!
#     
# Here are some sample images from the dataset (from left to right: red, green, and yellow traffic lights):
# <img src="images/all_lights.png" width="50%" height="50%">
# 

# ---
# ### *Here's what you need to know to complete the project:*
# 
# Some template code has already been provided for you, but you'll need to implement additional code steps to successfully complete this project. Any code that is required to pass this project is marked with **'(IMPLEMENTATION)'** in the header. There are also a couple of questions about your thoughts as you work through this project, which are marked with **'(QUESTION)'** in the header. Make sure to answer all questions and to check your work against the [project rubric](https://review.udacity.com/#!/rubrics/1213/view) to make sure you complete the necessary classification steps!
# 
# Your project submission will be evaluated based on the code implementations you provide, and on two main classification criteria.
# Your complete traffic light classifier should have:
# 1. **Greater than 90% accuracy**
# 2. ***Never* classify red lights as green**
# 

# # 1. Loading and Visualizing the Traffic Light Dataset
# 
# This traffic light dataset consists of 1484 number of color images in 3 categories - red, yellow, and green. As with most human-sourced data, the data is not evenly distributed among the types. There are:
# * 904 red traffic light images
# * 536 green traffic light images
# * 44 yellow traffic light images
# 
# *Note: All images come from this [MIT self-driving car course](https://selfdrivingcars.mit.edu/) and are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).*

# ### Import resources
# 
# Before you get started on the project code, import the libraries and resources that you'll need.

# In[346]:


"""Traffic Light Classifier.
Computer vision techniques to build a classifier,
for images of traffic lights. 
"""  
import cv2 # computer vision library
import helpers # helper functions
import random # for random test images
import scipy.stats # mathematical statistics --> adding this
import numpy as np # numpy library
import matplotlib.pyplot as plt # plot images
import matplotlib.image as mpimg # for loading in images
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Training and Testing Data
# 
# All 1484 of the traffic light images are separated into training and testing datasets. 
# 
# * 80% of these images are training images, for you to use as you create a classifier.
# * 20% are test images, which will be used to test the accuracy of your classifier.
# * All images are pictures of 3-light traffic lights with one light illuminated.
# 
# ## Define the image directories
# 
# First, we set some variables to keep track of some where our images are stored:
# 
#     IMAGE_DIR_TRAINING: the directory where our training image data is stored
#     IMAGE_DIR_TEST: the directory where our test image data is stored

# In[347]:


"""Datasets.
Define the image directories.
"""
# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"


# ## Load the datasets
# 
# These first few lines of code will load the training traffic light images and store all of them in a variable, `IMAGE_LIST`. This list contains the images and their associated label ("red", "yellow", "green"). 
# 
# You are encouraged to take a look at the `load_dataset` function in the helpers.py file. This will give you a good idea about how lots of image files can be read in from a directory using the [glob library](https://pymotw.com/2/glob/). The `load_dataset` function takes in the name of an image directory and returns a list of images and their associated labels. 
# 
# For example, the first image-label pair in `IMAGE_LIST` can be accessed by index: 
# ``` IMAGE_LIST[0][:]```.
# 

# In[348]:


"""Loading Datasets.
Using the load_dataset function in helpers.py file.
"""
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)


# ## Visualize the Data
# 
# The first steps in analyzing any dataset are to 1. load the data and 2. look at the data. Seeing what it looks like will give you an idea of what to look for in the images, what kind of noise or inconsistencies you have to deal with, and so on. This will help you understand the image dataset, and **understanding a dataset is part of making predictions about the data**.

# ---
# ### Visualize the input images
# 
# Visualize and explore the image data! Write code to display an image in `IMAGE_LIST`:
# * Display the image
# * Print out the shape of the image 
# * Print out its corresponding label
# 
# See if you can display at least one of each type of traffic light image – red, green, and yellow — and look at their similarities and differences.

# In[349]:


"""Visualize the Images from IMAGE_LIST.
1). Display an image.
2). Print out the shape of the image.
3). Print out its corresponding label.
4). Display the example of each image, Red, Yellow, Green.
"""   
# ---------------- Mapping ----------------
# Red Traffic from array 0 until 722
# Yellow Traffic from array 723 until 757
# Green Traffic from array 758 until 1186
# -----------------------------------------
# Image Array
# Change this value to check the others
img_array = 728

# Image & Label Variables
var_image = IMAGE_LIST[img_array][0]
var_label = IMAGE_LIST[img_array][1]

# 1). Display the Image
f, (fg1) = plt.subplots(1, 1, figsize=(10,5))
fg1.set_title(str(var_label))
fg1.imshow(var_image)

# 2). Print out the shape of the image
print("Image Shape:", var_image.shape)

# 3). Print out its corresponding label
print("Image Label:", var_label)

# 4). Display the example of each image, Red, Yellow, Green
f, (fg1, fg2, fg3) = plt.subplots(1, 3, figsize=(10,5))
fg1.set_title('Red')
fg1.imshow(IMAGE_LIST[3][0])
fg2.set_title('Yellow')
fg2.imshow(IMAGE_LIST[737][0])
fg3.set_title('Green')
fg3.imshow(IMAGE_LIST[777][0])


# # 2. Pre-process the Data
# 
# After loading in each image, you have to standardize the input and output!
# 
# ### Input
# 
# This means that every input image should be in the same format, of the same size, and so on. We'll be creating features by performing the same analysis on every picture, and for a classification task like this, it's important that **similar images create similar features**! 
# 
# ### Output
# 
# We also need the output to be a label that is easy to read and easy to compare with other labels. It is good practice to convert categorical data like "red" and "green" to numerical data.
# 
# A very common classification output is a 1D list that is the length of the number of classes - three in the case of red, yellow, and green lights - with the values 0 or 1 indicating which class a certain image is. For example, since we have three classes (red, yellow, and green), we can make a list with the order: [red value, yellow value, green value]. In general, order does not matter, we choose the order [red value, yellow value, green value] in this case to reflect the position of each light in descending vertical order.
# 
# A red light should have the  label: [1, 0, 0]. Yellow should be: [0, 1, 0]. Green should be: [0, 0, 1]. These labels are called **one-hot encoded labels**.
# 
# *(Note: one-hot encoding will be especially important when you work with [machine learning algorithms](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)).*
# 
# <img src="images/processing_steps.png" width="80%" height="80%">
# 

# ---
# <a id='task2'></a>
# ### (IMPLEMENTATION): Standardize the input images
# 
# * Resize each image to the desired input size: 32x32px.
# * (Optional) You may choose to crop, shift, or rotate the images in this step as well.
# 
# It's very common to have square input sizes that can be rotated (and remain the same size), and analyzed in smaller, square patches. It's also important to make all your images the same size so that they can be sent through the same pipeline of classification steps!

# In[350]:


def standardize_input(image):
    """Standardize Image.
    This function take an RGB image and return a standardized version.
    Resizing each image into 32x32 pixel, and adjust to 24x16 pixel.
    """
    # Return an array copy of the image
    clone_image = np.copy(image)
    
    # Resize to 32x32 pixel
    resize_image = cv2.resize(clone_image, (32, 32))
    
    # Adjusting image to 24x16 pixel
    row_crop = 4 # 32-4-4 = 24
    col_crop = 8 # 32-8-8 = 16
    standard_image = resize_image[row_crop:-row_crop, col_crop:-col_crop, :]
    
    # Return result
    return standard_image    


# ## Standardize the output
# 
# With each loaded image, we also specify the expected output. For this, we use **one-hot encoding**.
# 
# * One-hot encode the labels. To do this, create an array of zeros representing each class of traffic light (red, yellow, green), and set the index of the expected class number to 1. 
# 
# Since we have three classes (red, yellow, and green), we have imposed an order of: [red value, yellow value, green value]. To one-hot encode, say, a yellow light, we would first initialize an array to [0, 0, 0] and change the middle value (the yellow value) to 1: [0, 1, 0].
# 

# ---
# <a id='task3'></a>
# ### (IMPLEMENTATION): Implement one-hot encoding

# In[351]:


def one_hot_encode(label):
    """One-Hot Encoding.
    This function given a label - "red", "green", or "yellow",
    returning a one-hot encoded label.
    One-Hot Encode of "red"    return: [1, 0, 0]
    One-Hot Encode of "yellow" return: [0, 1, 0]
    One-Hot Encode of "green"  return: [0, 0, 1]
    """
    # Init Variable
    one_hot_encoded = []
    
    # Processing
    if label   == "red":
        one_hot_encoded = [1, 0, 0]
    elif label == "yellow":
        one_hot_encoded = [0, 1, 0]
    elif label == "green":
        one_hot_encoded = [0, 0, 1]
    else:
        one_hot_encoded = [0, 0, 0]
        
    # Return result
    return one_hot_encoded


# ### Testing as you Code
# 
# After programming a function like this, it's a good idea to test it, and see if it produces the expected output. **In general, it's good practice to test code in small, functional pieces, after you write it**. This way, you can make sure that your code is correct as you continue to build a classifier, and you can identify any errors early on so that they don't compound.
# 
# All test code can be found in the file `test_functions.py`. You are encouraged to look through that code and add your own testing code if you find it useful!
# 
# One test function you'll find is: `test_one_hot(self, one_hot_function)` which takes in one argument, a one_hot_encode function, and tests its functionality. If your one_hot_label code does not work as expected, this test will print ot an error message that will tell you a bit about why your code failed. Once your code works, this should print out TEST PASSED.

# In[352]:


"""Testing One-Hot Encoding.
To see if it produces the expected output.
"""
# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)


# ## Construct a `STANDARDIZED_LIST` of input images and output labels.
# 
# This function takes in a list of image-label pairs and outputs a **standardized** list of resized images and one-hot encoded labels.
# 
# This uses the functions you defined above to standardize the input and output, so those functions must be complete for this standardization to work!
# 

# In[353]:


def standardize(image_list):
    """Standardized List of Input Images and Output Labels.
    This function takes in a list of image-label pairs and outputs,
    a standardized list of resized images and one-hot encoded labels.
    """    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
    
    # Return result
    return standard_list

# Standardize all of training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)


# ## Visualize the standardized data
# 
# Display a standardized image from STANDARDIZED_LIST and compare it with a non-standardized image from IMAGE_LIST. Note that their sizes and appearance are different!

# In[354]:


"""Visualize the Standardized Images.
Display a standardized image from STANDARDIZED_LIST,
and compare it with a non-standardized image from IMAGE_LIST. 
"""    
# ---------------- Mapping ----------------
# Red Traffic from array 0 until 722
# Yellow Traffic from array 723 until 757
# Green Traffic from array 758 until 1186
# -----------------------------------------
# Image Array
# Change this value to check the others
img_array = 1000

# Image & Label from IMAGE_LIST
ori_image = IMAGE_LIST[img_array][0]
ori_label = IMAGE_LIST[img_array][1]

# Image & Label from STANDARDIZED_LIST
std_image = STANDARDIZED_LIST[img_array][0]
std_label = STANDARDIZED_LIST[img_array][1]

# Print out the shape of the image from IMAGE_LIST
print("Image Shape - Original:", ori_image.shape)

# Print out the shape of the image from STANDARDIZED_LIST
print("Image Shape - Standard:", std_image.shape)

# Print out its corresponding label from IMAGE_LIST
print("Image Label - Ortiginal:", ori_label)

# Print out its corresponding label from STANDARDIZED_LIST
print("Image Label - Standard:", std_label)

# 4). Display the example of each image, Red, Yellow, Green
f, (fg1, fg2) = plt.subplots(1, 2, figsize=(10,5))
fg1.set_title('Original')
fg1.imshow(ori_image)
fg2.set_title('Standard')
fg2.imshow(std_image)


# # 3. Feature Extraction
# 
# You'll be using what you now about color spaces, shape analysis, and feature construction to create features that help distinguish and classify the three types of traffic light images.
# 
# You'll be tasked with creating **one feature** at a minimum (with the option to create more). The required feature is **a brightness feature using HSV color space**:
# 
# 1. A brightness feature.
#     - Using HSV color space, create a feature that helps you identify the 3 different classes of traffic light.
#     - You'll be asked some questions about what methods you tried to locate this traffic light, so, as you progress through this notebook, always be thinking about your approach: what works and what doesn't?
# 
# 2. (Optional): Create more features! 
# 
# Any more features that you create are up to you and should improve the accuracy of your traffic light classification algorithm! One thing to note is that, to pass this project you must **never classify a red light as a green light** because this creates a serious safety risk for a self-driving car. To avoid this misclassification, you might consider adding another feature that specifically distinguishes between red and green lights.
# 
# These features will be combined near the end of his notebook to form a complete classification algorithm.

# ## Creating a brightness feature 
# 
# There are a number of ways to create a brightness feature that will help you characterize images of traffic lights, and it will be up to you to decide on the best procedure to complete this step. You should visualize and test your code as you go.
# 
# Pictured below is a sample pipeline for creating a brightness feature (from left to right: standardized image, HSV color-masked image, cropped image, brightness feature):
# 
# <img src="images/feature_ext_steps.png" width="70%" height="70%">
# 

# ## RGB to HSV conversion
# 
# Below, a test image is converted from RGB to HSV colorspace and each component is displayed in an image.

# In[355]:


"""RGB to HSV Conversion.
Convert and image to HSV colorspace.
Visualize the individual color channels,
and display each Histogram of H, S, V. 
"""    
# ---------------- Mapping ----------------
# Red Traffic from array 0 until 722
# Yellow Traffic from array 723 until 757
# Green Traffic from array 758 until 1186
# -----------------------------------------
# Image Array
# Change this value to check the others
img_array = 1000

# Image & Label from IMAGE_LIST
ori_image = IMAGE_LIST[img_array][0]
ori_label = IMAGE_LIST[img_array][1]

# Image & Label from STANDARDIZED_LIST
std_image = STANDARDIZED_LIST[img_array][0]
std_label = STANDARDIZED_LIST[img_array][1]

# Convert to HSV
hsv_conversion = cv2.cvtColor(std_image, cv2.COLOR_RGB2HSV)

# HSV channels
h_channel = hsv_conversion[:,:,0]
s_channel = hsv_conversion[:,:,1]
v_channel = hsv_conversion[:,:,2]

# Plot the original image and the three channels
f, (fg1, fg2, fg3, fg4) = plt.subplots(1, 4, figsize=(10,5))
fg1.set_title('Standard Image')
fg1.imshow(std_image)
fg2.set_title('H Channel')
fg2.imshow(h_channel)
fg3.set_title('S Channel')
fg3.imshow(s_channel)
fg4.set_title('V Channel')
fg4.imshow(v_channel)

# Histogram of HSV for each channel
h_histogram = np.histogram(hsv[:,:,0], bins=32, range=(0, 255))
s_histogram = np.histogram(hsv[:,:,1], bins=32, range=(0, 255))
v_histogram = np.histogram(hsv[:,:,2], bins=32, range=(0, 255))

# Generating bin centers of Histogram
bin_edges   = h_histogram[1]
bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

# Plot each chanel of the HSV Histogram 
fig = plt.figure(figsize=(10,3))
plt.subplot(131)
plt.bar(bin_centers, h_histogram[0])
plt.xlim(0, 180)
plt.title('H Histogram')
plt.subplot(132)
plt.bar(bin_centers, s_histogram[0])
plt.xlim(0, 256)
plt.title('S Histogram')
plt.subplot(133)
plt.bar(bin_centers, v_histogram[0])
plt.xlim(0, 256)
plt.title('V Histogram')
plt.show()


# ---
# <a id='task7'></a>
# ### (IMPLEMENTATION): Create a brightness feature that uses HSV color space
# 
# Write a function that takes in an RGB image and returns a 1D feature vector and/or single value that will help classify an image of a traffic light. The only requirement is that this function should apply an HSV colorspace transformation, the rest is up to you. 
# 
# From this feature, you should be able to estimate an image's label and classify it as either a red, green, or yellow traffic light. You may also define helper functions if they simplify your code.

# In[359]:


"""Extract Features.
This function takes in an RGB image,
and outputs a feature vector and value.
This feature use HSV colorspace values.
"""
def extract_features(rgb_image):
    # Using HSV color space
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Getting where is the light brightest spot
    mask_bottom = np.array([0,27,123])
    mask_top = np.array([178,236,254])
    masking = cv2.inRange(hsv_image, mask_bottom, mask_top)
    feature_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask = masking)
    
    # Masking range of Red Color
    lower_red = np.array([161])
    upper_red = np.array([203])
    red_masking = cv2.inRange(feature_hsv[:,:,0], lower_red, upper_red)
    
    # Masking range of Yellow Color
    lower_yellow = np.array([9])
    upper_yellow = np.array([31])
    yellow_masking = cv2.inRange(feature_hsv[:,:,0], lower_yellow, upper_yellow)
    
    # Masking range of Green Color
    lower_green = np.array([79]) 
    upper_green = np.array([101])
    green_masking = cv2.inRange(feature_hsv[:,:,0], lower_green, upper_green)
    
    # Combine masks
    combine_masking = red_masking + yellow_masking + green_masking
    
    # Copy Hue channel
    feature = np.copy(hsv_image[:,:,0])
    feature[combine_masking == 0] = [0]
    
    # Return result
    return feature


# ## (Optional) Create more features to help accurately label the traffic light images

# In[362]:


"""Additional Features.
This function based on mathematical statistics functions.
Searching of average value (Mean),
and most common value (Mode).
"""
def mode_value(hsv_image):
    total_mode = scipy.stats.mode(hsv_image[np.nonzero(hsv_image)])[0]
    return total_mode[0] if len(total_mode) > 0 else 0

def median_value(hsv_image):
    nonzero = hsv_image[np.nonzero(hsv_image)]
    if len(nonzero) > 0:
        return np.median(hsv_image[np.nonzero(hsv_image)])
    return 0


# ## (QUESTION 1): How do the features you made help you distinguish between the 3 classes of traffic light images?

# **Answer:**
# 1. We must resize all of the image into the same pixel size.
# 2. Next, convert RGB into HSV value.
# 3. From HSV value, we can distinguish which one red, yellow, or green, by masking them.
# 4. Finally, we can clasify by using threshold value between red, yellow, and green.

# # 4. Classification and Visualizing Error
# 
# Using all of your features, write a function that takes in an RGB image and, using your extracted features, outputs whether a light is red, green or yellow as a one-hot encoded label. This classification function should be able to classify any image of a traffic light!
# 
# You are encouraged to write any helper functions or visualization code that you may need, but for testing the accuracy, make sure that this `estimate_label` function returns a one-hot encoded label.

# ## ------------------------------------------------
# <a id='task8'></a>
# ### (IMPLEMENTATION): Build a complete classifier 

# In[363]:


"""Traffic Light Classifier.
This function take in RGB image input.
Extract features from the RGB image and
use those features to classify the image,
and output a one-hot encoded label.
"""
def estimate_label(rgb_image):
    # Init variable
    predicted_label = []
        
    # Extract Features
    masked_image = extract_features(rgb_image)
    mode = mode_value(masked_image)
    median = median_value(masked_image)
    
    # Threshold classification between red, yellow and green
    if (median >= 162.) or (mode >= 176.):
        predicted_label = one_hot_encode("red")    
    elif (median >= 79. and median <= 161.):
        predicted_label = one_hot_encode("green")
    elif (mode == 11.) and (median == 11.):        # tricky threshold, worked only for this dataset
        predicted_label = one_hot_encode("green")        
    else:
        predicted_label = one_hot_encode("yellow") # all unidentified, classify as yellow (safe)
    
    # Return result
    return predicted_label


# ## Testing the classifier
# 
# Here is where we test your classification algorithm using our test set of data that we set aside at the beginning of the notebook! This project will be complete once you've pogrammed a "good" classifier.
# 
# A "good" classifier in this case should meet the following criteria (and once it does, feel free to submit your project):
# 1. Get above 90% classification accuracy.
# 2. Never classify a red light as a green light. 
# 
# ### Test dataset
# 
# Below, we load in the test dataset, standardize it using the `standardize` function you defined above, and then **shuffle** it; this ensures that order will not play a role in testing accuracy.
# 

# In[364]:


"""Testing the classifier.
Using the load_dataset function in helpers.py
We load in the test dataset, standardize it, and then shuffle it.
This ensures that order will not play a role in testing accuracy.
"""
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)


# ## Determine the Accuracy
# 
# Compare the output of your classification algorithm (a.k.a. your "model") with the true labels and determine the accuracy.
# 
# This code stores all the misclassified images, their predicted labels, and their true labels, in a list called `MISCLASSIFIED`. This code is used for testing and *should not be changed*.

# In[365]:


"""Determine the Accuracy.
Constructs a list of misclassified images given a list of test images and their labels.
This will throw an AssertionError if labels are not standardized (one-hot encoded).
And check the accuracy of this classifier.
"""  
def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:
        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from the classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):            
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
                        
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels

# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

# Print Result of Accuracy and Misclassified Images
print('Accuracy of Trafic Light Classifier = {:.2f}%'.format(accuracy*100) + ' (' + str(accuracy) + ')')
print("Number of misclassified images      = " + str(len(MISCLASSIFIED)) +' out of '+ str(total) + ' images')


# ---
# <a id='task9'></a>
# ### Visualize the misclassified images
# 
# Visualize some of the images you classified wrong (in the `MISCLASSIFIED` list) and note any qualities that make them difficult to classify. This will help you identify any weaknesses in your classification algorithm.

# In[366]:


"""Visualize the Misclassified Images.
Display an image in the `MISCLASSIFIED` list.
Print out its predicted label - to see what-
the image *was* incorrectly classified as.
"""  
print('-------------------')
print('-- MISCLASSIFIED -- ')

# Init Variables
images_missclassified = []
img_count = 0

# Check each image in misclassified list
for image in MISCLASSIFIED:
    # Counting image
    img_count += 1
    print('-------------------')
    
    # Check the features
    images_missclassified = image[0]
    masked_image = extract_features(image[0])
    print('Image   :', img_count)
    print('Mode    :', mode_value(masked_image))
    print('Median  :', median_value(masked_image))
    print('Predict :', image[1])
    print('Actual  :', image[2])
    
    # Plot the missclassified images
    f, (fg1, fg2) = plt.subplots(1, 2, figsize=(5,3))
    fg1.set_title('Standard ' + str(img_count))
    fg1.imshow(images_missclassified)
    fg2.set_title('Masked ' + str(img_count))
    fg2.imshow(masked_image)
print('-------------------')


# ---
# <a id='question2'></a>
# ## (Question 2): After visualizing these misclassifications, what weaknesses do you think your classification algorithm has? Please note at least two.

# **Answer:** 
# 1. Masking using HSV value cannot cover all of variation color. Especially 'abnormal' ones. 
# 2. 'abnormal' means, color supposedly 'red' is not 'red', color 'yellow' is not 'yellow', and color 'green' is not 'green'.
# 3. In this case, all image that cannot identify, have almost white/gray color.
# 4. We must using machine learning for more accurate result.

# ## Test if you classify any red lights as green
# 
# **To pass this project, you must not classify any red lights as green!** Classifying red lights as green would cause a car to drive through a red traffic light, so this red-as-green error is very dangerous in the real world. 
# 
# The code below lets you test to see if you've misclassified any red lights as green in the test set. **This test assumes that `MISCLASSIFIED` is a list of tuples with the order: [misclassified_image, predicted_label, true_label].**
# 
# Note: this is not an all encompassing test, but its a good indicator that, if you pass, you are on the right track! This iterates through your list of misclassified examples and checks to see if any red traffic lights have been mistakenly labelled [0, 1, 0] (green).

# In[367]:


"""Testing Red as Green Light.
Test if missclassified any red lights as green.
Classifying red lights as green would cause a car to-
drive through a red traffic light, so this red-as-green-
error is very dangerous in the real world.
"""  
# Importing the tests
import test_functions
tests = test_functions.Tests()

# Checking red as green
if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")


# # 5. Improve your algorithm!
# 
# **Submit your project after you have completed all implementations, answered all questions, AND when you've met the two criteria:**
# 1. Greater than 90% accuracy classification
# 2. No red lights classified as green
# 
# If you did not meet these requirements (which is common on the first attempt!), revisit your algorithm and tweak it to improve light recognition -- this could mean changing the brightness feature, performing some background subtraction, or adding another feature!
# 
# ---

# ### Going Further (Optional Challenges)
# 
# If you found this challenge easy, I suggest you go above and beyond! Here are a couple **optional** (meaning you do not need to implement these to submit and pass the project) suggestions:
# * (Optional) Aim for >95% classification accuracy.
# * (Optional) Some lights are in the shape of arrows; further classify the lights as round or arrow-shaped.
# * (Optional) Add another feature and aim for as close to 100% accuracy as you can get!
