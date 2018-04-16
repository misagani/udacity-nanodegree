# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

This project will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.

To meet specifications in the project, take a look at the requirements in the [project rubric](https://review.udacity.com/#!/rubrics/322/view)

Goal
---

The goals of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on my work in a this written report

Reflection
---

Reflection describes the current pipeline, identifies its potential shortcomings and suggests possible improvements.

[//]: # (Image References)

[pipe1]: ./pipelines/solidWhiteRight_1_grayscale.jpg "Grayscale"
[pipe2]: ./pipelines/solidWhiteRight_2_blurred.jpg "Gaussian Blur"
[pipe3]: ./pipelines/solidWhiteRight_3_col_sel.jpg "Color Selection"
[pipe4]: ./pipelines/solidWhiteRight_4_masked.jpg "Region of Interest"
[pipe5]: ./pipelines/solidWhiteRight_5_canny.jpg "Canny Edge"
[pipe6]: ./pipelines/solidWhiteRight_6_houghed.jpg "Hough Transform"
[pipe7]: ./pipelines/solidWhiteRight_7_lines.jpg "Draw Lines"
[pipe8]: ./pipelines/solidWhiteRight_8_final.jpg "Weighted Image"

--- 

## My Pipelines

My pipeline consisted of 8 steps.
* Grayscale

![alt text][pipe1]

* Gaussian Blur

![alt text][pipe2]

* Color Selection

![alt text][pipe3]

* Region of Interest

![alt text][pipe4]

* Canny Edge

![alt text][pipe5]

* Hough Transform

![alt text][pipe6]

* Draw Lines

![alt text][pipe7]

* Weighted Image

![alt text][pipe8]

## Potential Shortcomings

### 1. Identify potential shortcomings with your current pipeline

One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


## Possible Improvements

### 1. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...

### 2. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...
### 3. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
