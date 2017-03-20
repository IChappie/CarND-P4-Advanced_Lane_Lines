#Advanced Lane Finding Project
---
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/test_undistort_output.png "Road Transformed"
[image3]: ./output_images/binary_combo_text_example.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image55]: ./output_images/color_fit_lines0.jpg "Fit Visual0"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/video.jpg "Output"
[image7]: ./output_images/binary_combo_example.jpg
[video1]: ./project_video.mp4 "Video"



###Camera Calibration
Before starting the implementation of the lane detection pipeline, the first thing that should be done is camera calibration. That would help us:

* undistort the images coming from camera, thus improving the quality of geometrical measurement.
* estimate the spatial resolution of pixels per meter in both *x* and *y* directions

The code for this step is contained in the second code cell of the IPython notebook located in "./submit/CarND_Advanced-Lane-Lines.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Undist image
One of the distortion-corrected images is as follows:
![alt text][image2]
####2. Creating thresholded binary image
The code for this step is contained in the forth code cell of the IPython notebook located in "./submit/CarND_Advanced-Lane-Lines.ipynb". 

The image gets undistorted first, then the perspective transformation is applied to the undistorted image. After that, the images are converted to HSV and HLS color space. L channel is used to track the bright regions; S channel is used to pick up lines under different color and contrast conditions, and r channel does well on the white lines. 

Also, some filtering is performed to reduce the noise,`cv2.medianBlur()` is used since it maintains the edges. `cv2.morphologyEx()` is used since it eliminates the noise, and smooth edges.

 Here's an example of my output for this step. 
![alt text][image3]


 Here's an example of my output for this step.  (note: it is the frame in project_video which I failed dectecting lines in last submittion)
![alt text][image7]

####3. Performing a perspective transform

The code for my perspective transfrom includes a function called `warpPerspective_step()`, which is contained in the fifth code cell of the IPython notebook located in "./submit/CarND_Advanced-Lane-Lines.ipynb". 

The `warpPerspective_step()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Locate the Lane Lines and Fit a Polynomial
After applying calibration, thresholding, and a perspective transform to a road image, we get a binary image where the lane lines stand out clearly. And now we need to locate the lane lines. The code for this step is contained in the sixth code cell of the IPython notebook.

I first take a **histogram** along all the columns in the *lower half* of the image like this:

	import numpy as np
	histogram = np.sum(img[img.shape[0]/2:,:], axis=0)

With this histogram I am adding up the pixel values along each column in the image. In my threshold binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. We can use that as a starting point for where to search for the lines. From that point, I can use a sliding window, place around the line centers, to find and follow the lines up to top of the frame.

The processing step is shown as follow:

![alt text][image55]

![alt text][image5]

####5. Calculate curvature and draw lane line on original image
The code for this step is contained in the eighth and ninth code cell of the IPython notebook located in "./submit/CarND_Advanced-Lane-Lines.ipynb". 

If lanes are found, the curvature is calculated using functions `cal_curvature()`. Since the two lines are present, for the coefficient of the polynomial the mean value is used. The *y* coordinate for which the polynomials are evaluated is the bottom of the image. After that, the lines are drawn on a warped image and then unwarped and added to the original image. The last thing is to print out the values of curvature and offset of the center of the car.Here is the result for discussed case:

![alt text][image6]

###Videos 
For the videos, the pipeline follows the basic pipeline applied to single images. 

I take some techniques to make results smooth. `cv2.mathchShape()` is used to compare the current polygon to one from a prior frame, which can reject a bad frame by allowing the pipeline to use the last good polygon instead. *Exponential smoothing* is used to averaging over two frames. I update the New frame as follows: New=alpha\*New+(1-alpha)\*Old.( I set alpha to 0.8). And in order to avoid the previous error, I used a random number *random_n* to recalculate.

Here is the link to video:[projec_output](./projec_output.mp4)

---

##Discussion

The biggest issue by far for me were sudden changes of light conditions. In those cases, the lines get either completely lost (going from bright to dark) or image gets filled with noise coming from the white spots. More advanced filtering and brightness equalization techniques have to be examined. 
 

