# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2a]: ./output_images/HOG_example_car.png
[image2b]: ./output_images/HOG_example_notcar.png
[image3a]: ./output_images/sliding_window.png
[image3b]: ./output_images/sliding_window_car.png
[image4]: ./output_images/process_label_map.png
[image5_T01]: ./output_images/heat_T01.png
[image5_T10]: ./output_images/heat_T10.png


## Rubric Points
#### Here I will consider the [rubric](https://review.udacity.com/#!/rubrics/513/view)  points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

*The code for this step is contained in the Section 1,2 (code cell [1],[2] of `P5.ipynb`).*

I download the training images from the follow links:
  * `vehicle`    :https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
  * `non-vehicle`:https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip

Then, I started by reading in all the `vehicle`(`car`) and `non-vehicle`(`notcar`) images.  Here is an example of one of each of the `car` and `notcar` classes (code cell [1] of `P5.ipynb`):

![alt text][image1]

I used the `get_hog_features` function for extracting HOG features of images (code cell [2] of `P5.ipynb`).  
I grabbed 8 images from each of the two classes and displayed them to get a feel for what the output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=8` and `cells_per_block=8`:

#### The car images
![alt text][image2a]

#### The notcar images
![alt text][image2b]

#### 2. Explain how you settled on your final choice of HOG parameters.

*The code for this step is contained in the Section 3 (code cell [4] of `P5.ipynb`).*

First, I considered the combinations of the Color_space and HOG parameters(Orientation, Pixels_per_cell, Pixels_per_cell). Then I trained the support vector machine (SVM) classifiers of the combinations based on the linear SVM. The result of the each training was saved as a pickle file for later using.

Here is the accuracy of the combinations.

| Combination | Color_space | Orientation | Pixels_per_cell | Cells_per_block | Accuracy    |
|:-----------:|:-----------:|:-----------:|:---------------:|:---------------:|:-----------:|
| A0          | RGB         | 9           | 8               | 2               | 0.9665      |
| A1          | HSV         | 9           | 8               | 2               | 0.9854      |
| A2          | HLS         | 9           | 8               | 2               | 0.9780      |
| A3          | YCrCb       | 9           | 8               | 2               | 0.9800      |
| A4          | YCrCb       | 5           | 8               | 2               | 0.9811      |
| A5          | YCrCb       | 9           | 8               | 2               | 0.9859      |
| A6          | YCrCb       | 13          | 8               | 2               | 0.9856      |
| A7          | YCrCb       | 9           | 6               | 2               | 0.9817      |
| A8          | YCrCb       | 9           | 8               | 2               | 0.9854      |
| A9          | YCrCb       | 9           | 16              | 2               | 0.9772      |


As a result, I settled on my final choice of the combination A5, that had the highest accuracy for testing datasets.
Here is the final combination(A5):
* Color_space     : YCrCb
* Orientation     : 9
* Pixels_per_cell : 8
* Cells_per_block : 2


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

*The code for this step is contained in the Section 3 (code cell [4] of `P5.ipynb` and functions of the `lesson_functions.py`).*

I trained a linear SVM (`train_Classifier` function of the code cell [4] of `P5.ipynb`) with the above combinations of the Color-space and HOG parameters by the following steps.
  * Extract the HOG features of the `car` and `notcar` images using the `extract_features` method.
  * Normalize the features by the `StandardScaler` method, and then split the results into 80% for training and 20% for testing under randomization.
  * Apply the `LinearSVC()` method for training.

 The test accuracy of the above final combination is 98.59%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

*The code for this step is contained in the Section 4 (code cell [5],[6] of `P5.ipynb`).*

I used the method basing on the `find_cars` function of the `Hog Sub-sampling Window Search` class (see in the `find_cars` function of the `lesson_functions.py`). The method includes the HOG feature extraction, sliding window search and classifier prediction. The sliding window search was done as follows:
  * The HOG features were extracted for the selected portion of images with a scale. Then, the HOG features were sub-sampled according to the size of the window under sliding window search, then feed them to the classifier. The the classifier prediction on the features for each window was done and the positives determine the car detection.

I explored several configurations (sizes, positions) of the sliding window.  The window sizes was in the scales of 0.8x, 1.0x, 1.5x , 1.8x, 2.0x. Instead of overlap, I defined the sliding window by the cells_per_step of 2 (number of cells per step) which used in the `find_cars` function. Here is the sliding windows in each scale for a test image returned by the `find_cars` function.

![alt text][image3a]

Here is the positive windows for the car detection in each scale, respectively.

![alt text][image3b]

I also explored the window sizes in scale less than 1x, but most of them had many false positive windows. Therefore, I finally used the follow scales with the positions limited in the Y ranges in the implementation.

| Scale       | ystart      | ystop      |
|:-----------:|:-----------:|:----------:|
| 0.8         | 380         | 500        |
| 1.0         | 360         | 500        |
| 1.5         | 360         | 560        |
| 1.8         | 400         | 600        |
| 2.0         | 440         | 700        |


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

*The code for this step is contained in the Section 5 (code cell [7],[8] of `P5.ipynb`).*

Ultimately, I searched on the above condition, five scales using YCrCb 3-channel HOG features ("ALL") plus spatially binned color and histograms of color in the feature vector (code cell[7],[8] of `P5.ipynb`, using the `find_cars_scales` function).
The positive windows for car detections were transformed and combined into a heatmap using the `add_heat` function. The region of higher levels of heat assigns the more overlapping of the positive windows, meaning the more reliable car detections.

A threshold for the heatmap was finally applied to remove the false positive region.
Then, the label method (`scipy.ndimage.measurements.label()`) is used to identify individual blobs in the heatmap, assumed each blob corresponded to a vehicle.
Here are the results through pipeline for all images in `test_images`(threshold is 3):

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a link to my video results:
  * [project_video_output_10sumX5](./project_video_output_10sumX5)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

*The code for this step is contained in the Section 6 of `P5.ipynb`.*

For the video, I implemented a pipeline for an integrated heatmap of the subsequent frames to reject the false positives and to get more reliable car detections. Here are the steps of implementation.
  * Only interested region is considered(remove the opposite lane region that doesn't really matter here with limitation of X range by xstart_stop=(672, image.shape[1]), and limit the Y ranges as shown above)
  * The positions of positive detections is estimated in the current frame using the above pipeline (using the `find_cars_scales` function)
  * A temporary heatmap is created from the positive detections and stored into a heatmap lists of the latest subsequent frames (number of the frames for integration `maxlen` is 10).
  * The heatmap list are integrated to a new heatmap using the `np.sum` for the current frame
  * A threshold method is applied for the heatmap to identify vehicle positions with the threshold value of (`maxlen`* `threshold`)
  * The label method is used to identify individual blobs in the heatmap

Here's an example result showing the heatmap from a series of frames of video, the result of and the bounding boxes then overlaid on the last frame of video:

#### Here is the positive windows, heatmap, and bounding boxes from the integration of latest 10 subsequent frames (`maxlen`=10, `threshold` = 4):
![alt text][image5_T10]

#### Here are the `test_video` results that show how it does change by the integration number `maxlen`. Noted that, `maxlen`=0 is the result without any integration of previous frames, and is found to be unstable.
  * maxlen:  1 ([test_video_output_1sumX4.mp4](./test_video_output_1sumX4.mp4))
  * maxlen: 10 ([test_video_output_10sumX4.mp4](./test_video_output_10sumX4.mp4))

#### Here are the `project_video` results that show how it does change by threshold values:
  * Threshold: 4 ([project_video_output_10sumX4](./project_video_output_10sumX4))
  * Threshold: 5 ([project_video_output_10sumX5](./project_video_output_10sumX5))
  * Threshold: 6 ([project_video_output_10sumX6](./project_video_output_10sumX6))
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Many false positive windows are found from classifier prediction. In order to remove the false positive windows, I used a large threshold temporarily. That limits the car detections. Some other classifier methods or neural network might be considered to provide a better results for the classification for vehicles in future study.
