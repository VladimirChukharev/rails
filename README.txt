 Files:

-README.txt
This file. How unnecessarily recursive.
 
-example.py
Python script template for the training script. Contains hints for the communication with Valohai interface as well as some related Keras examples.

-example.yaml
Example of a .yaml file that can be used to once more communicate with the Valohai interface.
 
-test.tar
Contains 300 (pre-augmented) images and the corresponding label files (test_labels_left.txt, test_labels_right.txt) that can be used as testing data for the neural network training process.

-train.tar
Contains 1500 (pre-augmented) images and the corresponding label files (train_labels_left.txt, right_labels_right.txt) that can be used as training data for the neural network training process.

-rails.avi
Contains the video from which the frames were extracted and augmented. Can be used for validation and - if the task is successful - demonstration purposes.


 Formatting of label files:

-"Left" and "right" in file names are defined from the perspective of the locomotive-thing. In images, left is essentially top and right is bottom rail.

-The labels are saved as ascii files to folders that correspond to the images in that folder. 

-Rows represent corresponding images (i.e., 20th row contains labels for the image 20.png).

-Rows contain (x,y) -pixel coordinates for five points for the corresponding rail and image. 

-The coordinates are formatted as [x1], [y1], [x2], [y2], ..., [x5],[y5], where
the number refers to the point number (1-5)
x refers to the pixel column
y refers to the pixel row
 
