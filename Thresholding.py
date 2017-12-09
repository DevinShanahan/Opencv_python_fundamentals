
# coding: utf-8

# Thresholding is the simplest method of image segmentation.
# More details please read;
# http://homes.di.unimi.it/ferrari/ImgProc2011_12/EI2011_12_16_segmentation_double.pdf

# In[1]:


import cv2
import numpy as np


# In[2]:


#only for TrackBar process
def nothing(x):
    pass


# In[24]:


image_org = cv2.imread('data/chrome.jpg')
image = cv2.cvtColor(image_org,cv2.COLOR_BGR2GRAY)


# In[27]:


maxval = 255
thresh=0
type_thresh = 2
cv2.namedWindow("Adjust",cv2.WINDOW_AUTOSIZE); #Threshold settings window
cv2.createTrackbar("Thresh", "Adjust", thresh, 200, nothing);
cv2.createTrackbar("Max", "Adjust", maxval, 255, nothing);

#Threshold methods correspond integer numbers in OpenCV Library,(binary threshold,otsu threshold etc)
#And threshold methods summable with each other like; cv2.BINARY_THRESH + cv2.OTSU_THRESH or 1 + 4
cv2.createTrackbar("Type", "Adjust", type_thresh, 4, nothing);

Threshold = np.zeros(image.shape, np.uint8)
i=1
# Infinite loop until we hit the escape key on keyboard
while 1:
    thresh = cv2.getTrackbarPos('Thresh', 'Adjust')
    maxval = cv2.getTrackbarPos('Max', 'Adjust')
    type_thresh = cv2.getTrackbarPos('Type', 'Adjust')
    retval,Threshold = cv2.threshold(image,thresh,maxval,type_thresh)
    # display images
    cv2.imshow('Adjust', Threshold)

    #cv2.imshow('Original', image_org)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        cv2.imwrite("outputs/threshold_ex{}.jpg".format(i),Threshold)
        i += 1
    elif k == 27:   # hit escape to quit
        break

cv2.destroyAllWindows()

# More info about thresholding and OTSU Method (widely used method as adaptive thresholding) please read,
#
# http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Global_Thresholding_Adaptive_Thresholding_Otsus_Binarization_Segmentations.php
#
# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold
#
# https://docs.opencv.org/2.4/doc/tutorials/imgproc/threshold/threshold.html
#
# https://www.learnopencv.com/opencv-threshold-python-cpp/
#
# https://docs.opencv.org/3.3.1/d7/d4d/tutorial_py_thresholding.html
