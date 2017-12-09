
# coding: utf-8

# FACE PROCESSING USING HAARCASCADES
#
#
# Please read short and important document from OpenCV website
# https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html

# In[1]:


import numpy as np
import cv2


# You can find haarcascade files in local (C:\your_opencv_folder\opencv\sources\data\haarcascades)

# In[2]:


#haarcascade file paths
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')


# In[5]:


#FACE & EYE DETECTION FROM WEBCAM
cap=cv2.VideoCapture(0)# Capturing video from the camera
# Step 3: Displaying it to the user
while(cap.isOpened()):
    ret, frame=cap.read()
    #Step 4: Perform Color Inversion on the frames
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)#This will convert BGR to RGB

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #prevent GUI crash
    cv2.startWindowThread()
    #Step 5:- Displaying it to the user
    cv2.imshow("Face and Eye detection",frame)
    #Step 6: Checking for user's interaction  with keyboard
    if cv2.waitKey(25) & 0xFF==ord('a'):#Exit whenever user press the a key on the keyboard
        cv2.imwrite("outputs/face_detection.jpg",frame)
        break
#Step 7: Releasing everything after the job is done
cap.release()
cv2.destroyAllWindows()


# In[3]:


#SMILE DETECTION FROM WEBCAM
cap2=cv2.VideoCapture(0)# Capturing video from the camera
# Step 3: Displaying it to the user
while(cap2.isOpened()):
    ret2, frame2=cap2.read()
    #Step 4: Perform Color Inversion on the frames
    gray2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)#This will convert BGR to RGB

    faces = face_cascade.detectMultiScale(gray2, 1.3, 5)
    for (x,y,w,h) in faces:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray2[y:y+h, x:x+w]
        roi_color = frame2[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray,1.7,22,minSize=(25,25))
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    #prevent GUI crash
    cv2.startWindowThread()
    #Step 5:- Displaying it to the user
    cv2.imshow("Smile Detection",frame2)
    #Step 6: Checking for user's interaction  with keyboard
    if cv2.waitKey(25) & 0xFF==ord('a'):#Exit whenever user press the a key on the keyboard
        cv2.imwrite("outputs/smile_detection.jpg",frame2)
        break
#Step 7: Releasing everything after the job is done
cap2.release()
cv2.destroyAllWindows()
