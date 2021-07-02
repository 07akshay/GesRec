#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from collections import deque
from sklearn.preprocessing import normalize
import os
import subprocess


# In[2]:


lower_blue = np.array([100,70,70])
upper_blue = np.array([140,255,255])
kernel = np.ones((5,5),np.uint8)
k2 = np.ones((3,3),np.uint8)
alphabet = np.zeros((200, 200, 3), dtype=np.uint8)
model = tf.keras.models.load_model('Downloads/model.h5')


# In[ ]:


flag=0
points=[]
blackboard = np.zeros((340,340,3), dtype=np.uint8)
cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret,frame = cap.read()
    if ret==True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        res = cv2.bitwise_and(frame,frame, mask= mask)
        edged = cv2.Canny(res, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours)>0:
            contour = contours[0]
            ((x,y),radius) = cv2.minEnclosingCircle(contour)
            M = cv2.moments(contour)
            center = (int(M['m10'] / (M['m00']+1)), int(M['m01'] / (M['m00']+1)))
            points.append(center)
        elif len(contours)==0:
            if len(points)!=0:
                bb = cv2.cvtColor(blackboard,cv2.COLOR_BGR2GRAY)
                bkb = cv2.resize(bb, (28,28),interpolation = cv2.INTER_NEAREST)
                bkb = cv2.filter2D(bkb,-1,k2)
                bkb = bkb/np.max(bkb)
                bb = bkb.reshape(1,28,28,1)
                ans = model.predict(bb)
                if flag==0:
                    num = np.argmax(ans[0])
                    print(num)
                    if num==1:
                        subprocess.call('firefox')
                    if num==2:
                        subprocess.call('subl')
                    if num==3:
                        subprocess.call('google-chrome-stable')
                    if num==4:
                        subprocess.call('libreoffice --writer')
                    if num==5:
                        subprocess.call('telegram-desktop')
                    if num==6:
                        subprocess.call('thunderbird')
                    flag=1
                cv2.imshow('bb',bkb)
                
        if cv2.waitKey(1) & 0xFF == ord('$'):
            break
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(res, points[i - 1], points[i], (0, 0, 0), 2)
            cv2.line(blackboard, points[i - 1], points[i], (255, 255, 255), 8)
        cv2.imshow('Frame',frame)
        cv2.imshow('blackboard',blackboard)

    else:
        break
cap.release()
cv2.destroyAllWindows()




# In[ ]:




