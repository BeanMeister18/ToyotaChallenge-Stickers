# import dependencies
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

from IPython.display import display, Javascript, Image

from base64 import b64decode, b64encode
import PIL
import io
import html
import time

from matplotlib import pyplot as plt
import cv2

def houghCircleDetector(img, og_image):
    
    img = cv2.medianBlur(img,3)
    img_edge = cv2.Canny(img,50,100)
    # plt.figure(figsize=(20,10))
    # #Note: matplotlib uses RGB format so had to convert BGR-to-RGB
    # plt.imshow(cv2.cvtColor(img_edge,cv2.COLOR_BGR2RGB))
    # plt.title('RGB Image',color='c')
    # plt.show()

    # circles = cv2.HoughCircles(img_edge,cv2.HOUGH_GRADIENT,1,minDist=20,param1=200,param2=70)
    circles = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, dp=1.3, minDist=20, param1=100, param2=45, minRadius=0, maxRadius=50)
    try:
      circles = np.uint16(np.around(circles))
      for val in circles[0,:]:
          cv2.circle(og_image,(val[0],val[1]),val[2],(255,0,0),2)
    except:
       pass

    # plt.figure(figsize=(20,10))
    # plt.subplot(121),plt.imshow(cv2.cvtColor(img_edge,cv2.COLOR_BGR2RGB)),plt.title('Input',color='c')
    # plt.subplot(122),plt.imshow(cv2.cvtColor(og_image,cv2.COLOR_BGR2RGB)),plt.title('Result',color='c')
    # plt.show()

#Shape Detection
# path_to_img = 'Metal_6.png'
def detectShapes(img):
    img = cv2.medianBlur(img,3)
    img_edge = cv2.Canny(img,50,100)
    # plt.figure(figsize=(20,10))
    # #Note: matplotlib uses RGB format so had to convert BGR-to-RGB
    # plt.imshow(cv2.cvtColor(img_edge,cv2.COLOR_BGR2RGB))
    # plt.title('RGB Image',color='c')
    # plt.show()
    # _,img_Otsubin = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours,_ = cv2.findContours(img_edge.copy(),1,2)

    # init the blank img
    blank = np.zeros(img.shape, dtype="uint8")
    blank[:] = -1
    for num,cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        # IF A LARGE CIRCLE
        if len(approx) > 10 and cv2.contourArea(cnt) > 1000:
            cv2.drawContours(blank,[cnt],-1,(0,255,0),2)
    cv2.imwrite("somethingproper.png", blank)

    # plt.figure(figsize=(200,100))
    # plt.subplot(131),plt.imshow(cv2.cvtColor(img_edge,cv2.COLOR_BGR2RGB)),plt.title('Input',color='c')
    # plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),plt.title('Result')
    # plt.show()
    return blank

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    outline = detectShapes(image)
    houghCircleDetector(outline, frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()