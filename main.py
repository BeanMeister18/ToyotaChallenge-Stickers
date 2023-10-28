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
    og_image = cv2.imread(og_image, 0)
    
    img = cv2.medianBlur(img,3)
    img_edge = cv2.Canny(img,50,100)
    plt.figure(figsize=(20,10))
    #Note: matplotlib uses RGB format so had to convert BGR-to-RGB
    plt.imshow(cv2.cvtColor(img_edge,cv2.COLOR_BGR2RGB))
    plt.title('RGB Image',color='c')
    plt.show()

    # circles = cv2.HoughCircles(img_edge,cv2.HOUGH_GRADIENT,1,minDist=20,param1=200,param2=70)
    circles = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, dp=1.3, minDist=20, param1=100, param2=45, minRadius=0, maxRadius=50)
    circles = np.uint16(np.around(circles))
    print(circles)
    for val in circles[0,:]:
        cv2.circle(og_image,(val[0],val[1]),val[2],(255,0,0),2)

    plt.figure(figsize=(20,10))
    plt.subplot(121),plt.imshow(cv2.cvtColor(img_edge,cv2.COLOR_BGR2RGB)),plt.title('Input',color='c')
    plt.subplot(122),plt.imshow(cv2.cvtColor(og_image,cv2.COLOR_BGR2RGB)),plt.title('Result',color='c')
    plt.show()

#Shape Detection
# path_to_img = 'Metal_6.png'
def detectShapes(img_path):
    img = cv2.imread(img_path,0)
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
    # # plt.subplot(133),plt.imshow(cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)),plt.title('Original')
    # plt.show()
    return blank

og_imge = "Metal_6.jpg"
outline = detectShapes(og_imge)
houghCircleDetector(outline, og_imge)
