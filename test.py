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

image = cv2.imread("Metal_2.jpg")
plt.figure(figsize=(20,10))
#Note: matplotlib uses RGB format so had to convert BGR-to-RGB
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title('RGB Image',color='c')
plt.show()

# cv2.imwrite("testttt.png", image)
# cv2.imshow("test", image)