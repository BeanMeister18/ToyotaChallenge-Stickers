import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model('simple_ai_model.h5')

def predict(image):

	def preprocess_image(img):
		img = Image.open(img)
		img = img.resize((64, 64))
		img = np.array(img)
		img = np.expand_dims(img, axis=0)
		return img / 255.0

	processed_image = preprocess_image(image)
	predictions = model.predict(processed_image)
	predicted_class = np.argmax(predictions)
	return predicted_class

def detectShapes(img):
	img = cv2.medianBlur(img, 3)
	img_edge = cv2.Canny(img, 50, 100)
	contours, _ = cv2.findContours(img_edge.copy(), 1, 2)

	# init the blank img
	blank = np.zeros(img.shape, dtype="uint8")
	blank[:] = -1
	for num, cnt in enumerate(contours):
		x, y, w, h = cv2.boundingRect(cnt)
		approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
		# IF A LARGE CIRCLE
		if len(approx) > 10 and cv2.contourArea(cnt) > 1000:
			cv2.drawContours(blank, [cnt], -1,
							(0, 255, 0), thickness=cv2.FILLED)
	cv2.imwrite("filled.png", blank)

	# plt.figure(figsize=(200,100))
	# plt.subplot(131),plt.imshow(cv2.cvtColor(img_edge,cv2.COLOR_BGR2RGB)),plt.title('Input',color='c')
	# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),plt.title('Result')
	# plt.show()
	return blank


def elipsis(image, ogImage):

	# Set our filtering parameters
	# Initialize parameter setting using cv2.SimpleBlobDetector
	params = cv2.SimpleBlobDetector_Params()

	# Set Area filtering parameters
	params.filterByArea = True
	params.minArea = 100
	params.maxArea = float('inf')

	# Set Circularity filtering parameters
	params.filterByCircularity = False
	# params.minCircularity = 0.7

	# Set Convexity filtering parameters
	# params.filterByConvexity = True
	# params.minConvexity = 0.2

	# Set inertia filtering parameters
	params.filterByInertia = True
	params.minInertiaRatio = 0.02

	# Create a detector with the parameters
	detector = cv2.SimpleBlobDetector_create(params)

	# Detect blobs
	keypoints = detector.detect(image)

	# Draw blobs on our image as red circles
	blank = np.zeros((1, 1))
	print(keypoints)
	for keypoint in keypoints:
		x = round(keypoint.pt[0])
		y = round(keypoint.pt[1])
		r = 32
		try:
			cv2.imwrite("tmp.png", ogImage[y-r:y+r, x-r:x+r])
			prediction = predict("tmp.png")
			print(prediction)
			if (prediction == 0):
				ogImage = cv2.drawKeypoints(ogImage, (keypoint,), blank, (0, 120, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			if (prediction == 1):
				ogImage = cv2.drawKeypoints(ogImage, (keypoint,), blank, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		except Exception as e:
			print(e)

		
		# try:
		# 	cv2.imwrite("training/holes/"+str(random.randint(0,10000000000))+".png", crop)
		# except Exception as e:
		# 	print(e)
		
	# putting on the og image, for visualization
	# plt.imshow(cv2.cvtColor(bloobs,cv2.COLOR_BGR2RGB))
	# plt.title('RGB Image',color='c')
	# plt.show()
	return ogImage


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
	print("Cannot open camera")
	exit()
while True:
	# Capture frame-by-frame
	ret, frame = cap.read()

	# IF WE WANT AN IMAGE INSTEAD OF VIDEO FEED
	# frame = cv2.imread("pictures/Metal_2.jpg")

	# if frame is read correctly ret is True
	if not ret:
		print("Can't receive frame (stream end?). Exiting ...")
		break
	# Our operations on the frame come here
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Display the resulting frame
	outline = detectShapes(image)
	newImage = elipsis(outline, frame)
	# houghCircleDetector(outline, frame)
	cv2.imshow('frame', newImage)
	if cv2.waitKey(1) == ord('q'):
		break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()