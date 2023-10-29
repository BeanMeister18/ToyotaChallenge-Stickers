# import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model('simple_ai_model.h5')

# model prediction function, takes in an image path
def predict(image_path):

	def preprocess_image(image_path):
		img = Image.open(image_path)
		img = img.resize((64, 64))
		img = np.array(img)
		img = np.expand_dims(img, axis=0)
		return img / 255.0

	processed_image = preprocess_image(image_path)
	predictions = model.predict(processed_image)
	predicted_class = np.argmax(predictions)
	return predicted_class

# canny algorithm to detect shapes, will take in the image and 
def detectShapes(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.medianBlur(img, 3)
	img_edge = cv2.Canny(img, 50, 100)
	contours, _ = cv2.findContours(img_edge.copy(), 1, 2)

	# Create a blank image to draw the found shapes onto.
	blank = np.zeros(img.shape, dtype="uint8")
	blank[:] = -1
	
	for num, cnt in enumerate(contours):

		x, y, w, h = cv2.boundingRect(cnt)
		approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
		
		# IF A LARGE CIRCLE
		if len(approx) > 10 and cv2.contourArea(cnt) > 1000:
			cv2.drawContours(blank, [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)
	
	# just outputting the filled image
	# cv2.imwrite("filled.png", blank)

	return blank

# Function to find the ellipsis
# Parameters: The b/w image, and the original image to paint on top of
def ellipsis(image, ogImage):

	# Set our filtering parameters
	# Initialize parameter setting using cv2.SimpleBlobDetector
	def getParams():

		params = cv2.SimpleBlobDetector_Params()

		params.filterByArea = True
		params.filterByCircularity = False
		params.filterByInertia = True
		# params.filterByConvexity = True

		params.minArea = 100
		params.maxArea = float('inf')
		params.minInertiaRatio = 0.02
		# params.minCircularity = 0.7
		# params.minConvexity = 0.2

		return params


	# get the parameters and detect the blobs
	params = getParams()
	detector = cv2.SimpleBlobDetector_create(params)
	keypoints = detector.detect(image)

	# Draw blobs on our image as red circles
	blank = np.zeros((1, 1))
	
	for keypoint in keypoints:
		
		x = round(keypoint.pt[0])
		y = round(keypoint.pt[1])
		r = 32

		try:
			cv2.imwrite("tmp.png", ogImage[y-r:y+r, x-r:x+r])
			prediction = predict("tmp.png")

			if (prediction == 0):
				ogImage = cv2.drawKeypoints(ogImage, (keypoint,), blank, (0, 120, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			if (prediction == 1):
				ogImage = cv2.drawKeypoints(ogImage, (keypoint,), blank, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		
		except Exception as e:
			print(e)

		# used for data collection
		# try:
		# 	cv2.imwrite("training/holes/"+str(random.randint(0,10000000000))+".png", crop)
		# except Exception as e:
		# 	print(e)
		
	return ogImage

if __name__ == "__main__":

	# set up camera capture
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

	while True:
		ret, frame = cap.read()

		# IF WE WANT AN IMAGE INSTEAD OF VIDEO FEED, we can load it here
		# frame = cv2.imread("pictures/Metal_2.jpg")
		
		# cv algorithm
		outline = detectShapes(frame)
		annotatedImage = ellipsis(outline, frame)
		
		# Display the resulting frame
		cv2.imshow('frame', annotatedImage)
		if cv2.waitKey(1) == ord('q'):
			break
	
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()