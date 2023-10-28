import numpy as np
import matplotlib.pyplot as plt
import cv2

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
	params.minArea = 1000

	# Set Circularity filtering parameters
	# params.filterByCircularity = True
	# params.minCircularity = 0.7

	# # Set Convexity filtering parameters
	# params.filterByConvexity = True
	# params.minConvexity = 0.2q

	# # # Set inertia filtering parameters
	# params.filterByInertia = True
	# params.minInertiaRatio = 0.01

	# Create a detector with the parameters
	detector = cv2.SimpleBlobDetector_create(params)

	# Detect blobs
	keypoints = detector.detect(image)

	# Draw blobs on our image as red circles
	blank = np.zeros((1, 1))
	bloobs = cv2.drawKeypoints(ogImage, keypoints, blank, (0, 0, 255),
							cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# plt.imshow(cv2.cvtColor(bloobs,cv2.COLOR_BGR2RGB))
	# plt.title('RGB Image',color='c')
	# plt.show()
	return bloobs


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
	# frame = cv2.imread("Metal_2.jpg")

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