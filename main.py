# import the necessary packages
import numpy as np
import cv2
import math


def viewImage(image):
	cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
	cv2.imshow('Display', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
	if brightness != 0:
		if brightness > 0:
			shadow = brightness
			highlight = 255
		else:
			shadow = 0
			highlight = 255 + brightness
		alpha_b = (highlight - shadow) / 255
		gamma_b = shadow

		buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
	else:
		buf = input_img.copy()

	if contrast != 0:
		f = 131 * (contrast + 127) / (127 * (131 - contrast))
		alpha_c = f
		gamma_c = 127 * (1 - f)

		buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

	return buf


def getBorderPoint(radius, angle, currentPoint, targetPoint):
	newA = radius * math.sin(angle)
	newB = newA / math.tan(angle)

	if currentPoint[0] < targetPoint[0]:
		X = currentPoint[0] + newA
	else:
		X = currentPoint[0] - newA

	if currentPoint[1] < targetPoint[1]:
		Y = currentPoint[1] + newB
	else:
		Y = currentPoint[1] - newB

	return int(X), int(Y)


daubeIndex = 0
smallestRadius = 0
indexToDistance = {}
#newImage = np.zeros((s*2, s*3, 3), dtype = np.uint8)

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread("curling.jpg")
#image = cv2.resize(image, (2000,2000))

#contrast_brightness = apply_brightness_contrast(image, 37, 77)

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
viewImage(hsv_img)

# detect circles in the image

#RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
gray = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY)
viewImage(gray)

img = cv2.GaussianBlur(gray, (5, 5), 0)
viewImage(img)

#img = cv2.medianBlur(gray,5)

ret, threshold = cv2.threshold(img, 73, 255, 0)
viewImage(threshold)

#img = cv2.GaussianBlur(threshold, (5, 5), 0)
#viewImage(img)

circles = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT, 1.2, 100, param1=15, param2=36)
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	i = 0
	for (x, y, r) in circles:
		if smallestRadius == 0 or r < smallestRadius:
			smallestRadius = r
			daubeIndex = i
		i = i + 1

	i = 0
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle

		if r < 70:
			cv2.circle(image, (x, y), r, (0, 255, 0), 2)
			cv2.rectangle(image, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

			if i != daubeIndex:
				a = abs(circles[daubeIndex][0] - x)
				b = abs(circles[daubeIndex][1] - y)
				print(a, b)
				c = math.sqrt(a ** 2 + b ** 2)

				if c != 0:
					alpha = math.asin(a / c)

					indexToDistance[i] = c - r - circles[daubeIndex][2]

					cm = (indexToDistance[i] / r) * 12.5

					cv2.putText(image, "{0:.2f}cm".format(cm), (getBorderPoint(r, alpha, (x, y), (circles[daubeIndex][0], circles[daubeIndex][1]))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
					cv2.line(image, (getBorderPoint(circles[daubeIndex][2], alpha, (circles[daubeIndex][0], circles[daubeIndex][1]), (x, y))), (getBorderPoint(r, alpha, (x, y), (circles[daubeIndex][0], circles[daubeIndex][1]))), [0, 255, 0], 2)

			i = i + 1

	# show the output image
	output = cv2.resize(threshold, (500,500))
	viewImage(image) #np.hstack([image, output])
else:
	print("No circles found")
