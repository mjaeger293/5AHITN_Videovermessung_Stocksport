# import the necessary packages
import numpy as np
import cv2
import math


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



daubeIndex = 0
smallestRadius = 0
indexToDistance = {}
#newImage = np.zeros((s*2, s*3, 3), dtype = np.uint8)

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread("kreise_gimp_beruehren.png")
#image = cv2.resize(image, (2000,2000))

contrast_brightness = apply_brightness_contrast(image, 37, 77)

gray = cv2.cvtColor(contrast_brightness, cv2.COLOR_BGR2GRAY)
output = image.copy()

# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100, param1=10, param2=26)
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
		print(x,y,r)
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle

		cv2.circle(output, (x, y), r, (0, 255, 0), 2)
		cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

		if i != daubeIndex:
			a = abs(circles[daubeIndex][0] - x)
			b = abs(circles[daubeIndex][1] - y)

			c = math.sqrt(a ** 2 + b ** 2)
			c = (c / r) * 12.5;
			#c = c - r - circles[daubeIndex][2]

			indexToDistance[i] = c

			cv2.putText(output, "{0:.2f}cm".format(c), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 0), 1)
			cv2.line(output, (circles[daubeIndex][0], circles[daubeIndex][1]), (x, y), [0, 255, 0], 2)

		i = i + 1
	# show the output image
	output = cv2.resize(output, (500,500))
	cv2.imshow("output", output) #np.hstack([image, output])
	cv2.waitKey()
	print(circles[daubeIndex])
else:
	print("No circles found")
