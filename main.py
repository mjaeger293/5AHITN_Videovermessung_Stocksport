# import the necessary packages
import numpy as np
import argparse
import cv2

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread("kreise_gimp_verarbeitet_kontrast_helligkeit.png")
#image = cv2.resize(image, (2000,2000))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
output = image.copy()

# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100, param1=10, param2=25)
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		print(x,y,r)
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 2)
		cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
	# show the output image
	output = cv2.resize(output, (500,500))
	cv2.imshow("output", output) #np.hstack([image, output])
	cv2.waitKey()
else:
	print("No circles found")