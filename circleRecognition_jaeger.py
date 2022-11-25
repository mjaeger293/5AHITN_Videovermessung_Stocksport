# import the necessary packages
import numpy as np
import cv2
import math


# Change contrast and brightness
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

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread("images/undistorted/Image__2022-11-04__09-31-22_11zon.jpg")
image = cv2.resize(image, (500,500))
cv2.imshow("image", image) #np.hstack([image, output])
cv2.waitKey()
#image = cv2.resize(image, (2000,2000))

#gray = apply_brightness_contrast(image, -106, 122)
#cv2.imshow("kontrast", gray) #np.hstack([image, output])
#cv2.waitKey()
image = (255-image)
cv2.imshow("image", image) #np.hstack([image, output])
cv2.waitKey()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("image", gray) #np.hstack([image, output])
cv2.waitKey()

#mask = cv2.inRange(gray, (22, 93, 0), (180 , 255, 255)) # (22, 93, 0), (45, 255, 255)   #(15, 50, 0), (70, 255, 255)
#cv2.imshow("image", mask)
#cv2.waitKey()

#mask_inv = cv2.bitwise_not(mask)
#cv2.imshow("image", mask_inv)
#cv2.waitKey()

#mask2 = cv2.inRange(gray, (10, 82, 25), (22, 84, 255))
#cv2.imshow("mask2", mask2)
#cv2.waitKey()

#gray = cv2.bitwise_and(image, image, mask=mask_inv)
#cv2.imshow("image", gray)
#cv2.waitKey()

gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
cv2.imshow("image", gray) #np.hstack([image, output])
cv2.waitKey()

# apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
gray = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("image", gray) #np.hstack([image, output])
cv2.waitKey()
gray = cv2.medianBlur(gray, 5)
cv2.imshow("image", gray) #np.hstack([image, output])
cv2.waitKey()

# Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
							 cv2.THRESH_BINARY, 11, 3.5)

cv2.imshow("image", gray) #np.hstack([image, output])
cv2.waitKey()

kernel = np.ones((2, 2), np.uint8)
gray = cv2.erode(gray, kernel, iterations=1)
# gray = erosion
cv2.imshow("image", gray) #np.hstack([image, output])
cv2.waitKey()

gray =  cv2.dilate(gray, kernel, iterations=1)
cv2.imshow("image", gray)
cv2.waitKey()

output = gray.copy()

# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 260, param1=20, param2=34)
print("Detected circles!", circles)
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

		if True:
			cv2.circle(output, (x, y), r, (0, 255, 0), 2)
			cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

			if i != daubeIndex:
				a = abs(circles[daubeIndex][0] - x)
				b = abs(circles[daubeIndex][1] - y)

				c = math.sqrt(a ** 2 + b ** 2)
				c = (c / r) * 12.5;
				#c = c - r - circles[daubeIndex][2]

				indexToDistance[i] = c

				cv2.putText(output, "{0:.2f}cm".format(c), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 1)
				cv2.line(output, (circles[daubeIndex][0], circles[daubeIndex][1]), (x, y), [0, 255, 0], 2)

		i = i + 1
	# show the output image
	test = cv2.resize(output, (500,500))
	cv2.imshow("image", output) #np.hstack([image, output])
	cv2.waitKey()
	print(circles[daubeIndex])
else:
	print("No circles found")
