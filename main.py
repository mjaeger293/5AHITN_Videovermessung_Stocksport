# import the necessary packages
import random

from skimage.metrics import structural_similarity
import numpy as np
import cv2
import math
from PIL import Image
import matplotlib.pyplot as plt



daubeIndex = 0
smallestRadius = 0
indexToDistance = {}
isFirstImage = True
daube = None



def viewImage(image):
    image = cv2.resize(image, (500, 500))
    cv2.namedWindow('Display')
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def drawStuff(s, diff_box, mask, filled_after, color):
    x, y, w, h = cv2.boundingRect(s)
    cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
    cv2.drawContours(mask, [s], 0, (255, 255, 255), -1)
    cv2.drawContours(filled_after, [s], 0, color, -1)


def testing(before, after):
    global daube
    global isFirstImage

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image Similarity: {:.4f}%".format(score * 100))

    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])
    #viewImage(diff_box)

    ret, bw_img = cv2.threshold(diff_box, 35, 255, cv2.THRESH_BINARY)

    #viewImage(bw_img)

    newImg = cv2.cvtColor(bw_img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(newImg, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()

    results = []

    for c in contours:
        area = cv2.contourArea(c)

        if area > 200:
            results.append(c)

    for r in results:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        drawStuff(r, diff_box, mask, filled_after, color)

    #drawStuff(daube, diff_box, mask, filled_after, (0, 0, 255))

    viewImage(mask)
    #cv2.imshow('filled after', cv2.resize(filled_after, (500, 500)))
    #viewImage(cv2.resize(diff_box, (500, 500)))
    #cv2.waitKey()

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    res = []
    daube = None
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,  1.2, 100, param1=70, param2=30)
    if circles is not None:
        #circles = np.uint16(np.around(circles))
        circles = np.round(circles[0, :]).astype("int")
        #for i in circles[0, :]:
        #    cv2.circle(after, (i[0], i[1]), i[2], (0, 255, 0), 1)
        #    cv2.circle(after, (i[0], i[1]), 2, (0, 0, 255), 3)

        for (x, y, r) in circles:
            if 10 < r < 60:
                if daube is None or r < daube[2]:
                    if daube is not None:
                        res.append(daube)
                    daube = (x, y, r)
                else:
                    res.append((x, y, r))

    i = 0
    for (x, y, r) in res:
        cv2.circle(after, (x, y), r, (0, 255, 0), 2)
        cv2.rectangle(after, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

        a = abs(daube[0] - x)
        b = abs(daube[1] - y)
        print(a, b)
        c = math.sqrt(a ** 2 + b ** 2)

        if c != 0:
            alpha = math.asin(a / c)

            indexToDistance[i] = c - r - daube[2]

            cm = (indexToDistance[i] / r) * 12.5

            cv2.putText(after, "{0:.2f}cm".format(cm),
                        (getBorderPoint(r, alpha, (x, y), (daube[0], daube[1]))),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            cv2.line(after, (
                getBorderPoint(daube[2], alpha, (daube[0], daube[1]),
                               (x, y))),
                     (getBorderPoint(r, alpha, (x, y), (daube[0], daube[1]))),
                     [255, 0, 0], 2)
    i += 1

    cv2.circle(after, (daube[0], daube[1]), daube[2], (0, 0, 255), 2)

    viewImage(after)

# Load images

im1 = cv2.imread('images\\bruh\\nig\\base.png')
im2 = cv2.imread('images\\bruh\\nig\\config.png')
im3 = cv2.imread('images\\bruh\\nig\\i1.png')
im4 = cv2.imread('images\\bruh\\nig\\i2.png')

testing(im1, im2)

testing(im1, im3)

testing(im1, im4)

'''
circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1.2, 100, param1=70, param2=30)
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

        if r < 60:
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

                    cv2.putText(image, "{0:.2f}cm".format(cm),
                                (getBorderPoint(r, alpha, (x, y), (circles[daubeIndex][0], circles[daubeIndex][1]))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    cv2.line(image, (
                        getBorderPoint(circles[daubeIndex][2], alpha, (circles[daubeIndex][0], circles[daubeIndex][1]),
                                       (x, y))),
                             (getBorderPoint(r, alpha, (x, y), (circles[daubeIndex][0], circles[daubeIndex][1]))),
                             [0, 255, 0], 2)

            i = i + 1

    output = cv2.resize(threshold, (500, 500))
    viewImage(image)  # np.hstack([image, output])
else:
    print("No circles found")
    
    '''
