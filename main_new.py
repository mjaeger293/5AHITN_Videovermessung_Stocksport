import random
import shutil

from skimage.metrics import structural_similarity
import numpy as np
import cv2
import math
from images.camera_calibration.calibration import *

calibrationImage = None

NEW_CONFIG = False
UNDISTORTED_IMAGES_OUTPUT_PATH = "./images/undistorted"
BRIGHTNESS_IMAGES_OUTPUT_PATH = "./images/compare"
CALIBRATION_IMAGES_INPUT_PATH = "./images/camera_calibration/1811"
DISTORTED_IMAGES_INPUT_PATH = "./images/04_11/edited"
CALIBRATION_FILE_OUTPUT_PATH = "./calibration.json"

run = True


def drawStuff(s, diff_box, mask, filled_after, color):
    x, y, w, h = cv2.boundingRect(s)
    cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
    cv2.drawContours(mask, [s], 0, (255, 255, 255), -1)
    cv2.drawContours(filled_after, [s], 0, color, -1)


def calcBrightness(hsv, target):
    step = 1
    plus = False
    minus = False

    while hsv[..., 2].mean() < target - 0.3 or hsv[..., 2].mean() > target + 0.3:
        if hsv[..., 2].mean() > target:
            if minus:
                return hsv
            plus = True
            hsv = decrease_brightness(hsv, step)
        else:
            if plus:
                return hsv
            minus = True
            hsv = increase_brightness(hsv, step)

    return hsv


def increase_brightness(temp, value=30):
    hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    temp = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return temp


def decrease_brightness(temp, value=30):
    hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = value
    v[v < lim] = 0
    v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    temp = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return temp


def viewImage(image):
    image = cv2.resize(image, (768, 548))
    cv2.namedWindow('Display')
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def undistort(img_path, K, D, DIM):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # viewImage(undistorted_img)

    return undistorted_img


if NEW_CONFIG:
    getCalibrationParams(CALIBRATION_IMAGES_INPUT_PATH, CALIBRATION_FILE_OUTPUT_PATH)

with open(CALIBRATION_FILE_OUTPUT_PATH, 'r') as f:
    config = json.load(f)

K = np.array(config[0]['K'])
D = np.array(config[0]['D'])
DIM = np.array(config[0]['DIM'])

images = os.listdir(DISTORTED_IMAGES_INPUT_PATH)
shutil.rmtree(UNDISTORTED_IMAGES_OUTPUT_PATH)

for img in images:
    if img.endswith(".jpg"):
        if not os.path.exists(UNDISTORTED_IMAGES_OUTPUT_PATH):
            os.makedirs(UNDISTORTED_IMAGES_OUTPUT_PATH)

        uimg = undistort(DISTORTED_IMAGES_INPUT_PATH + "\\" + img, K, D, DIM)
        cv2.imwrite(UNDISTORTED_IMAGES_OUTPUT_PATH + "\\" + img, uimg)
        print(f"{img}: saved at " + UNDISTORTED_IMAGES_OUTPUT_PATH)

        if img.endswith("calibration.jpg"):
            calibrationImage = uimg

i = 1
for after in os.listdir(UNDISTORTED_IMAGES_OUTPUT_PATH):
    if after.endswith(".jpg"):
        print(i)

        image = cv2.imread(UNDISTORTED_IMAGES_OUTPUT_PATH + "\\" + after)

        imgHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        calib = cv2.cvtColor(calibrationImage, cv2.COLOR_RGB2HSV)

        # adjusted_img = calcBrightness(imgHSV, calib[..., 2].mean())
        adjusted_img = imgHSV

        print(f"original: {calib[..., 2].mean()}")
        print(f"result: {adjusted_img[..., 2].mean()}")

        if i == 1:
            cv2.imwrite(BRIGHTNESS_IMAGES_OUTPUT_PATH + "\\" + after, cv2.cvtColor(calib, cv2.COLOR_HSV2RGB))
        else:

            res = cv2.cvtColor(adjusted_img, cv2.COLOR_HSV2RGB)

            cv2.imwrite(BRIGHTNESS_IMAGES_OUTPUT_PATH + "\\res_" + after, res)

            before_gray = cv2.cvtColor(calibrationImage, cv2.COLOR_BGR2GRAY)
            after_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            (score, diff) = structural_similarity(before_gray, after_gray, full=True)
            print("Image Similarity: {:.4f}%".format(score * 100))

            diff = (diff * 255).astype("uint8")

            thresh = cv2.threshold(diff, 190, 255, cv2.THRESH_BINARY_INV)[1]

            contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.RETR_TREE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            mask = np.zeros(image.shape, dtype='uint8')

            results = []

            for c in contours:
                area = cv2.contourArea(c)

                if area > 12000:
                    # if cv2.arcLength(c, False) > 1000:
                    results.append(c)

            print(len(results))
            for r in results:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.drawContours(mask, [r], 0, (255, 255, 255), cv2.FILLED)
                cv2.drawContours(res, [r], 0, color, cv2.FILLED)
                # der weiße Stock besteht aus 2 Konturen daher sind anstatt 6 7 Konturen gefunden

            viewImage(mask)

            viewImage(res)
        # cv2.imwrite('brightness.png', imgHSV[..., 2])

        # calibrationImage = after

        # diff = (diff * 255).astype("uint8")
        # diff_box = cv2.merge([diff, diff, diff])

        # ret, bw_img = cv2.threshold(diff_box, 35, 255, cv2.THRESH_BINARY)

        # viewImage(bw_img)

        i += 1
