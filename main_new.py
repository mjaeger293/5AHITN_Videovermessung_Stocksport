import random

from skimage.metrics import structural_similarity
import numpy as np
import cv2
import math
from images.camera_calibration.calibration import *

calibrationImage = None

NEW_CONFIG = False
UNDISTORTED_IMAGES_OUTPUT_PATH = "./images/undistorted"
CALIBRATION_IMAGES_INPUT_PATH = "./images/camera_calibration/1811"
DISTORTED_IMAGES_INPUT_PATH = "./images/04_11/edited"
CALIBRATION_FILE_OUTPUT_PATH = "./calibration.json"

run = True


def viewImage(image):
    image = cv2.resize(image, (768, 548))
    cv2.namedWindow('Display')
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def undistort(img_path, K, D, DIM):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #viewImage(undistorted_img)

    if not os.path.exists(UNDISTORTED_IMAGES_OUTPUT_PATH):
        os.makedirs(UNDISTORTED_IMAGES_OUTPUT_PATH)

    img_path = img_path.replace("/", "\\")
    fname = img_path.split("\\").pop()
    cv2.imwrite(UNDISTORTED_IMAGES_OUTPUT_PATH + "\\" + fname, undistorted_img)
    print(f"{fname}: saved at " + UNDISTORTED_IMAGES_OUTPUT_PATH)

    return undistorted_img


if NEW_CONFIG:
    getCalibrationParams(CALIBRATION_IMAGES_INPUT_PATH, CALIBRATION_FILE_OUTPUT_PATH)

with open(CALIBRATION_FILE_OUTPUT_PATH, 'r') as f:
    config = json.load(f)

K = np.array(config[0]['K'])
D = np.array(config[0]['D'])
DIM = np.array(config[0]['DIM'])


undistorted_images = []

images = os.listdir(DISTORTED_IMAGES_INPUT_PATH)

for img in images:
    if img.endswith(".jpg"):
        undistorted_images.append(undistort(DISTORTED_IMAGES_INPUT_PATH + "\\" + img, K, D, DIM))
        if img.endswith("calibration.jpg"):
            calibrationImage = undistorted_images.pop()

#while

for after in undistorted_images:

    before_gray = cv2.cvtColor(calibrationImage, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image Similarity: {:.4f}%".format(score * 100))

    viewImage(diff)

    #diff = (diff * 255).astype("uint8")
    #diff_box = cv2.merge([diff, diff, diff])

    #ret, bw_img = cv2.threshold(diff_box, 35, 255, cv2.THRESH_BINARY)

    #viewImage(bw_img)

    print("f")



