import cv2
import numpy as np
import os
import glob
import os
from os import listdir


def viewImage(image):
    image = cv2.resize(image, (960, 540))
    cv2.namedWindow('Display')
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


images = []

folder_dir = "D:\\Schule\\5AHITN\Schweiger\\5AHITN_Videovermessung_Stocksport\\images\\camera_calibration\\1811"
for img in os.listdir(folder_dir):
    if img.endswith("test2.jpg"):
        images.append(img)

print(len(images))

i = 0

CHECKERBOARD = (6,9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in images:

    i+=1
    print(f"Image: {i}")
    img = cv2.imread("1811\\" + fname)

    pattern = cv2.imread("pattern.png")

    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (51, 51), 0)

    thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)[1]

    inverted_image = cv2.bitwise_not(thresh)
    viewImage(inverted_image)

    corners_2 = cv2.goodFeaturesToTrack(inverted_image,100,0.01,290)
    corners_2 = np.int0(corners_2)

    print(len(corners_2))

    for i in corners_2:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 25, (0, 0, 255), -1)

    viewImage(img)

    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(pattern, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("TRUE")
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)
    else:
        print("FALSE")

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")
