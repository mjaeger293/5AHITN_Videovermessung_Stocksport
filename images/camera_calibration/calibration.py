import cv2
import numpy as np
import os


def viewImage(image):
    image = cv2.resize(image, (768, 548))
    cv2.namedWindow('Display')
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


CHECKERBOARD = (6, 8)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = []

folder_dir = "D:\\Schule\\5AHITN\Schweiger\\5AHITN_Videovermessung_Stocksport\\images\\camera_calibration\\1811"
for img in os.listdir(folder_dir):
    if img.endswith(".jpg"):
        images.append(img)

print(len(images))

#images.append("Image__2022-11-18__08-16-05.jpg")

i = 1

for fname in images:

    img = cv2.imread("1811\\" + fname)

    print(f"{i}: " + fname)

    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."

    thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)[1]

    inverted_image = cv2.bitwise_not(thresh)

    gray = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)

    #viewImage(gray)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("TRUE")
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, True)
        #viewImage(img)

    else:
        print("FALSE")

    i += 1


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

DIM = (img.shape[:2][1], img.shape[:2][0])


def undistort(img_path):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #viewImage(undistorted_img)
    #print(img_path.split("\\").pop())
    cv2.imwrite("../undistorted/" + img_path.split("\\").pop(), undistorted_img)
    print("done")


images.clear()
folder_dir = "D:\\Schule\\5AHITN\Schweiger\\5AHITN_Videovermessung_Stocksport\\images\\04_11\\edited"

for img in os.listdir(folder_dir):
    if img.endswith(".jpg"):
        undistort(folder_dir + "\\" + img)


