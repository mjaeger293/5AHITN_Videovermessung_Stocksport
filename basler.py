'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)
'''
import time

from pypylon import pylon
import cv2
import platform
import os

img = pylon.PylonImage()
camera = None
ii = 0
test = 0

print("Waiting for camera ...")
# conecting to the first available camera
while camera is None:
    try:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    except:
        pass

def basler_to_OpenCV(grabResult):
    image = converter.convert(grabResult)
    return image.GetArray()


def save_pic_to_file(image):
    if platform.system() == 'Windows':
        # The JPEG format that is used here supports adjusting the image
        # quality (100 -> best quality, 0 -> poor quality).
        ipo = pylon.ImagePersistenceOptions()
        quality = 90
        ipo.SetQuality(quality)
        test = + 1
        if not os.path.exists("/images/capturedImages"):
            os.makedirs("/images/capturedImages")
            cv2.imwrite("../undistorted/" + "image" + str(test), basler_to_OpenCV(image))
        else:
            cv2.imwrite("../undistorted/" + "image" + str(test), basler_to_OpenCV(image))


print(camera.GetDeviceInfo().GetModelName() + " has been found!")
camera.Open()
# Grabing Continusely (video) with minimal delay
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
camera.StartGrabbing()

while True:
    with camera.RetrieveResult(5000) as result:
        input("Bitte drücken Sie eine Taste ein neues Bild aufzunehmen")
        img.AttachGrabResultBuffer(result)
        save_pic_to_file(img)
        print("Picture has been saved")
