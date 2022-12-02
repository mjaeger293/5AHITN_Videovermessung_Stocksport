'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)
'''
import time

import numpy as np
from pypylon import pylon
import cv2
import platform
import os
from PIL import Image, ImageEnhance

IMAGES_OUTPUT_PATH = "./images/capturedImages"
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
    # converting to opencv bgr format
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    image = converter.convert(grabResult)

    return image.GetArray()


def save_pic_to_file(image):
    if platform.system() == 'Windows':
        # The JPEG format that is used here supports adjusting the image
        # quality (100 -> best quality, 0 -> poor quality).
        ipo = pylon.ImagePersistenceOptions()
        quality = 70
        ipo.SetQuality(quality)

        if not os.path.exists(IMAGES_OUTPUT_PATH):
            os.makedirs(IMAGES_OUTPUT_PATH)

        image.Save(pylon.ImageFileFormat_Jpeg, IMAGES_OUTPUT_PATH + "\\" + "picture" + str(test) + ".jpg", ipo)
        image.Release()


print(camera.GetDeviceInfo().GetModelName() + " has been found!")
camera.Open()
camera.AutoTargetValue.SetValue(150)
camera.GainAuto.SetValue("Continuous")
# Grabing Continusely (video) with minimal delay

camera.StartGrabbing()

while True:
    with camera.RetrieveResult(5000) as result:
        input("Bitte drücken Sie eine Taste ein neues Bild aufzunehmen")
        img = pylon.PylonImage()
        img.AttachGrabResultBuffer(result)
        save_pic_to_file(img)
        print("Picture has been saved")
        test = test + 1
