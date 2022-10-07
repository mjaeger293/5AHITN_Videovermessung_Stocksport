"""
A simple Program for grabing video from basler camera and converting it to opencv img.
"""
from pypylon import pylon
import cv2

# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
# camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Grabs a single image from the camera
while True:
    input("Press Enter to take a picture ...")

    camera.StartGrabbing()
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data
            img = converter.Convert(grabResult)
            image = img.GetArray()

            cv2.namedWindow('title', cv2.WINDOW_NORMAL)
            cv2.imshow('title', image)
            cv2.waitKey(0)
        # code below lets user display a video
        # k = cv2.waitKey(1)
        # if k == 27:
        #    break
        grabResult.Release()

        # Releasing the resource
        camera.StopGrabbing()

        cv2.destroyAllWindows()
        break
