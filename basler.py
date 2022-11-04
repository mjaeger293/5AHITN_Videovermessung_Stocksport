'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)
'''
import time

from pypylon import pylon
import cv2
import platform

res = pylon.PylonImage()
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


def save_pic_to_file(image):
    if platform.system() == 'Windows':
        # The JPEG format that is used here supports adjusting the image
        # quality (100 -> best quality, 0 -> poor quality).
        ipo = pylon.ImagePersistenceOptions()
        quality = 90
        ipo.SetQuality(quality)
        test = + 1
        filename = "images/test/saved_pypylon_img_%d.jpeg" % test
        image.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)
    else:
        filename = "saved_pypylon_img_%d.png" % 2
        img.Save(pylon.ImageFileFormat_Png, filename)


print(camera.GetDeviceInfo().GetModelName() + " has been found!")
camera.Open()
pylon.FeaturePersistence.Load("test.txt", camera.GetNodeMap())
nodemap = camera.GetNodeMap()

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

while camera.IsGrabbing():
    with camera.RetrieveResult(5000) as result:
        time.sleep(8)
        res.AttachGrabResultBuffer(result)
        save_pic_to_file(res)
        print("Photo has been taken!")

        if result.GrabSucceeded():
            # Access the image data
            image = converter.Convert(result)
            img = image.GetArray()
            cv2.namedWindow('title', cv2.WINDOW_NORMAL)
            cv2.imshow('title', img)
            k = cv2.waitKey(1)
            if k == 27:
                break
            result.Release()

# Releasing the resource
camera.StopGrabbing()

cv2.destroyAllWindows()
