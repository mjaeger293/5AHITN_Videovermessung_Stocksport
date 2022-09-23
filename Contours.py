import cv2
import numpy as np

class Image:
    def __init__(self, name, path):
        self.name = name
        self.path = path
    def loadnshow(self):
        img = cv2.imread(self.path)
        img = cv2.resize(img, (500, 500))
        #w, h = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bordes = cv2.Canny(gray, 100, 100)

        contours, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (255, 0, 0), 2)

        for i in contours:
            x, y, w, h = cv2.boundingRect(i)
            radius = h/2
            print(h/2)
            center = ((x+x+w)//2, (y+y+h)//2)
            cv2.circle(img, (center), 2, (255, 0, 0), 2)
            cv2.putText(img, "R: " + str(radius), (center[0]-45, center[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.imshow(self.name+f" [{str(len(contours))} Figuren erkannt]", img)
        print(f"{str(len(contours))} Konturen erkannt in {self.name}")



circulos = Image("kreise_gimp_verarbeitet_kontrast_helligkeit.png", "kreise_gimp_verarbeitet_kontrast_helligkeit.png")
circulos.loadnshow()
#circulos2 = Image("Circulos 2", "img/circulos2.png")
#circulos2.loadnshow()
#pelotas = Image("Pelotas", "img/pelotas2.jpg")
#pelotas.loadnshow()

cv2.waitKey(0)
cv2.destroyAllWindows()