import cv2
import keyboard
from PIL import Image
from numpy import array

from Matrice import Matrice


class ImageData:
    def __init__(self, pathImage: str = None, image: array = None):
        if pathImage is not None:
            self.__image: array = array(Image.open(pathImage))
            self.__answer: int = int('CMFD' in pathImage)  # convertit False/True en 0/1

        else:
            self.__image: array = image
            self.__answer: int = -1

        self.__dim: tuple[int, int] = self.__image.shape

    def getAnswer(self) -> int:
        return self.__answer

    def getImageList(self) -> list[list[int]]:
        imageArray: array = self.__image.reshape((1, self.__dim[0] * self.__dim[1]))

        return imageArray.tolist()[0]

    def getMatrice(self) -> Matrice:
        return Matrice(self.__image.tolist())

    def getDim(self) -> tuple[int, int]:
        return self.__dim

    def showImage(self):
        while 1:
            cv2.imshow('test', self.__image)

            if cv2.waitKey(50) and (ord('q') == 0xFF or keyboard.is_pressed('space')):
                break
