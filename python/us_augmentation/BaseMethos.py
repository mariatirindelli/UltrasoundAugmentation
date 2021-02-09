import cv2
import numpy as np

class BaseMethod():
    def __init__(self):
        self.name = ""

    @staticmethod
    def _blur(image, kernel, sigma):

        blur_image = cv2.GaussianBlur(image * 255, (kernel, kernel) , sigma)
        blur_image = blur_image / np.max(blur_image)
        return blur_image

    def execute(self, image, label):
        raise NotImplementedError

