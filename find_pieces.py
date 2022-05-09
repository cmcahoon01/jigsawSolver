import cv2
import numpy as np
from skimage.filters import sobel
from scipy import ndimage as ndi
from util import show


def get_contours(image, visualize=False):
    # resize image
    image = cv2.GaussianBlur(image, (5, 5), 0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)

    # displaying the image after drawing contours
    show(threshold, visualize)
    kernel = np.ones((7, 7), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5, 5), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    threshold[:, 0] = 255
    threshold[0, :] = 255
    threshold[:, -1] = 255
    threshold[-1, :] = 255
    show(threshold, visualize)

    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
    i = 0
    # list for storing names of shapes
    for contour in contours:

        # here we are ignoring first counter because
        # find contour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # using drawContours() function
        cv2.drawContours(image, [contour], 0, (0, 0, 255), 5)

    # displaying the image after drawing contours
    show(image, visualize)

    return contours


if __name__ == '__main__':
    image = cv2.imread('images/connected.jpg')
    get_contours(image, visualize=True)
