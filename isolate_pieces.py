import cv2
import numpy as np
from skimage.filters import sobel
from scipy import ndimage as ndi
from util import show

scale = 0.25
jigsaw = cv2.imread('images/aligned_jigsaw.jpg')

elevation_map = np.array(sobel(jigsaw))

elevation_map = np.clip((elevation_map * 255 * 3), 0, 255).astype(np.uint8)
# print(np.max(elevation_map))
#
# cv2.imshow('elevation_map', elevation_map)
# cv2.waitKey(0)
# print(elevation_map.shape, jigsaw.shape)

mask = cv2.inRange(elevation_map, (100, 100, 50), (200, 255, 100))

# mask = cv2.bitwise_not(mask)
filled = np.array(ndi.binary_fill_holes(mask)).astype(np.uint8) * 255

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
res = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)

show(mask)
show(res)

threshold = cv2.bitwise_not(res)
contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
i = 0
# list for storing names of shapes
blank = np.zeros(jigsaw.shape, np.uint8)
for contour in contours:
    # here we are ignoring first counter because
    # find contour function detects whole image as shape
    if i == 0:
        i = 1
        continue

    # using drawContours() function
    cv2.drawContours(jigsaw, [contour], 0, (0, 0, 255), -1)

show(jigsaw)
