import cv2
import numpy as np
from util import show
from find_pieces import get_contours
from cluster_pieces import extract_features, isolate_pieces

def find_colors(images):
    pass

if __name__ == '__main__':
    img = cv2.imread('images/connected.jpg')
    cnt = get_contours(img)[1:]
    locs = isolate_pieces(img, cnt, visualize=False)
    extract_features(img, locs, cnt, visualize=True)
