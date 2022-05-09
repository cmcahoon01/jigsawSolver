import cv2
import numpy as np
from util import show
from math import atan2
from find_pieces import get_contours


def isolate_pieces(image, pieces, visualize=False):
    locations = np.zeros((len(pieces), 4), dtype=np.int32)
    for i, piece in enumerate(pieces):
        top = np.min(piece[:, 0, 1])
        bot = np.max(piece[:, 0, 1])
        left = np.min(piece[:, 0, 0])
        right = np.max(piece[:, 0, 0])
        locations[i] = (left, top, right, bot)
    if not visualize:
        return locations

    image = np.array(image)
    index = 2
    left = locations[index][0]
    top = locations[index][1]
    right = locations[index][2]
    bot = locations[index][3]
    cv2.circle(image, (left, top), 30, (0, 255, 255), -1)
    show(image, visualize)
    box = image[top:bot, left:right]
    show(box, visualize)
    return locations


def extract_features(image, locations, contours, visualize=False):
    boxes = []

    mask = np.zeros(image.shape[:-1], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    for index in range(len(locations)):
        cropped = masked.copy()
        box = mask[locations[index][1]:locations[index][3],
              locations[index][0]:locations[index][2]]
        cropped = cropped[locations[index][1]:locations[index][3],
              locations[index][0]:locations[index][2]]
        show(cropped, visualize)

        cont = np.array(contours[index])
        for i in range(len(cont)):
            cont[i][0][0] -= locations[index][0]
            cont[i][0][1] -= locations[index][1]

        epsilon = 0.003 * cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, epsilon, True)
        if visualize:
            box = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(box, [approx], -1, (0, 0, 255), 3)
            show(box, visualize)

        lines = np.zeros((len(approx), 2, 2), dtype=np.int32)
        for i in range(len(approx)):
            if i == len(approx) - 1:
                lines[i] = (approx[i][0], approx[0][0])
            else:
                lines[i] = (approx[i][0], approx[i + 1][0])
        line_lengths = np.zeros(len(lines))
        for i in range(len(lines)):
            line_lengths[i] = np.linalg.norm(lines[i][0] - lines[i][1])

        num_indicators = 5
        longest = np.argsort(line_lengths)[-1:-1 - num_indicators:-1]
        # if visualize:
        #     # box = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
        #     # cv2.line(box, lines[longest][0], lines[longest][1], (0, 255, 0), 3)
        #     for i in longest4:
        #         cv2.line(box, lines[i][0], lines[i][1], (0, 255, 0), 3)
        #     show(box, visualize)

        angles = np.zeros((num_indicators,))
        for i in range(len(angles)):
            angles[i] = atan2(lines[longest[i]][0][1] - lines[longest[i]][1][1],
                              lines[longest[i]][0][0] - lines[longest[i]][1][0])
            if angles[i] < 0:
                angles[i] += np.pi
        horizontal = [[longest[0], angles[0]]]
        vertical = []
        other = []
        for i in range(1, len(angles)):
            threshold = np.pi / 4
            diff = abs(angles[i] - angles[0])
            if diff > np.pi / 2:
                diff = np.pi - diff
            if diff < threshold:
                horizontal.append([longest[i], angles[i]])
            elif diff < np.pi - threshold:
                vertical.append([longest[i], angles[i]])
            else:
                other.append([longest[i], angles[i]])
        if visualize:
            # box = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
            for i, j in horizontal:
                cv2.line(box, lines[i][0], lines[i][1], (0, 255, 0), 3)
            for i, j in vertical:
                cv2.line(box, lines[i][0], lines[i][1], (255, 0, 0), 3)
            for i, j in other:
                cv2.line(box, lines[i][0], lines[i][1], (0, 255, 255), 3)
            show(box, visualize)

        horizontal = np.array(horizontal)
        vertical = np.array(vertical)
        all_angles = np.array(horizontal[:, 1])
        for _, angle in vertical:
            val = angle - np.pi / 2 if angle > np.pi / 2 else angle + np.pi / 2
            all_angles = np.append(all_angles, val)

        all_angles = [np.pi - angle if angle > np.pi / 2 else angle for angle in all_angles]
        average_orientation = np.mean(all_angles)

        # rotate box
        box = rotate(box, average_orientation)
        cropped = rotate(cropped, average_orientation)
        show(box, visualize)
        show(cropped, visualize)

        boxes.append(cropped)


def rotate(image, angle):
    padding = (np.array(image.shape[:2])/3).astype(int)
    padded = np.pad(image, ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant')
    (h, w) = padded.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D((cX, cY), np.degrees(angle), 1.0)
    rotated = cv2.warpAffine(padded, m, (w, h))
    return rotated


if __name__ == '__main__':
    img = cv2.imread('images/connected.jpg')
    cnt = get_contours(img)[1:]
    locs = isolate_pieces(img, cnt, visualize=False)
    extract_features(img, locs, cnt, visualize=True)
