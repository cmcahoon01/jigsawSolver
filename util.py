import cv2


def show(image, visualize=True, name="image"):
    if visualize:
        image = cv2.resize(image, (640, 480))
        cv2.imshow(name, image)
        cv2.waitKey(0)
