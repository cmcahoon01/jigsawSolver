import cv2
import numpy as np
from math import floor
from histogram_matching import match_histograms, mean_diff, equalize_both
from util import show

scale = 0.25
padding = 40

# get image
solution = cv2.imread('images/jigsaw.jpg')
# resize image
solution = cv2.GaussianBlur(solution, (3, 3), 0)
# solution = cv2.resize(solution, (0, 0), fx=scale, fy=scale)
solution = cv2.GaussianBlur(solution, (3, 3), 0)

corners = [(padding, padding), (padding, solution.shape[0] - padding),
           (solution.shape[1] - padding, solution.shape[0] - padding),
           (solution.shape[1] - padding, padding)]
edge_width = 80
inner_corners = [(padding + edge_width, padding + edge_width),
                 (padding + edge_width, solution.shape[0] - padding - edge_width),
                 (solution.shape[1] - padding - edge_width, solution.shape[0] - padding - edge_width),
                 (solution.shape[1] - padding - edge_width, padding + edge_width)]

sol_corners = np.array([(11, 23), (16, 738), (996, 730), (1002, 28)])*4

transformation_matrix = cv2.getPerspectiveTransform(np.float32(sol_corners), np.float32(corners))
warped_solution = cv2.warpPerspective(solution, transformation_matrix, (solution.shape[1], solution.shape[0]))
# crop
warped_solution = warped_solution[padding:warped_solution.shape[0] - padding,
                  padding:warped_solution.shape[1] - padding]
show(warped_solution)
#
# reference = cv2.imread('images/reference.jpg')
# # resize image
# reference = cv2.resize(reference, (0, 0), fx=scale, fy=scale)
# reference = cv2.GaussianBlur(reference, (3, 3), 0)
#
# ref_corners = [(35, 39), (29, 730), (988, 737), (991, 46)]
#
# transformation_matrix = cv2.getPerspectiveTransform(np.float32(ref_corners), np.float32(corners))
# warped_reference = cv2.warpPerspective(reference, transformation_matrix, (reference.shape[1], reference.shape[0]))
# # crop
# warped_reference = warped_reference[padding:warped_reference.shape[0] - padding,
#                    padding:warped_reference.shape[1] - padding]

# sol_gray = cv2.cvtColor(warped_solution, cv2.COLOR_BGR2GRAY)
# ref_gray = cv2.cvtColor(warped_reference, cv2.COLOR_BGR2GRAY)
#
# mask = np.ones((warped_solution.shape[0], warped_solution.shape[1]), dtype=np.uint8)
# mask[edge_width:warped_solution.shape[0] - edge_width, edge_width:warped_solution.shape[1] - edge_width] = 0
#
# # cv2.imshow("solution", warped_solution)
# # cv2.waitKey(0)
# solution_edge = cv2.bitwise_and(warped_solution, warped_solution, mask=mask)
# reference_edge = cv2.bitwise_and(warped_reference, warped_reference, mask=mask)
#
# cv2.imshow("solution", solution_edge)
# cv2.waitKey(0)
# cv2.imshow("reference", reference_edge)
# cv2.waitKey(0)
#
# transformed_reference, transformed_solution = equalize_both(reference_edge, solution_edge)
# cv2.imshow("transformed_reference", transformed_reference)
# cv2.waitKey(0)
#
# cv2.imshow("transformed_solution", transformed_solution)
# cv2.waitKey(0)

# cv2.imshow("solution", warped_solution)
# cv2.waitKey(0)


# np_solution = np.array(warped_solution)
# np_reference = np.array(warped_reference)
#
# diff = np.abs(np_solution - np_reference)
# diff = np.linalg.norm(diff, axis=2)
# diff_image = diff.astype(np.uint8).clip(0, 255)
#
# _, thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)
# image = np.array(warped_reference)
# image[thresh == 0] = (0, 0, 0)

# threshold = 250
# mask = diff > threshold
# image[mask] = (255, 255, 255)
# new = cv2.bitwise_and(image, image, mask=thresh)
# diff = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
#                              cv2.THRESH_BINARY, 15, 30)
#
# show difference
# cv2.imshow('difference', diff_image)
# cv2.waitKey(0)
#
# cv2.imshow('difference', thresh)
# cv2.waitKey(0)
#
# mixed = cv2.addWeighted(warped_solution, 0.3, warped_reference, 0.7, 0)
# cv2.imshow('mixed', mixed)
# cv2.waitKey(0)
