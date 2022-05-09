import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt


def mean_diff(ref_image, src_image, mask):
    ref_blurred = cv2.GaussianBlur(ref_image, (41, 41), 0)
    src_blurred = cv2.GaussianBlur(src_image, (41, 41), 0)
    masked_ref = cv2.bitwise_and(ref_blurred, ref_blurred, mask=mask)
    masked_src = cv2.bitwise_and(src_blurred, src_blurred, mask=mask)
    masked_ref = cv2.cvtColor(masked_ref, cv2.COLOR_BGR2HSV)
    masked_src = cv2.cvtColor(masked_src, cv2.COLOR_BGR2HSV)

    diff = masked_src - masked_ref

    # visual_diff = ((diff + 128)/2).astype(np.uint8)
    # cv2.imshow("Visual Diff", visual_diff)
    # cv2.waitKey(0)

    avg_diff = np.average(diff, axis=(0, 1))
    print(avg_diff)

    transformed_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2HSV)
    transformed_ref = (transformed_ref + diff).astype(np.uint8)
    transformed_ref = cv2.cvtColor(transformed_ref, cv2.COLOR_HSV2BGR)
    return transformed_ref


def match_histograms(ref, src, mask):
    matched = exposure.match_histograms(src, ref, multichannel=True)
    (fig, axs) = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
    # loop over our source image, reference image, and output matched
    # image
    for (i, image) in enumerate((src, ref, matched)):
        # convert the image from BGR to RGB channel ordering
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # loop over the names of the channels in RGB order
        for (j, color) in enumerate(("red", "green", "blue")):
            # compute a histogram for the current channel and plot it
            (hist, bins) = exposure.histogram(image[..., j],
                                              source_range="dtype")
            axs[j, i].plot(bins, hist / hist.max())
            # compute the cumulative distribution function for the
            # current channel and plot it
            (cdf, bins) = exposure.cumulative_distribution(image[..., j])
            axs[j, i].plot(bins, cdf)
            # set the y-axis label of the current plot to be the name
            # of the current color channel
            axs[j, 0].set_ylabel(color)

    # set the axes titles
    axs[0, 0].set_title("Source")
    axs[0, 1].set_title("Reference")
    axs[0, 2].set_title("Matched")
    # display the output plots
    plt.tight_layout()
    plt.show()


def equalize_both(ref, src):
    ref_eq = exposure.equalize_hist(ref)
    src_eq = exposure.equalize_hist(src)

    return ref_eq, src_eq
