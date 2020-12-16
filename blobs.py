from math import sqrt

import cv2
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import blob_dog, blob_doh, blob_log

import matplotlib.pyplot as plt

image = cv2.imread("./debug/whole.png")
image_gray = rgb2gray(image)

blobs_log = blob_log(image_gray, max_sigma=30, threshold=.075)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.075)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_list = [blobs_log, blobs_dog]
colors = ['yellow', 'lime']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(2, 2, figsize=(8, 9), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image)

    for blob in blobs:
        y, x, r = blob

        f_circle = 10
        circle = plt.Circle((x, y), r + f_circle, color=color, linewidth=2, fill=False)

        # TODO: expand radius by 'threshold', find overlapping circles and remove the smaller

        ax[idx].add_patch(circle)

    ax[idx].set_axis_off()

    ax[idx + 2].set_title("Shapes")
    ax[idx + 2].imshow(image)

    for blob in blobs:
        y, x, r = blob

        f_rectangle = 15
        rectangle = plt.Rectangle((x - r - f_rectangle, y - r - f_rectangle), (r + f_rectangle) * 2, (r + f_rectangle) * 2, color=color, linewidth=2, fill=True)

        # TODO: expand radius by 'threshold', find overlapping circles and remove the smaller
        ax[idx + 2].add_patch(rectangle)

    ax[idx + 2].set_axis_off()

plt.tight_layout()
plt.show()
