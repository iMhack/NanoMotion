import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
import skimage.filters
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate

from scipy.fftpack import fft2, fftshift

import cv2

original_image = rgb2gray(cv2.imread("./debug/box_0-100.png"))
#original_rts_image = rgb2gray(cv2.imread("./debug/box_0-199.png"))
original_rts_image = rotate(original_image, 0)

# Now try working in frequency domain
# First, band-pass filter both images

low_sigma = 0.5
high_sigma = 25
image = skimage.filters.difference_of_gaussians(original_image, low_sigma, high_sigma)
rts_image = skimage.filters.difference_of_gaussians(original_rts_image, low_sigma, high_sigma)

# f_radius = 1000
# f_amount = 1000
# image = skimage.filters.unsharp_mask(original_image, radius=f_radius, amount=f_amount)
# rts_image = skimage.filters.unsharp_mask(original_rts_image, radius=f_radius, amount=f_amount)
# image = original_image

# window images
wimage = image * skimage.filters.window('hann', image.shape)
rts_wimage = rts_image * skimage.filters.window('hann', image.shape)

# work with shifted FFT magnitudes
image_fs = np.abs(fftshift(fft2(wimage)))
rts_fs = np.abs(fftshift(fft2(rts_wimage)))

# Create log-polar transformed FFT mag images and register
shape = image_fs.shape
radius = shape[0] // 8  # only take lower frequencies
warped_image_fs = warp_polar(image_fs, radius=radius, output_shape=shape,
                             scaling='log', order=0)
warped_rts_fs = warp_polar(rts_fs, radius=radius, output_shape=shape,
                           scaling='log', order=0)

warped_image_fs = warped_image_fs[:shape[0] // 2, :]  # only use half of FFT
warped_rts_fs = warped_rts_fs[:shape[0] // 2, :]
shifts, error, phasediff = phase_cross_correlation(warped_image_fs,
                                                   warped_rts_fs,
                                                   upsample_factor=400)

# Use translation parameters to calculate rotation and scaling parameters
shiftr, shiftc = shifts[:2]
recovered_angle = (360 / shape[0]) * shiftr
klog = shape[1] / np.log(radius)
shift_scale = np.exp(shiftc / klog)

original_rotated_image = rotate(original_rts_image, -recovered_angle)
rotated_image = rotate(rts_image, -recovered_angle)

original_image_product = np.fft.fft2(original_image) * np.fft.fft2(original_rts_image).conj()
original_cc_image = np.fft.fftshift(np.fft.ifft2(original_image_product))

rotated_image_product = np.fft.fft2(original_image) * np.fft.fft2(rotated_image).conj()
rotated_cc_image = np.fft.fftshift(np.fft.ifft2(rotated_image_product))

fig, axes = plt.subplots(4, 3, figsize=(10, 10))
ax = axes.ravel()
ax[0].set_title("Original base image")
ax[0].imshow(original_image, cmap='gray')
ax[1].set_title("Original modified image")
ax[1].imshow(original_rts_image, cmap='gray')
ax[2].set_title("Original rotated image")
ax[2].imshow(original_rotated_image, cmap='gray')

ax[3].set_title("Base image")
ax[3].imshow(image, cmap='gray')
ax[4].set_title("Modified image")
ax[4].imshow(rts_image, cmap='gray')
ax[5].set_title("Rotated image")
ax[5].imshow(rotated_image, cmap='gray')

ax[6].set_title("Original Image FFT\n(magnitude; zoomed)")
center = np.array(shape) // 2
ax[6].imshow(image_fs[center[0] - radius:center[0] + radius,
                      center[1] - radius:center[1] + radius],
             cmap='magma')
ax[7].set_title("Modified Image FFT\n(magnitude; zoomed)")
ax[7].imshow(rts_fs[center[0] - radius:center[0] + radius,
                    center[1] - radius:center[1] + radius],
             cmap='magma')
ax[8].set_title("Original cross-correlation")
ax[8].imshow(original_cc_image.real)

ax[9].set_title("Log-Polar-Transformed\nOriginal FFT")
ax[9].imshow(warped_image_fs, cmap='magma')
ax[10].set_title("Log-Polar-Transformed\nModified FFT")
ax[10].imshow(warped_rts_fs, cmap='magma')
ax[11].set_title("Rotated cross-correlation")
ax[11].imshow(rotated_cc_image.real)

fig.tight_layout(pad=1)
plt.show()

print("Rotation angle: %f." % (recovered_angle))
print("Scaling factor: %f." % (shift_scale))

shifts, error, phasediff = phase_cross_correlation(original_image,
                                                   original_rts_image,
                                                   upsample_factor=400)

print("Original translation: %s." % (shifts))

image = skimage.filters.difference_of_gaussians(original_image, 0.5, 22)
rts_image = skimage.filters.difference_of_gaussians(original_rts_image, 0.5, 22)

shifts, error, phasediff = phase_cross_correlation(image,
                                                   rts_image,
                                                   upsample_factor=400)

print("Translation: %s." % (shifts))

shifts, error, phasediff = phase_cross_correlation(original_image,
                                                   rotated_image,
                                                   upsample_factor=400)

print("Rotated translation: %s." % (shifts))
