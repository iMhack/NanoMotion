import cv2
import numpy as np
import skimage.filters
from skimage.color import rgb2gray
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate, warp_polar

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift

angle = 0
original_image = rgb2gray(cv2.imread("./debug/a.png"))
original_rts_image = rotate(original_image, angle)

# Now try working in frequency domain
# First, band-pass filter both images
low_sigma = 0.5
high_sigma = 25
image = skimage.filters.difference_of_gaussians(original_image, low_sigma, high_sigma)
rts_image = skimage.filters.difference_of_gaussians(original_rts_image, low_sigma, high_sigma)

# image = skimage.filters.gaussian(original_image, low_sigma)
# rts_image = skimage.filters.gaussian(original_rts_image, low_sigma)

# f_radius = 1000
# f_amount = 1000
# image = skimage.filters.unsharp_mask(original_image, radius=f_radius, amount=f_amount)
# rts_image = skimage.filters.unsharp_mask(original_rts_image, radius=f_radius, amount=f_amount)
# image = original_image

# window images
original_wimage = original_image * skimage.filters.window('hann', image.shape)
wimage = image * skimage.filters.window('hann', image.shape)
rts_wimage = rts_image * skimage.filters.window('hann', image.shape)

inverted_wimage = skimage.filters.difference_of_gaussians(original_wimage, low_sigma, high_sigma)

# work with shifted FFT magnitudes
original_image_fs = np.abs(fftshift(fft2(original_wimage)))
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

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
ax = axes.ravel()
ax[0].set_title("Original base image")
ax[0].imshow(original_image, cmap='magma')
ax[1].set_title("Original rotated image")
ax[1].imshow(original_rts_image, cmap='gray')
ax[2].set_title("Original windowed image")
ax[2].imshow(original_wimage, cmap='magma')

ax[4].set_title("Filtered base image")
ax[4].imshow(image, cmap='magma')
ax[5].set_title("Filtered rotated image")
ax[5].imshow(rts_image, cmap='gray')
ax[6].set_title("Filtered windowed image")
ax[6].imshow(wimage, cmap='magma')
ax[7].set_title("Filtered windowed image (i)")
ax[7].imshow(inverted_wimage, cmap='magma')

ax[8].set_title("Original image FFT\n(magnitude; zoomed)")
center = np.array(shape) // 2
ax[8].imshow(original_image_fs[center[0] - radius:center[0] + radius,
                               center[1] - radius:center[1] + radius],
             cmap='magma')
ax[9].set_title("Filtered image FFT\n(magnitude; zoomed)")
center = np.array(shape) // 2
ax[9].imshow(image_fs[center[0] - radius:center[0] + radius,
                      center[1] - radius:center[1] + radius],
             cmap='magma')

ax[10].set_title("Original cross-correlation")
ax[10].imshow(original_cc_image.real)
ax[11].set_title("Rotated cross-correlation")
ax[11].imshow(rotated_cc_image.real)

ax[12].set_title("Log-Polar-Transformed\nOriginal FFT")
ax[12].imshow(warped_image_fs, cmap='magma')
ax[13].set_title("Log-Polar-Transformed\nModified FFT")
ax[13].imshow(warped_rts_fs, cmap='magma')

fig.tight_layout(pad=1)
plt.show()

print("Original angle: %f." % (angle))
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
