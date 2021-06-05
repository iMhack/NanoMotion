from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
import matlab
import matlab.engine


image = data.camera()
shift = (-22.4895, 13.23774)

# The shift corresponds to the pixel offset relative to the reference image
offset_image = ndi.shift(image, shift)
print(f"Known offset (row, col): {shift}")

detected_shift, error, phase = phase_cross_correlation(image, offset_image, upsample_factor=100)

print(f"Detected pixel offset (row, col) with Python: {-detected_shift}")

engine = matlab.engine.start_matlab()

reference = matlab.double(np.fft.fft2(image).tolist(), is_complex=True)
moved = matlab.double(np.fft.fft2(offset_image).tolist(), is_complex=True)

output, Greg = engine.dftregistration(reference, moved, 100, nargout=2)
error, phase, row_shift, col_shift = output[0]

print(f"Detected pixel offset (row, col) with Matlab: {-row_shift, -col_shift}")

engine.quit()

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
                               figsize=(8, 3))

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Offset image')


plt.show()
