# Simple iterative image reconstruction algorithm
# ML-EM (maximum likelihood - expectation maximisation)
# x^(k+1) = x^k / (A^T 1) * A^T * m / (A * x^k)

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale
import numpy as np
import matplotlib.pyplot as plt


plt.ion()
activity_level = 0.1
true_object = shepp_logan_phantom()
true_object = rescale(activity_level * true_object, 0.5)

fig, axs = plt.subplots(2, 3, figsize=(20, 10))
axs[0, 0].imshow(true_object, cmap='Greys_r')
axs[0, 0].set_title('Object')

# Generate simulated sinogram data
azi_angles = np.linspace(0, 180, 180, endpoint=False)
sinogram = radon(true_object, azi_angles, circle=False)

axs[0, 1].imshow(sinogram.T, cmap='Greys_r')
axs[0, 1].set_title('Sinogram')

mlem_rec = np.ones(true_object.shape)   # iteration 0 (k=0)
sino_ones = np.ones(sinogram.shape)
sens_image = iradon(sino_ones, azi_angles, circle=False, filter_name=None)   # sensitivity image = back projection A^T

for iteration in range(20):
    fp = radon(mlem_rec, azi_angles, circle=False)  # Forward projection of mlem_rec at iteration k A x^k
    ratio = sinogram / (fp + 1e-6)  # ratio = m / (A * x^k)
    correction = iradon(ratio, azi_angles, circle=False, filter_name=None) / sens_image     # A^T * m / (A * x^k)

    axs[1, 0].imshow(mlem_rec, cmap='Greys_r')
    axs[1, 0].set_title('MLEM Recon')
    axs[1, 1].imshow(fp.T, cmap='Greys_r')
    axs[1, 1].set_title('Forward Projection of Recon')
    axs[0, 2].imshow(ratio.T, cmap='Greys_r')
    axs[0, 2].set_title('Ratio Sinogram')
    axs[1, 2].imshow(correction, cmap='Greys_r')
    axs[1, 2].set_title('BP of ratio')

    mlem_rec = mlem_rec * correction
    axs[1, 0].imshow(mlem_rec, cmap='Greys_r')
    axs[1, 0].set_title(f'MLEM Recon It={iteration + 1}')

    plt.show()
    plt.pause(0.5)

plt.show(block=True)
