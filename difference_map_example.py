import time
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from skimage import io, color, transform
import numpy as np


# Import Object
Nx = 64
Ny = 64

url = 'images/cat.png'
image = io.imread(url)

if len(image.shape) == 3:
    if image.shape[2] == 4:
        image = image[:, :, 0:3]
    x_true = color.rgb2gray(image)
else:
    x_true = image
x_true = transform.resize(x_true, (Nx, Ny))
x_true /= np.max(x_true)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111)
# im = ax.imshow(x_true)
# plt.colorbar(im)
# ax.set_title('Input image')
# plt.show()

# Now take the Fourier transform of the Input image
# and calculate its Fourier Magnitude
X_true = fftn(x_true)
M_true = np.abs(X_true)

# Make an initial Support
supp = np.zeros((Nx, Ny))
supp[16:48, 16:48] = 1

# Have a look at the magnitude and the support
# fig = plt.figure(figsize=(16, 6))
# ax = fig.add_subplot(121)
# im = ax.imshow(supp)
# plt.colorbar(im)
# ax.set_title("Initial Support")
#
# ax = fig.add_subplot(122)
# im = ax.imshow(fftshift(np.log10(M_true)))
# plt.colorbar(im)
# ax.set_title('Fourier Magnitude data')
# plt.show()


# PROJECTION OPERATORS
def P_S(x, S_in):
    """
    Projection of x onto support S_in
    :param x:
    :type x:
    :param S_in:
    :type S_in:
    :return:
    :rtype:
    """
    x_new = x*S_in['supp']
    return x_new


def P_M(x, M_in):
    """
    Projection of x onto Magnitude.
    """
    X = fftn(x)
    X_new = X/np.abs(X)*M_in['M_data']
    # X_new = M_in['M_data'] * np.exp(1j * np.angle(X))
    x_new = ifftn(X_new)
    return x_new

# THE DIFFERENCE MAP
# "Phase Retrieval by Iterative Projections" (2003) J Opt Soc Ann A, VOl 20, pp40-55
# x_{n+1} = x_n + beta*(P_S*R_M*x_n - P_M*R_S*x_n)
# R_M*x = (1+gamma_M)*P_M*x - gamma_M*x
# R_S*x = (1+gamma_S)*P_S*x - gamma_S*x


def R_M(x, gamma_M, M_in):
    return (1+gamma_M)*P_M(x, M_in) - gamma_M*x


def R_S(x, gamma_S, S_in):
    return (1+gamma_S)*P_S(x, S_in) - gamma_S*x


def DifferenceMap(x, beta, gamma_S, gamma_M, M_in, S_in):
    x_PMRS = P_M(R_S(x, gamma_S, S_in), M_in)
    x_PSRM = P_S(R_M(x, gamma_M, M_in), S_in)   # relaxed projection of x on the support constraint

    x_new = x + beta*(x_PMRS - x_PSRM)
    return x_new, x_PSRM


# Shrinkwrap
C_lp = np.zeros((Nx, Ny))
C_lp[20:44, 20:44] = 1
C_lp = ifftshift(C_lp)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111)
# im = ax.imshow(fftshift(C_lp))
# plt.colorbar(im)
# ax.set_title('fftshift convolution kernel')
# plt.show()


def convolution_filter(x, kernel):
    return ifftn(fftn(x)*kernel)


x_lp = convolution_filter(x_true, C_lp)
X_lp = fftn(x_lp)
M_lp = np.abs(X_lp)

# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(221)
# im = ax.imshow(x_true, clim=[0, 1])
# plt.colorbar(im)
# ax.set_title('x_true')
#
# ax = fig.add_subplot(222)
# im = ax.imshow(np.abs(x_lp), clim=[0, 1])
# plt.colorbar(im)
# ax.set_title('low-pass filtered x_true')
#
# ax = fig.add_subplot(223)
# im = ax.imshow(fftshift(np.log10(M_true)), clim=[0, 2.5])
# plt.colorbar(im)
# ax.set_title('Fourier Magnitude data')
#
# ax = fig.add_subplot(224)
# im = ax.imshow(fftshift(np.log10(M_lp)), clim=[0, 2.5])
# plt.colorbar(im)
# ax.set_title('Fourier Magnitude of LP-filtered data')
# plt.show()

# HIGH PASS FILTER
C_hp = 1 - C_lp

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111)
# im = ax.imshow(fftshift(C_hp))
# plt.colorbar(im)
# ax.set_title('fftshifted kernel')
# plt.show()

x_hp = convolution_filter(x_true, C_hp)
X_hp = fftn(x_hp)
M_hp = np.abs(X_hp)

# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(221)
# im = ax.imshow(x_true, clim=[0, 1])
# plt.colorbar(im)
# ax.set_title('x_true')
#
# ax = fig.add_subplot(222)
# im = ax.imshow(np.abs(x_hp), clim=[0, 1])
# plt.colorbar(im)
# ax.set_title('high-pass filtered x_true')
#
# ax = fig.add_subplot(223)
# im = ax.imshow(fftshift(np.log10(M_true)), clim=[0, 2.5])
# plt.colorbar(im)
# ax.set_title('Fourier Magnitude data')
#
# ax = fig.add_subplot(224)
# im = ax.imshow(fftshift(np.log10(M_hp)), clim=[0, 2.5])
# plt.colorbar(im)
# ax.set_title('Fourier Magnitude of HP-filtered data')
# plt.show()


# Phase Retrieval Starts!
# Input data to the algorithm
S_in = {'supp': supp}
M_in = {'M_data': M_true}

it_max = 61

# CM parameters
beta = 0.99
gamma_M = -1/beta   # Random projections and the optimization of an algorithm for phase
                    # retrieval, J. Phys. A-Math Gen, Vol 36, pp 2995-3007
gamma_S = 1/beta

# Starting iterates
x = np.random.rand(Nx, Ny)

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
plt.ion()

im1 = None
im2 = None

for it in range(it_max):
    # Update rules: Difference Map
    x, x_PS = DifferenceMap(x, beta, gamma_S, gamma_M, M_in, S_in)
    x_sol = x_PS

    # Shrinkwrap
    if it % 10 == 9:
        x_mod = convolution_filter(x_sol, kernel=C_lp)
        x_mod = np.abs(x_mod)
        x_mod /= np.max(x_mod)
        supp = x_mod > 0.12
        S_in = {'supp': supp}

    if it % 2 == 0:
        # Visualize
        if im1 is None:
            im1 = ax1.imshow(np.abs(x_sol))
        else:
            im1.set_data(np.abs(x_sol))
        ax1.set_title('Current Reconstruction it=%d' % it)
        if im2 is None:
            im2 = ax2.imshow(supp)
        else:
            im2.set_data(supp)
        ax2.set_title('Current Support it=%d' % it)

        # plt.gcf()
        # fig.canvas.draw()
        plt.show()
        plt.pause(0.5)

plt.show()

