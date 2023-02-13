import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from numpy.fft import fft2, fftshift, ifftshift, ifft2
import matplotlib.pyplot as plt


def init_wave_vector(f, z1, z2, px, py, npx, npy):
    # Coefs propagation 'g' & rétro-propagation 'gr' pour le PtP
    dz = z2 - z1  # propagation z2 - z1
    fr = f * 1e6  # f en Hz
    Lambda = 3e8 / fr
    k0 = 2 * np.pi / Lambda
    delta_kx = (2 * np.pi) / (npx * px)  # Pas - echantillonnage dans le domaine spectral
    delta_ky = (2 * np.pi) / (npy * py)
    kx = np.arange(-(npx / 2), (npx / 2), 1) * delta_kx  # Coordonnées dans le domaine spectral
    ky = np.arange(-(npy / 2), (npy / 2), 1) * delta_ky
    [KX, KY] = np.meshgrid(kx, ky)  # Calcul de la constante de propagation suivant z
    kz = csqrt(k0 ** 2 - KY ** 2 - KX ** 2)  # Contrary to np.sqrt only working w/ positive numbers,
                                             # csqrt works with negative ones.
    return kz


def init_propagator(kz, z1, z2, npx, npy, z3):
    # Ici on détérmine la convention de signe utilisée par matlab lorsque l'on prend la racine carree d'un nombre
    # complexe. Si la partie imaginaire est positive on prend le conjugué de la racine. Dans ces conditions, la
    # partie imaginaire donne un terme décroissant dans l'exponentielle compte tenu de la convention choisie (-jkz)
    g = np.zeros((npx, npy), dtype=complex)  # Initialisation matrice contenant terme de propagation: exp(-jkz)
    gr = np.zeros((npx, npy), dtype=complex)  # Initialisation matrice rétro-propagation
    dz = z2 - z1  # propagation z2 - z1
    kz[np.imag(kz) > 0] = np.conj(kz[np.imag(kz) > 0])
    g = np.exp(-1j * kz * dz)  # propagation, dz>0, précautions déjà prises sur kz ci-dessus
    # Pour la rétro-propagation, on ne propage que la partie réelle (on décide partie imag=0):
    gr[np.imag(kz) == 0] = np.exp(-1j * kz[np.imag(kz) == 0] * (-dz))

    if z3 is not None:
        dz3 = z3 - z1  # IDEM pour 3e plan
        g3 = np.zeros((npx, npy))
        g3 = np.exp(-1j * kz * dz3)
        gr3 = np.zeros((npx, npy), dtype=complex)
        gr3[np.imag(kz) == 0] = np.exp(-1j * kz[np.imag(kz) == 0] * (-dz3))
    else:
        gr3 = None

    return g, gr, gr3


def create_support(npx, npy, debug=False):
    # Make an initial Support
    supp = np.zeros((npx, npy))
    supp[npx//4:3*npx//4, npy//4:3*npy//4] = 1
    if debug:
        plt.figure()
        plt.imshow(supp, cmap='jet')
        plt.colorbar()
        plt.show()
    return supp


def init_zeros_phase(npx, npy):
    return np.zeros((npx, npy))


def init_random_phase(npx, npy, debug=False):
    phase_init = np.random.random((npx, npy))
    if debug:
        plt.figure()
        plt.imshow(phase_init, cmap='jet')
        plt.colorbar()
        plt.show()
    return phase_init


def get_coordinates(img, px, py):
    npx, npy = img.shape
    x = np.arange(-(npx - 1) / 2 * px, px * npx / 2, px)
    y = np.arange(-(npy - 1) / 2 * py, py * npy / 2, px)
    return x, y


def plane2plane(f, apmlitudes, z_planes, px, py, npx, npy, phase_simulations, Niter, pp, labels, cmap='jet'):
    if len(apmlitudes) < 2:
        print('Error; Number of planes must be at least 2.')
        return
    elif len(apmlitudes) == 2:
        A1, A2 = apmlitudes
        A3 = None
    elif len(apmlitudes) == 3:
        A1, A2, A3 = apmlitudes
    else:
        print(f'Error; Maximum 3 planes are authorized, but {len(apmlitudes)} were given.')

    if len(z_planes) < 2:
        print('Error; Number of planes must be at least 2.')
        return
    elif len(z_planes) == 2:
        z1, z2 = z_planes
        z3 = None
    elif len(z_planes) == 3:
        z1, z2, z3 = z_planes
    else:
        print(f'Error; Maximum 3 planes are authorized, but {len(z_planes)} were given.')

    if len(labels) == 3:
        Etit1, Etit2, Etit3 = labels
    else:
        Etit1, Etit2, Etit3 = 'E1', 'E2', 'E3'

    if len(phase_simulations) != len(apmlitudes):
        print(f'Error! The number of phase simulations N_simu={len(phase_simulations)} is different from the number of'
              f' planes N_planes={len(apmlitudes)}.')

    # Le pas de la FFT intervient ici - On peut suréchantillonner la FFT pour diminuer le pas en delta_k
    # Dans notre cas il faut appliquer l'opération au power spcetrum, au propagateur et au support
    kz = init_wave_vector(f, z1=z1, z2=z2, px=px, py=py, npx=npx, npy=npy)
    g, gr, gr3 = init_propagator(kz, z1, z2, npx, npy, z3)
    #PhaseIni = init_zeros_phase(npx, npy)
    PhaseIni = init_random_phase(npx, npy, debug=False)

    x, y = get_coordinates(A1, px, py)
    x_simu, y_simu = get_coordinates(phase_simulations[0], px, py)

    # Phase Retrieval Starts!
    # Input data to the algorithm
    # There should be a support for each plane if they were of different size,
    # but here we assume the supports identical
    support = create_support(npx, npy, debug=False)
    S_in = {'support1': A1, 'support2': A2}
    # M_in = {'M_data': M_true}

    Ak = A1 * np.exp(1j * PhaseIni)   # Initialisation du champ plan 1
    Bk = A2 * np.exp(1j * PhaseIni)   # Initialisation du champ plan 2
    N = npx * npy * px * py / 2 / np.pi    # coef de normalisation

    plt.ion()
    fig1 = plt.figure(figsize=(8, 6))
    ax3 = fig1.add_subplot(223)
    emap3 = ax3.imshow(np.angle(phase_simulations[0]), extent=[min(x), max(x), min(y), max(y)],
                       cmap=cmap, interpolation=None)
    ax3.set_title(f'Phase({Etit1}) - direct HFSS', fontsize=12, color='b')
    fig1.colorbar(emap3, ax=ax3)

    ax4 = fig1.add_subplot(224)
    emap4 = ax4.imshow(np.angle(phase_simulations[1]), extent=[min(x), max(x), min(y), max(y)],
                       cmap=cmap, interpolation=None)
    ax4.set_title(f'Phase({Etit2}) - direct HFSS', fontsize=12, color='b')
    fig1.colorbar(emap4, ax=ax4)
    plt.tight_layout()

    itot = 0

    for i in range(0, Niter+1):
        Ak = A1 * np.exp(1j * np.angle(Ak))  # avec amplitude mesurée
        PWspec2 = ifftshift(N * (ifft2(fftshift(Ak))))  # Spectre d'ondes planes...
        PWspec2 = PWspec2 * g     # ...au plan 2
        Bk = fftshift((1/N)*(fft2(ifftshift(PWspec2))))   # éval du champ au plan 2
        Bk = A2 * np.exp(1j * np.angle(Bk))   # avec amplitude mesurée
        PWspec1 = ifftshift(N*(ifft2(fftshift(Bk))))    # Spectre d'ondes planes
        PWspec1 = PWspec1 * gr  # ... au plan 1
        Ak = fftshift((1 / N) * (fft2(ifftshift(PWspec1))))     # éval au plan 1

        if i % pp == 0:
            ax = fig1.add_subplot(221)
            ax.cla()
            emap1 = ax.imshow(np.angle(Ak), extent=[min(x), max(x), min(y), max(y)], cmap=cmap, interpolation=None)
            if i == 0:
                title1 = ax.set_title(f'Phase rec.({Etit1}), iteration {itot + i}', fontsize=12, color='b')
            else:
                title1.set_text(f'Phase rec.({Etit1}), iteration {itot + i}')
            fig1.colorbar(emap1, ax=ax)
            emap1.set_clim(vmin=-np.pi, vmax=np.pi)

            ax2 = fig1.add_subplot(222)
            ax2.cla()
            emap2 = ax2.imshow(np.angle(Bk), extent=[min(x), max(x), min(y), max(y)],
                               cmap=cmap, interpolation=None)
            if i == 0:
                title2 = ax2.set_title(f'Phase rec.({Etit2}), iteration {itot + i}', fontsize=12, color='b')
            else:
                title2.set_text(f'Phase rec.({Etit2}), iteration {itot + i}')
            fig1.colorbar(emap2, ax=ax2)

            plt.tight_layout()
            plt.show()
            plt.pause(0.5)

    itot += i
    plt.ioff()
    plt.show()

    return Ak, Bk