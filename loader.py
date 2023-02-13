import json
import os
import glob
import time
import numpy as np
import pandas as pd
import pylab
import scipy
from numpy.lib.scimath import sqrt as csqrt
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.fft import fft2, fftshift, ifftshift, ifft2
from numpy.matlib import repmat
from skimage import transform

from courants_equivalents import initial_guess, apply_polar_rules, residuals1, cost_cpl, cost_cpl1
from generate_fvec import f_vec
from downsamples import DownSampleS
from green import GreenFun2D
from plane2plane import plane2plane
import yaml
from progress_bar import progressbar


def load_data(directory, extension="*.txt", file_list=[]):
    """
    Read files with given extension
    :param url:
    :type url:
    :return:
    :rtype:
    """
    directory = directory.replace('/', '\\')
    files = glob.glob(os.path.join(directory, extension))
    data = dict()

    for file in file_list:
        filepath = os.path.join(directory, file).replace('/', '\\')
        if filepath in files:
            try:
                d = np.genfromtxt(filepath, dtype=float, skip_header=2)
                data[file] = d
            except:
                print(f"Impossible to load file {file}.")
    return data


def LectureHFSSplanCPLX(directory, plan, x, y, npx, npy, zSource, plot_image=True, cmap=cm.winter):
    # function[dis, ExR, EyR, EzR, ExI, EyI, EzI] = LectureHFSSplanCPLX(rep, plan, npx, npy, zSource)
    # - HFSS
    # INFORMATION:
    # To create your own variable in the fields calculator.
    # Phase: selecting: E, then
    # ComplexPhase, then
    # Export = > calculate grid points
    # E: ComplexeMag = > same grid
    # ____________________________
    # HFSS Output E on grid is read here
    data = load_data(directory, extension="*", file_list=[plan])
    E = data[plan]

    # size(E)
    xmax = np.max(E[:, 0])
    xmin = np.min(E[:, 0])
    ymax = np.max(E[:, 1])
    ymin = np.min(E[:, 1])
    zplan = np.min(E[:, 2])
    dis = zplan - zSource  # source - plane distance

    E = np.delete(E, np.s_[0:3], axis=1)    # x, y, z removed

    # FOR ComplexMagE file: only 3 components of E
    # Ex[:, 1] = E[:, 1]
    # Ey[:, 1] = E[:, 2]
    # Ez[:, 1] = E[:, 3]

    # FOR E file: 6 components of E
    ExC = np.zeros((len(E), 1), dtype=complex)
    EyC = np.zeros((len(E), 1), dtype=complex)
    EzC = np.zeros((len(E), 1), dtype=complex)
    ExC[:, 0] = E[:, 0] + 1j * E[:, 1]
    EyC[:, 0] = E[:, 2] + 1j * E[:, 3]
    EzC[:, 0] = E[:, 4] + 1j * E[:, 5]

    Etgr = np.sqrt(E[:, 0]**2 + E[:, 2]**2)
    Etgi = np.sqrt(E[:, 1]**2 + E[:, 3]**2)

    Er = np.sqrt(E[:, 0]**2 + E[:, 2]**2 + E[:, 4]**2)
    Ei = np.sqrt(E[:, 1]**2 + E[:, 3]**2 + E[:, 5]**2)

    # Er = np.sqrt(np.real(ExC)**2 + np.real(EyC)**2 + np.real(EzC)**2)
    # Ei = np.sqrt(np.imag(ExC)**2 + np.imag(EyC)**2 + np.imag(EzC)**2)

    EC = Er + 1j * Ei
    EtgC = Etgr + 1j * Etgi

    ExC = np.fliplr(np.reshape(ExC, newshape=(npy, npx))).T
    EyC = np.fliplr(np.reshape(EyC, newshape=(npy, npx))).T
    EzC = np.fliplr(np.reshape(EzC, newshape=(npy, npx))).T

    EC = np.fliplr(np.reshape(EC, newshape=(npy, npx)).T)
    EtgC = np.fliplr(np.reshape(EtgC, newshape=(npy, npx)).T)

    if plot_image:
        fig = plt.figure(figsize=plt.figaspect(0.7))
        fig.suptitle(f'{plan}: Amplitudes des composantes - d = {str(int(dis * 1000))} mm',
                     fontsize=12, color='k')

        ax = fig.add_subplot(221)
        surf = ax.imshow(abs(EC), cmap=cmap)
        ax.set_title('Ampl(E)', fontsize=12)
        fig.colorbar(surf, ax=ax)

        ax2 = fig.add_subplot(222)
        surf2 = ax2.imshow(abs(ExC), cmap=cmap)
        ax2.set_title('Ampl(Ex)', fontsize=12)
        fig.colorbar(surf2, ax=ax2)

        ax3 = fig.add_subplot(223)
        surf3 = ax3.imshow(abs(EyC), cmap=cmap)
        ax3.set_title('Ampl(Ey)', fontsize=12)
        fig.colorbar(surf3, ax=ax3)

        ax4 = fig.add_subplot(224)
        surf4 = ax4.imshow(abs(EzC), cmap=cmap)
        ax4.set_title('Ampl(Ez)', fontsize=12)
        fig.colorbar(surf4, ax=ax4)
        fig.tight_layout()

        fig1 = plt.figure(figsize=plt.figaspect(0.7))
        fig1.suptitle(f'{plan}: Phases des composantes - d = {str(int(dis * 1000))}, mm',
                      fontsize=12, color='k')

        ax = fig1.add_subplot(221)
        surf = ax.imshow(np.angle(EC), cmap=cmap)
        ax.set_title('Phase(E)', fontsize=12)
        fig1.colorbar(surf, ax=ax)
        #
        ax2 = fig1.add_subplot(222)
        surf2 = ax2.imshow(np.angle(ExC), cmap=cmap)
        ax2.set_title('Phase(Ex)', fontsize=15)
        fig1.colorbar(surf2, ax=ax2)

        ax3 = fig1.add_subplot(223)
        surf3 = ax3.imshow(np.angle(EyC), cmap=cmap)
        ax3.set_title('Phase(Ey)', fontsize=15)
        fig1.colorbar(surf3, ax=ax3)

        ax4 = fig1.add_subplot(224)
        surf4 = ax4.imshow(np.angle(EzC), cmap=cmap)
        ax4.set_title('Phase(Ez)', fontsize=15)
        fig1.colorbar(surf4, ax=ax4)
        fig1.tight_layout()
        plt.show()

    return dis, ExC, EyC, EzC, EC, EtgC


def image_padding(img, Npx, Npy, plot_img=True):
    old_image_height, old_image_width = img.shape

    # create new image of desired size and color (blue) for padding
    new_image_width = Npx
    new_image_height = Npy
    result = np.zeros((new_image_height, new_image_width), dtype=img.dtype)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center + old_image_height,
           x_center:x_center + old_image_width] = img
    if plot_img:
        fig = plt.figure(figsize=plt.figaspect(0.7))
        ax = fig.add_subplot(121)
        old = ax.imshow(img, cmap='jet')
        ax.set_title("Original Image")
        plt.colorbar(old, ax=ax)
        ax1 = fig.add_subplot(122)
        new = ax1.imshow(result, cmap='jet')
        ax1.set_title("Padded Image")
        plt.colorbar(new, ax=ax1)
        plt.tight_layout()
        plt.show()
    return result


def get_coordinates(npx, npy, px, py):
    """
    Calculate coordinates associated with an image of size (npx, npy) with resolution (px, py)
    along direction (X, Y) corresponding to the width and height of the image.
    :param npx:
    :type npx:
    :param npy:
    :type npy:
    :param px:
    :type px:
    :param py:
    :type py:
    :return:
    :rtype:
    """
    x = np.arange(-(npx - 1) / 2 * px, px * npx / 2, px)
    y = np.arange(-(npy - 1) / 2 * py, py * npy / 2, px)
    PPM = 1/px
    return x, y, PPM


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y


def correlate2D(a, b):
    """
    Correlation between two 2D-arrays.
    :param a:
    :type a:
    :param b:
    :type b:
    :return:
    :rtype:
    """
    # Calculating mean values
    AM = mean2(a)   # np.mean(a)
    BM = mean2(b)   # np.mean(b)
    # Vectorized versions of c,d,e
    c_vect = (a - AM) * (b - BM)
    d_vect = (a - AM)**2
    e_vect = (b - BM)**2

    # Finally get r using those vectorized versions
    r_out = np.sum(c_vect) / float(np.sqrt(np.sum(d_vect) * np.sum(e_vect)))
    return r_out


def save_dict_as_json(dico, filename):
    # Convert dictionary
    try:
        dico1 = dict()
        for key, item in dico.items():
            if isinstance(item, np.ndarray) and len(item.shape) == 2:
                dico1[key] = np.asarray([str(val) for row in item for val in row]).tolist()
            elif isinstance(item, np.ndarray) and len(item.shape) == 1:
                dico1[key] = np.asarray([str(val) for val in item]).tolist()
            elif isinstance(item, float):
                dico1[key] = str(item)

        with open(filename, 'w') as f:
            json.dump(dico1, f)
        print("File saved successful.")
    except EOFError:
        print("Impossible to save file.")


def load_json_dict(filename):
    # Open the file and load the file
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            for key, item in data.items():
                if isinstance(item, list):
                    data[key] = np.asarray(item, dtype=float)
                elif isinstance(item, str):
                    data[key] = float(item)
            if 'real_CP' in data.keys() and 'imag_CP' in data.keys():
                data['CP'] = data['real_CP'] + 1j*data['imag_CP']
        return data
    except ImportError:
        print('Impossible to load file.')
        return None


if __name__ == "__main__":

    # Import Images
    directory = 'images/CornetKu18_15GHz'
    zSource = 0.13
    npx = 201     # nbre points selon l'axe Y(horizontal)
    npy = 201   # nbre points selon l'axe Z (vertical)
    polar = 'V'
    px = 0.001   # taille pixel
    py = px
    f = 15000    # fréquence en MHz
    fic1 = 'CornetKu18_15GHz_z139'  # fic sortie HFSS plan 1
    fic2 = 'CornetKu18_15GHz_z150'  # fic sortie HFSS plan 2
    fic3 = 'CornetKu18_15GHz_z200'  # fic sortie HFSS plan 3
    Nit = 300    # Nbre itération PtP
    x = np.arange(-(npx-1)/2*px, px*npx/2, px)
    y = np.arange(-(npy-1)/2*py, py*npy/2, px)
    x1 = x
    y1 = y
    PPM = 1/px
    d1, Ex1, Ey1, Ez1, E1, Etg1 = LectureHFSSplanCPLX(directory, fic1, x, y, npx, npy, zSource,
                                                      plot_image=False, cmap='jet')
    d2, Ex2, Ey2, Ez2, E2, Etg2 = LectureHFSSplanCPLX(directory, fic2, x, y, npx, npy, zSource,
                                                      plot_image=False, cmap='jet')
    d3, Ex3, Ey3, Ez3, E3, Etg3 = LectureHFSSplanCPLX(directory, fic3, x, y, npx, npy, zSource,
                                                      plot_image=False, cmap='jet')
    z1 = float(fic1.split("_z")[1]) / 1000
    z2 = float(fic2.split("_z")[1]) / 1000
    z3 = float(fic3.split("_z")[1]) / 1000
    Etit1 = 'E1'
    Etit2 = 'E2'
    Etit3 = 'E3'

    tangential_only = True  # True = only tangential field (Thermographie); False = vectorial component
    if tangential_only:
        E1, E2, E3 = [Etg1, Etg2, Etg3]
        Etit1, Etit2, Etit3 = ['Etg1', 'Etg2', 'Etg3']
    else:   # only V field
        E1, E2, E3 = [Ey1, Ey2, Ey3]
        Etit1, Etit2, Etit3 = ['Ey1', 'Ey2', 'Ey3']

    A1, A2, A3 = [abs(E1), abs(E2), abs(E3)]

    # Préparation FFT et binning des images (DownSample)
    fs = npx / (max(x) - min(x))     # échantillonage = PPM
    [f_x, df_x, idx_0_x] = f_vec(fs, npx)    # x space frequency vector
    [f_y, df_y, idx_0_y] = f_vec(fs, npy)    # y space frequency vector
    fmin, fmax = min(f_x), max(f_x)
    A1f = fftshift(fft2(A1))
    A2f = fftshift(fft2(A2))
    A3f = fftshift(fft2(A3))

    # Suppression des dernières ligne et colonne => dimensions paires indispensables pour les FFT
    A1 = A1[0:npx - 1, 0:npy - 1]
    A2 = A2[0:npx - 1, 0:npy - 1]
    A3 = A3[0:npx - 1, 0:npy - 1]

    npx, npy = A1.shape
    x = x[0:-1]
    y = y[0:-1]

    # Downsample
    DwnSampleS = 1
    if DwnSampleS > 1:
        [A1, npx, npy] = DownSampleS(A1, DwnSampleS)
        [A2, npx, npy] = DownSampleS(A2, DwnSampleS)
        [A3, npx, npy] = DownSampleS(A3, DwnSampleS)
        x = x[::DwnSampleS]
        y = y[::DwnSampleS]
        PPM = PPM / DwnSampleS
        px *= DwnSampleS

    # Resize image - alternative way
    resize_image = True
    do_padding = False
    normalize = False

    if resize_image:
        grandissement = 5
        Nx, Ny = npx//grandissement, npy//grandissement
        #grandissement = npx/Nx
        A1 = transform.resize(A1, (Nx, Ny), anti_aliasing=True)
        A2 = transform.resize(A2, (Nx, Ny), anti_aliasing=True)
        npx, npy = A1.shape
        px *= grandissement
        py = px
        x, y, PPM = get_coordinates(npx, npy, px, py)
    if do_padding:
        A1 = image_padding(A1, 2*npx, 2*npy, plot_img=False)
        A2 = image_padding(A2, 2*npx, 2*npy, plot_img=False)
        npx, npy = A1.shape

    if normalize:
        A1max = np.max(A1)
        A2max = np.max(A2)
        A1 /= A1max
        A2 /= A2max

    #------------------------------------------------------------
    # Algorithme PtP
    #------------------------------------------------------------
    # Usuel avec 2 Plans
    Result = None
    plane_to_plane = False
    if plane_to_plane:
        apmlitudes = [A1, A2]
        z_planes = [z1, z2]
        labels = [Etit1, Etit2, Etit3]

        Ak, Bk = plane2plane(f, apmlitudes, z_planes, px, py, npx, npy, phase_simulations=[Ey1, Ey2],
                             Niter=300, pp=10, labels=labels, cmap='jet')

        # Sortie et sauvegarde pour éventuelle utilisation ultérieure
        # de la cartographie reconstituée (E complexe)
        Result = dict()
        # Result['CP'] = Ak
        Result['real_CP1'] = np.real(Ak)
        Result['imag_CP1'] = np.imag(Ak)
        Result['real_CP2'] = np.real(Bk)
        Result['imag_CP2'] = np.imag(Bk)
        Result['x'] = x
        Result['y'] = y
        Result['delta_x'] = max(x) - min(x)
        Result['delta_y'] = max(y) - min(y)
        Result['pixel'] = px

        save_p2p = True
        if save_p2p:
            # for key, value in Result.items():
            #     if isinstance(value, np.ndarray):
            #         Result[key] = value.tolist()
            #     elif isinstance(value, (np.float, np.float64)):
            #         Result[key] = str(value)
            save_directory = 'images/P2P'
            savename = f'CplxField_z= {z1 * 1000} mm-f= {f} MHz-pixel= {px * 1000} mm'
            FicNameEPtP = os.path.join(os.getcwd(), save_directory, savename).replace('/', '\\')

            # Save as yaml
            # ff = open(FicNameEPtP + '.yaml', 'w+', encoding='utf8')
            # yaml.dump(Result, ff, default_flow_style=False, allow_unicode=True)    # Carto reconstituée pour exploitation ultérieure
            # Save as JSON
            save_dict_as_json(Result, FicNameEPtP + '.json')

    #----------------------------------------------------------------
    # Surface de Huyghens reconstitué à partir de la cartographie PtP
    #----------------------------------------------------------------
    # Dimensions Surface Huyghens
    # Spécifications de convergence pour LSQR
    Xs = 0.075   # taille surface Huyghens en x (environ ouverture de l'antenne)
    Ys = 0.075   # taille surface Huyghens en y
    LStol = 1e-8   # tolérance pour convergence méhode moindres carrés
    LSmaxit = 20000     # nombre max d'itérations
    zSource = z1 - zSource  # HS décalée sur l'ouverture
    w = 2 * np.pi * f*1e6  # pulsation
    mu0 = 4 * np.pi * 1e-7
    NxSources = 12    # Nombre de courants équivalents dans
                        # la direction x: DOIT ETRE PAIR
    Lambda = 3e8 / (f * 1e6)
    k0 = 2 * np.pi / Lambda

    save_directory = 'images/P2P'
    savename = f'CplxField_z= 139.0 mm-f= 15000 MHz-pixel= 5.0 mm.json'
    FicNameEPtP = os.path.join(os.getcwd(), save_directory, savename).replace('/', '\\')
    Result = load_json_dict(FicNameEPtP)
    if Result is None:
        Eobs = A1
        DimX = max(x) - min(x)
        DimY = max(y) - min(y)
    else:
        Eobs = Result['CP']
        npx, npy = len(Result['x']), len(Result['y'])
        Eobs = np.reshape(Eobs, (npx, npy))
        DimX = Result['delta_x']
        DimY = Result['delta_y']

    #DwnSampleS = int(len(x) / Eobs.shape[0])  # Coefficient de binning sur l'image (plusieurs pour comparaisons)
    DownSampleS = [1]

    if DwnSampleS > 1:
        # Eobs, npx, npy = DownSampleS(Eobs, DwnSampleS)
        #DwnSampleS = DwnSampleS[0]
        Eobs = transform.downscale_local_mean(Eobs, (DwnSampleS, DwnSampleS))
        npx, npy = Eobs.shape
        x = x[0::DwnSampleS]
        y = y[0::DwnSampleS]
        px *= DwnSampleS
        py *= DwnSampleS

    diag_ray_HFSS = True
    if diag_ray_HFSS:
        RPdispo = 1     # lecture diagramme de rayonnement direct si disponible
        directRP_file = os.path.join(os.getcwd(), directory, 'CornetKuRadPatternHFSS_15GHz.csv')\
            .replace('/', '\\')
        directRP = pd.read_csv(directRP_file, sep=';', header=None, skip_blank_lines=True,
                         usecols=range(3))

    # Réarrangement image Nx x Ny => vecteur colonne 1 x Nx.Ny
    Nx, Ny = Eobs.shape
    Nobs = Nx * Ny  # nombre de "mesures" de champ
    Em = np.reshape(Eobs, (Nobs, 1))     # CP sous forme d'un vecteur: les colonnes de E
                                    # sont les unes sous les autres dans Em
    #Ym = np.zeros((Nx, Ny))
    Ym = repmat(y, Nx, 1)
    Xm = repmat(x, Ny, 1)    # Xm et Ym coordonnées des valeurs de CP dans Em => vecteurs
                               # à Nx.Ny valeurs
    Xm = np.reshape(Xm, (Nx * Ny, 1))
    Ym = np.reshape(Xm, (Nx * Ny, 1))

    Zm = np.full((Nx*Ny, 1), zSource)
    r_obs = np.concatenate((Xm, Ym, Zm), axis=1).T      # coordonnées champ observés

    # Plot Courants sur la Surface 2D
    plot_surface = False
    if plot_surface:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        eq_current_map = ax.imshow(abs(Eobs), extent=[min(x), max(x), min(y), max(y)], cmap='jet')
        fig.colorbar(eq_current_map, ax=ax)
        ax.set_title(f'Amplitude CP reconstruit (en z = {1000*zSource:.0f} mm - {Nx * Ny:.0f} valeurs')
        ax.axhline(-DimX/2, color='m', linewidth=1)
        ax.axhline(DimX/2, color='m', linewidth=1)
        ax.axvline(-DimY/2, color='m', linewidth=1)
        ax.axvline(DimY/2, color='m', linewidth=1)
        plt.tight_layout()
        plt.show()

    # Discrétisation de la surface rayonnante équivalente,
    # sur laquelle on place les courants équivalents J
    nxs = NxSources     # nombre selon x DOIT ETRE PAIR
    dx = Xs / nxs   # pas - discrétisation spatiale des J sur surface Huyghens
    nysF = np.floor(Ys / Xs * nxs)     # nombre selon y (cellules carrées)
    nysC = np.ceil(Ys / Xs * nxs)   # nombre selon y(cellules carrées)
    if nysF % 2 == 0:     # nys DOIT ETRE PAIR
        nys = int(nysF)
    else:
        nys = int(nysC)
    dy = dx
    Xs = nxs * dx   # surface ajustée
    Ys = nys * dx
    posx = - np.round(nxs/2) * dx + dx / 2 + np.arange(0, nxs)*dx
    posy = - np.round(nys/2) * dy + dy / 2 + np.arange(0, nys)*dy

    Ns = int(nxs * nys)
    Rsou = np.zeros((Ns, 2))   # Points surface équivalente (xk, yk) pour k de 1 à nxs.nys -
                                # NB zk = 0 par définition
    for i in range(0, nxs):
        for j in range(0, nys):
            Rsou[(i*nys) + j, :] = np.array([posx[i], posy[j]])
    Rsoux = np.repeat(posx, nxs)
    Rsouy = np.matlib.repmat(posy, 1, nxs)[0, :]
    r_source = np.zeros((3, Ns))     # coordonnées de l'ensemble des points sources où on place les J
                                     # r_source(1,:)=Rsou(:,1); r_source(2,:)=Rsou(:,2);
    r_source[0, :] = Rsoux
    r_source[1, :] = Rsouy

    #------------------------------------------------------------------------------------------
    # Calcul de la matrice de Green G et résolution de GJ=Eobs (moindres carrés) pour obtenir J
    # - Pas de Jz ni de Ez => G a 2Nobs lignes, 2Ns colonnes) : G2D.Jxy = Exy
    # - E : vecteur complexe du champ : 2Nobs lignes - V or H polar (Ex = 0 ou Ey = 0)
    # - J : vecteur inconnu des courants équivalents : 2Ns lignes (Jx, Jy, Jx2,Jy2,...JxNs,JyNs)
    # Calcul de la matrice: G=GreenFun2D( k, r_obs, r_sou ) avec :
    # - R=norm(r_sou-r_obs); % distance source-observable
    # - Rhat=(r_obs-r_sou)/R ; % vecteur source-observable
    # - DyadRhat=(Rhat*Rhat') ; % produit dyadique
    # - G_scal=exp(-1i*k*R)/(4*pi*R); % fonction de Green scalaire
    # - G = mu0*k*3e8*( ( 3/((k*R)^2) + 3*1i/(k*R) - 1 )*DyadRhat + ( 1 - 1i/(k*R) - 1/(k*R)^2 )*eye(3) )*G_scal;
    # - G(:,3)=[];G(3,:)=[]; % on supprime dernières lignes et colonnes (z)

    EmXY = apply_polar_rules(Nobs, Em, polar=polar)

    # Calcul de la fonction de Green contenant seulement les composantes x et y tangentielles pour le calcul du champ
    # lointain
    G2D = np.zeros((2*Nobs, 2*Ns), dtype=complex)
    #for i in range(0, Nobs):
    for i in progressbar(range(Nobs), f"Green Matrix Computation: {2 * Nobs} x {2 * Ns}", 40):
        Mxy = np.zeros((2, 2 * Ns), dtype=complex)
        for j in range(0, Ns):
            Gs = GreenFun2D(k0, r_obs[:, i], r_source[:, j])    # Matrix (2 x 2) here because we removed z components
            Mxy[:, 2*j:2*j+2] = Gs
        G2D[2*i:2*i+2, :] = Mxy

    # Conditionnement de la matrice
    # deltab = 0    # incertitude on measurement
    # measure_relative_error = np.linalg.norm(deltab)/np.linalg.norm(b)
    sigma_propagator = np.linalg.svd(G2D)[1]
    conditionnement_propagator = max(sigma_propagator)/min(sigma_propagator)


    method = [4]
    x_initial = initial_guess(Ns=Ns, mode="zeros")
    init = np.zeros((2 * Ns, 1), dtype=float)
    if 1 in method:
        # Method 1: only real solution
        J2D, success = scipy.optimize.leastsq(cost_cpl, init, args=(G2D, EmXY))
    elif 2 in method:
        # Method 2: treat complex as real
        sol, success = scipy.optimize.leastsq(cost_cpl1, x_initial.view(np.double),
                                              args=(G2D.view(np.double), EmXY.view(np.double)))
        J2D = sol.view(np.complex128)
    elif 3 in method:
        # Method 3: treat complex numbers
        J2D, res, rnk, sv = pylab.lstsq(G2D, EmXY, rcond=None)
    elif 4 in method:
        # Method 4 -
        sol2 = scipy.optimize.lsq_linear(A=G2D, b=EmXY[:, 0], max_iter=LSmaxit, verbose=1)
        J2D = sol2['x']
        itn = sol2['nit']
    elif 5 in method:
        # Method 5 - send ComplexWarning
        sol1 = scipy.sparse.linalg.lsqr(A=G2D, b=EmXY, atol=LStol, btol=LStol, iter_lim=LSmaxit,
                                        x0=x_initial, show=True)
        [J2D, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var] = sol1
        # istop: 1 = x is an approximate solution obeying the tolerance limits;
        #        2 = istop indicates that the system is inconsistent and thus x is rather an approximate solution to
        #        3 = The estimate of cond(Abar) has exceeded conlim.
        #        7 = The iteration limit has been reached.
        #            the corresponding least-squares problem.
        # itn = Iteration number upon termination.
        # r1norm = norm(r), where r = b - Ax.
        # r2norm = sqrt( norm(r)^2  +  damp^2 * norm(x - x0)^2 ). Equal to r1norm if damp == 0.
        # anorm = Estimate of Frobenius norm of Abar = [[A]; [damp*I]].
        # acond = Estimate of cond(Abar).
        # arnorm = Estimate of norm(A'@r - damp^2*(x - x0)).
        # xnorm = norm(x)
    plt.figure()
    plt.plot()

    E_GxJ = G2D.dot(J2D)   # Champ reconstitué = G2D.J2D, en vecteur colonne longueur(2.Nobs) amplitude
                        # amplitude des Ex et Ey (au lieu de Ex complexe et Ey complexe)
    E_GxJ_amp = (E_GxJ * np.conj(E_GxJ))**0.5
    Eret = np.zeros((Nobs,), dtype=complex)
    Eret = (E_GxJ_amp[::2]**2 + E_GxJ_amp[1::2]**2)**0.5
    # Amplitude of the field must be a real number. We only checked that.
    if np.all(np.imag(Eret) == 0):
        Eret = np.real(Eret)
    else:
        print("E amplitude is not fully real.")

    EretG2D = np.reshape(Eret, (Nx, Ny))    # Amplitude du champ reconstitué, en matrice(Nx x Ny)
    J2Dabs = np.abs(J2D)
    J2DX = np.zeros((Ns,), dtype=complex)
    J2DY = np.zeros((Ns,), dtype=complex)
    J2DX = J2D[::2]
    J2DY = J2D[1::2]

    #Rco = scipy.signal.correlate2d(abs(Eobs), EretG2D)   # corrélation champ reconstruit / champ original
    Rco = correlate2D(a=abs(Eobs), b=EretG2D)

    savefig = False
    fig_Jeq = plt.figure()
    plt.plot(Rsoux, Rsouy, 'o', fillstyle=None, color=[0.2, 0.2, 0.2])
    plt.xlim([-Xs/2*1.1, Xs/2*1.1])
    plt.ylim([-Ys/2*1.1, Ys/2*1.1])
    # plt.axis('equal')
    #plt.title(f'Allure des {2 * nxs * nys} Jeq ({itn} it. -R ={Rco:.5f}) - Nobs={Nobs}')
    plt.title(f'Allure des {2 * nxs * nys} Jeq (R ={Rco:.5f}) - Nobs={Nobs}')
    #plt.axis([-Xs * 1.1 / 2, Xs * 1.1 / 2, - Ys * 1.1 / 2, Ys * 1.1 / 2])
    text_prop = dict(backgroundcolor=[.65, .65, .95], fontsize=11,
                     bbox=dict(fc="m", ec="k", lw=2))
    plt.text(posx[0], posy[0], f'{nxs * nys} Jeq - {Nobs} Eobs \ndx = {dx / Lambda:.4f} lambda',
             **text_prop)
    plt.axhline(-Xs / 2, color='b', linewidth=1)
    plt.axhline(Xs / 2, color='b', linewidth=1)
    plt.axvline(-Ys / 2, color='b', linewidth=1)
    plt.axvline(Ys / 2, color='b', linewidth=1)

    plt.quiver(Rsoux, Rsouy, np.abs(J2DX), np.zeros((Ns,)), linewidth=0.5, color='r')
    plt.quiver(Rsoux, Rsouy, np.zeros((Ns,)), np.abs(J2DY), linewidth=0.5, color='b')
    #plt.tight_layout()
    plt.show()

    if savefig:
        FicName = fic1
        strout0 = os.path.join(directory, FicName +
                               f'_2DGrMat_HuygSurf={Xs * 1000:.0f}_x_{Ys * 1000:.0f}' +
                               f'mm_After_{itn}_iter-{Nobs}_Eobs-{Ns}_Js').replace('/', '\\')
        strout0 += ".png"
        plt.savefig(strout0)

    #------------------------------------------------------------------------------------------


