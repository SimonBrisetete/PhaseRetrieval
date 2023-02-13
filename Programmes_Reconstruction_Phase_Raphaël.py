#!/usr/bin/env python
# coding: utf-8

# # La méthode du Transport d'Intensité
# La méthode de transport d'intensité est une technique utilisée en traitement d'image pour
# reconstruire la phase d'une image à partir de sa fréquence de diffraction. Elle repose sur
# le fait que, dans une image de diffraction, l'intensité de chaque pixel est directement liée à la
# phase de l'onde qui a généré cette intensité. En utilisant cette propriété, on peut reconstruire la
# phase de l'image en résolvant un système d'équations qui relie l'intensité de chaque pixel à sa
# phase correspondante.
# Pour utiliser la méthode de transport d'intensité, on commence par appliquer un filtre de diffraction
# à l'image d'origine, généralement à l'aide de la transformée de Fourier rapide (FFT).
# Cela permet de passer de l'espace de l'image à l'espace de fréquence, où chaque pixel est associé à
# une fréquence spécifique. Ensuite, on résout un système d'équations qui relie l'intensité de chaque pixel
# à sa phase correspondante en utilisant une méthode de résolution de systèmes d'équations telle que
# l'algorithme de Jacobi ou de Gauss-Seidel. Une fois le système résolu, on obtient une image de phase
# reconstruite qui peut être utilisée pour reconstruire l'image d'origine en utilisant la transformée
# de Fourier inverse (IFFT).
# 
# La méthode de transport d'intensité est souvent utilisée dans les domaines de la microscopie optique
# et de la diffraction des rayons X, où il est souvent difficile de mesurer directement la phase
# de l'onde. Elle peut également être utilisée dans d'autres domaines où la phase de l'onde est importante,
# tels que la communication optique et la cryptographie.

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def jacobi(fft_image, max_iter=100, tol=1e-6):
    # Récupération des dimensions de l'image
    rows, cols = fft_image.shape
    
    # Initialisation de la phase reconstruite à zéro
    phase = np.zeros_like(fft_image)
    
    # Boucle sur les itérations
    for i in range(max_iter):
        # Copie de la phase reconstruite précédente
        phase_prev = phase.copy()
        
        # Boucle sur les pixels de l'image
        for x in range(1, rows-1):
            for y in range(1, cols-1):
                # Mise à jour de la phase du pixel courant en utilisant la formule de l'algorithme de Jacobi
                phase[x, y] = (fft_image[x, y] - phase[x-1, y] - phase[x, y-1] - phase[x+1, y] - phase[x, y+1]) / 4
        
        # Calcul de la différence entre la phase reconstruite précédente et la nouvelle phase reconstruite
        diff = np.abs(phase - phase_prev)
        
        # Si la différence est inférieure à la tolérance, on sort de la boucle
        if np.max(diff) < tol:
            break
    
    # Renvoi de la phase reconstruite
    return phase


# Cette fonction prend en entrée l'image de diffraction au format FFT (`fft_image`) et des paramètres
# optionnels `max_iter` et `tol` qui déterminent respectivement le nombre maximum d'itérations et la
# tolérance utilisées par l'algorithme de Jacobi. Elle retourne la phase reconstruite de l'image sous
# forme de tableau NumPy.

# Chargement de l'image au format PNG
image = Image.open('cat.png')
# Conversion de l'image en niveaux de gris
image_gray = image.convert('L')
# Conversion de l'image en tableau NumPy
image_array = np.array(image_gray)
# Application du filtre de diffraction à l'image
image_fft = np.fft.fft2(image_array)
# Résolution du système d'équations de transport d'intensité
# Pour cet exemple, nous utilisons l'algorithme de Jacobi
image_phase = jacobi(image_fft)

# Appliquer la transformée de Fourier inverse à la phase reconstruite
image_reconstructed = np.fft.ifft2(image_phase)

# Conversion de la phase reconstruite en image au format PNG
image_reconstructed = Image.fromarray(np.uint8(image_reconstructed))
##image_reconstructed.save('image_reconstructed.png')

# Afficher la phase reconstruite
plt.imshow(image_reconstructed, cmap='gray')
plt.show()

# -------------------------------------------------------------------------------------------------------------------------------
# # La méthode de Gerchberg-Saxton
# La méthode de Gerchberg-Saxton est une méthode iterative utilisée pour la reconstruction de la phase
# d'une image à partir de sa modulo de l'amplitude. Elle a été initialement développée pour la correction
# de distorsions optiques, mais elle est également utilisée dans d'autres domaines, tels que la microscopie
# et l'imagerie médicale.
# 
# L'algorithme de Gerchberg-Saxton se déroule en plusieurs étapes :
# 
# 1. Un masque de phase aléatoire est appliqué à l'image originale pour produire une image masquée.
# Le masque de phase est une image complexe qui contient des informations sur la phase de chaque pixel de
# l'image.
# 
# 2. La transformée de Fourier de l'image masquée est calculée. La transformée de Fourier est une
# représentation de l'image dans le domaine de fréquence, qui permet de mettre en évidence les différentes
# composantes de l'image en fonction de leur fréquence spatiale.
# 3. L'image est reconstruite à partir de la transformée de Fourier de l'image masquée en utilisant
# l'algorithme de transformée de Fourier inverse. Cette étape permet de retrouver une image approximative
# de l'image originale, mais la phase de chaque pixel est perdue.
# 
# 4. La transformée de Fourier de l'image reconstruite est calculée.
# 
# 5. Le masque de phase est appliqué à la transformée de Fourier de l'image reconstruite pour produire une
# image reconstruite masquée.
# 
# 6. La transformée de Fourier inverse de l'image reconstruite masquée est calculée pour produire une
# nouvelle image reconstruite.
# 
# 7. L'image masquée est mise à jour en utilisant l'image reconstruite masquée.
# 
# 8. Les étapes 3 à 7 sont répétées jusqu'à ce que l'image reconstruite converge vers l'image originale.
# 
# La méthode de Gerchberg-Saxton est une méthode itérative qui peut être utilisée pour reconstruire
# la phase d'une image à partir de sa modulo de l'amplitude, en utilisant des transformations de Fourier
# et en itérant entre la reconstruction de l'image et l'application du masque de phase. Elle est souvent
# utilisée pour corriger les distorsions optiques et dans d'autres domaines où la phase est importante.

# ## Version 1
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt


def gerchberg_saxton(target_amplitude, initial_guess, max_iter=100, tol=1e-6):
    """
    Reconstruit la phase d'une image en utilisant l'algorithme de Gerchberg-Saxton et l'optimisation de Wirtinger Flow.
    
    Paramètres:
    - target_amplitude (tableau 2D numpy) : Spectre d'amplitude cible de l'image.
    - initial_guess (tableau 2D numpy) : Estimation initiale de la phase de l'image.
    - max_iter (int, facultatif) : Nombre maximum d'itérations à effectuer.
    - tol (float, facultatif) : Tolérance pour le critère de convergence.

    Retourne :
    - tableau 2D numpy : La phase reconstruite de l'image.
    """
    # Calcul de la FFT de l'estimation initiale
    initial_fft = scipy.fftpack.fft2(initial_guess)
    
    # Mise à l'erreur initiale à sa valeur maximale
    error = np.inf
    
    # Itération jusqu'à ce que l'erreur soit inférieure à la tolérance ou que le nombre maximum d'itérations
    # soit atteint
    for i in range(max_iter):
        # Calcul de la FFT inverse de l'estimation actuelle
        current_guess = scipy.fftpack.ifft2(initial_fft)
        
        # Mise à la phase de l'estimation actuelle au spectre d'amplitude cible
        current_fft = target_amplitude * np.exp(1j * np.angle(current_guess))
        
        # Calcul de l'erreur entre l'estimation actuelle et l'estimation initiale
        error = np.abs(current_fft - initial_fft).mean()
        
        # Mise à jour de l'estimation initiale avec l'estimation actuelle
        initial_fft = current_fft
        
        # Vérification que l'erreur est inférieure à la tolérance
        if error < tol:
            break
    
    # Retour de la phase reconstruite
    return np.angle(current_fft)

# Chargement de l'image et conversion en nuance de gris
image = plt.imread('cat.png')
gray_image = np.mean(image, axis=2)

# Calcul de la FFT de l'image en nuance de gris
fft = scipy.fftpack.fft2(gray_image)

# Reconstruction de la phase en utilisant Gerchberg-Saxton et Wirtinger Flow
reconstructed_phase = gerchberg_saxton(np.abs(fft), np.zeros_like(gray_image))

# Afficher la phase reconstruite
plt.imshow(reconstructed_phase, cmap='gray')
plt.show()


# ## Version 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift

# Chargez l'image au format PNG et convertissez-la en niveaux de gris
image = plt.imread('cat.png').mean(axis=2)
# Créez un masque de phase aléatoire
random_phase = np.exp(1j*np.random.rand(image.shape[0], image.shape[1]))
# Appliquer le masque de phase à l'image
image_masked = image * random_phase
# Calculer la transformée de Fourier de l'image masquée
fft_image_masked = fft2(image_masked)
# Définissez un nombre maximum d'itérations
max_iter = 2

# Exécutez la boucle de Gerchberg-Saxton
for i in range(max_iter):
    # Calculer l'image reconstruite en utilisant la transformée de Fourier de l'image masquée
    reconstructed_image = np.abs(ifft2(fft_image_masked))
    # Calculer la transformée de Fourier de l'image reconstruite
    fft_reconstructed_image = fft2(reconstructed_image)
    # Appliquer le masque de phase à la transformée de Fourier de l'image reconstruite
    fft_reconstructed_image_masked = fft_reconstructed_image * random_phase
    # Calculer la transformée de Fourier inverse de l'image reconstruite masquée
    reconstructed_image_masked = np.abs(ifft2(fft_reconstructed_image_masked))
    # Mettre à jour l'image masquée en utilisant l'image reconstruite masquée
    image_masked = reconstructed_image_masked * random_phase

print(f"nbr iterations = {i}")
# Afficher l'image reconstruite finale
plt.imshow(reconstructed_image, cmap='gray')
plt.show()


# -------------------------------------------------------------------------------------------------------------------------------

# # La méthode Hybrid Input-Output
# La méthode Hybrid Input Output (HIO) est une technique de reconstruction de la phase utilisée en
# imagerie de diffraction des électrons (ED) et en imagerie optique diffractive (DOI). Elle a été
# proposée par Fienup en 1982 et est devenue l'une des méthodes de reconstruction de la phase les plus
# utilisées en ED et en DOI.
# 
# En ED et en DOI, l'objectif est de reconstruire l'image de l'objet à partir de mesures de diffraction
# de l'objet. L'image de l'objet peut être décrite par sa fonction de distribution de densité de
# charge (CDD), qui est liée à la fonction de diffraction de l'objet par la loi de Fourier.
# La reconstruction de la phase consiste donc à retrouver la CDD à partir de la fonction de diffraction
# mesurée.
# 
# La méthode HIO consiste à itérativement mettre à jour la CDD en utilisant la fonction de diffraction
# mesurée et en vérifiant si la CDD ainsi obtenue satisfait la condition de conservation de la charge.
# Si c'est le cas, la CDD est conservée et la méthode passe à l'itération suivante. Si ce n'est pas le cas,
# la CDD est modifiée de manière à satisfaire la condition de conservation de la charge.
# 
# Voici un résumé des étapes de la méthode HIO:
# 
# 1. Initialiser la CDD reconstruite à zéro.
# 2. Définir un paramètre de mise à jour beta (généralement compris entre 0 et 1).
# 3. Définir un critère d'arrêt, par exemple en utilisant la différence entre la CDD reconstruite à
# l'itération précédente et la CDD reconstruite à l'itération courante.
# 4. Boucler jusqu'à ce que le critère d'arrêt soit atteint:
# 5. Calculer la transformée de Fourier de la CDD reconstruite.
# 6. Mettre à jour la transformée de Fourier en utilisant la formule suivante:
# $F_{new} = (1 - \beta) \cdot F_{old} + \beta \cdot F_{measured} \cdot \exp(1j \cdot \angle(F_{reconstructed}))$
# 7. Calculer la nouvelle CDD reconstruite en utilisant la transformée de Fourier mise à jour.
# 8. Vérifier si la CDD reconstruite satisfait la condition de conservation de la charge. Si ce n'est pas
# le cas, la CDD est modifiée de manière à satisfaire la condition de conservation de la charge.
# La méthode HIO est généralement utilisée en combinaison avec d'autres techniques de reconstruction de
# la phase, comme la méthode Error Reduction (ER).

import numpy as np
import imageio
import matplotlib.pyplot as plt


# Chargez l'image au format PNG et convertissez-la en niveaux de gris
image = plt.imread('cat.png').mean(axis=2)


def reconstruct_phase_hio(image, beta=0.8, max_iter=100, tol=1e-6):
    # Initialiser la phase reconstruite à zéro
    phase_reconstructed = np.zeros_like(image, dtype=float)
    
    # Calculer la transformée de Fourier de l'image mesurée
    F_measured = np.fft.fft2(image)
    
    for i in range(max_iter):
        # Calculer la transformée de Fourier de la phase reconstruite
        F_reconstructed = np.fft.fft2(phase_reconstructed)
        
        # Mettre à jour la transformée de Fourier
        F_new = (1 - beta) * F_measured + beta * F_measured * np.exp(1j * np.angle(F_reconstructed))
        
        # Calculer la nouvelle phase reconstruite
        phase_reconstructed_new = np.fft.ifft2(F_new).real
        
        # Calculer la différence entre la phase reconstruite précédente et la phase reconstruite courante
        diff = np.abs(phase_reconstructed_new - phase_reconstructed).max()
        
        # Mettre à jour la phase reconstruite
        phase_reconstructed = phase_reconstructed_new
        
        # Si la différence est inférieure au seuil de tolérance, arrêter la boucle
        if diff < tol:
            break
    
    return phase_reconstructed, i


# In[41]:


reconstructed_phase, iter_nbr = reconstruct_phase_hio(image, beta=0.8, max_iter=20, tol=1e-6)
print(f"nbr iterations = {iter_nbr}")

plt.imshow(reconstructed_phase, cmap='gray')
plt.show()


# -------------------------------------------------------------------------------------------------------------------------------

# # La méthode Plane-to-Plane

# Le programme ci-dessous définit une fonction nommée `plane_to_plane` qui permet de reconstruire une image de phase à partir d'une image en niveau de gris en utilisant la méthode Plane-to-Plane.
# 
# La fonction prend en entrée cinq arguments :
# 
# - `I` : une image en niveau de gris sous forme de tableau NumPy de dimension MxN (M lignes et N colonnes)
# - `K1` : une matrice de calibration de la caméra 1 sous forme de tableau NumPy de dimension 3x3
# - `K2` : une matrice de calibration de la caméra 2 sous forme de tableau NumPy de dimension 3x3
# - `w` : un poids de l'optimisation de Wirtinger Flow sous forme de flottant (par défaut 0.01)
# - `max_iter` : le nombre d'itérations maximal de l'algorithme sous forme d'entier (par défaut 1000)
# La fonction retourne en sortie une image de phase reconstruite sous forme de tableau NumPy de dimension MxN.
# 
# Le corps de la fonction commence par calculer la matrice fondamentale F à partir des matrices de calibration K1 et K2. Ensuite, la fonction initialise l'image de phase à zéro et entame une boucle qui itère jusqu'à convergence de l'algorithme ou jusqu'au nombre d'itérations maximal. A chaque itération, la fonction calcule les dérivées de l'image `I` et utilise ces dérivées ainsi que l'algorithme de Wirtinger Flow pour mettre à jour l'image de phase. Enfin, la fonction calcule l'erreur de l'image de phase et vérifie si cette erreur est suffisamment petite pour considérer que l'algorithme a convergé. Si c'est le cas, la boucle s'arrête, sinon elle continue jusqu'au nombre d'itérations maximal.

# In[14]:


import numpy as np
import cv2

def plane_to_plane(I, K1, K2, w=0.01, max_iter=1000):
    """
    Reconstruit la phase à partir d'une image en utilisant la méthode Plane-to-Plane
    
    Args:
    - I: image en niveau de gris (numpy array de dimension MxN)
    - K1: matrice de calibration de la caméra 1 (numpy array de dimension 3x3)
    - K2: matrice de calibration de la caméra 2 (numpy array de dimension 3x3)
    - w: poids de l'optimisation de Wirtinger Flow (float, par défaut 0.5)
    - max_iter: nombre d'itération maximal (int, par défaut 1000)
    
    Returns:
    - phase: image de phase reconstruite (numpy array de dimension MxN)
    """
    
    # Calculer la matrice fundamental F
    F = np.matmul(np.linalg.inv(K1).T, np.matmul(K2, np.linalg.inv(K2).T))
    
    # Initialiser la phase à zéro
    phase = np.zeros_like(I)
    
    # Répéter jusqu'à convergence ou jusqu'au nombre d'itération maximal
    for i in range(max_iter):
        # Calculer les dérivées de l'image
        dy, dx = np.gradient(I)
        
        # Appliquer l'optimisation de Wirtinger Flow
        phase = phase - w * (dx * np.sin(2 * phase) + dy * np.cos(2 * phase))
        
        # Calculer l'erreur
        error = np.abs(np.sin(phase))
        
        # Vérifier la convergence
        if np.mean(error) < 1e-6:
            break
    
    return phase

# Charger l'image au format PNG et la convertir en niveau de gris
image = cv2.imread("cat.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Définir les matrices de calibration de la caméra 1 et de la caméra 2
K1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
K2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Reconstruire la phase à partir de l'image en niveau de gris
phase = plane_to_plane(gray, K1, K2)

# Afficher l'image de phase reconstruite
plt.imshow(phase, cmap='gray')
plt.show()


# Les matrices `K1` et `K2` sont des matrices de calibration de deux caméras.
# 
# La matrice de calibration d'une caméra est une matrice qui contient les informations sur les paramètres optiques et géométriques de la caméra. Elle est généralement utilisée pour transformer les coordonnées pixel d'une image en coordonnées réelles dans l'espace.
# 
# La forme générale d'une matrice de calibration est la suivante :
# 
# $$
# \begin{bmatrix}
# fx & 0 & cx \\
# 0 & fy & cy \\
# 0 & 0 & 1
# \end{bmatrix}
# $$
# 
# où `fx` et `fy` sont les facteurs de focalisation de l'image en x et en y, `cx` et `cy` sont les coordonnées du centre optique de l'image en x et en y, et `1` est un coefficient de normalisation.
# 
# Dans le programme, `K1` et `K2` sont des matrices de calibration de deux caméras différentes. Elles sont utilisées pour calculer la matrice fondamentale `F` qui relie les deux plans de phase et d'image de ces caméras. La matrice fondamentale `F` est utilisée par la fonction `plane_to_plane` pour minimiser l'erreur entre ces deux plans lors de la reconstruction de l'image de phase.
# 
# Les matrices `K1` et `K2` sont initialisées comme des matrices identités de dimension `3x3`.
# 
# Dans le cas où `K1` et `K2` sont des matrices identités, cela signifie que les caméras 1 et 2 sont des caméras parfaites, c'est-à-dire des caméras sans distorsion ni aberration chromatique.
# 
# `Cf : mon rapport de stage`

# -------------------------------------------------------------------------------------------------------------------------------
