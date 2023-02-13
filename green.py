import numpy as np


def GreenFun2D(k, r_obs, r_source):
    """
    Calcul du propagateur de Green au point r_obs, pour un courant situé en r_source.
    r_obs et r_sou doivent être des vecteurs colonnes (produit dyadique = matrice (3 x 3)
    :param k: 
    :type k: 
    :param r_obs: 
    :type r_obs: 
    :param r_sou: 
    :type r_sou: 
    :return: 
    :rtype: 
    """
    mu0 = 4 * np.pi * 1e-7
    w = k * 3e8

    if (r_obs.shape or r_source.shape) != (3,):
        print('----- false dimension of vector (check dim = column [3,1]:-----')
        return
    R = np.linalg.norm(r_source - r_obs)    # source-observable distance
    Rhat = (r_obs - r_source) / R      # source-observable normalized vector
    DyadRhat = np.outer(Rhat, Rhat)         # dyadic product

    green_scal = np.exp(-1j * k * R) / (4 * np.pi * R)      # CONVENTION -jkr
    g1 = (3 / ((k * R)**2) + 3*1j / (k * R) - 1) * DyadRhat
    g2 = (1 - 1j / (k * R) - 1/(k * R)**2) * np.eye(3)

    # green_scal = np.exp(1j * k * R) / (4 * np.pi * R)       # CONVENTION +jkr
    # g1 = ( 3 / ((k * R)**2) -  3*1j / (k * R) -1 ) * DyadRhat
    # g2 = ( 1 + 1j / (k * R) - 1 / ((k * R)**2) ) * np.eye(3)

    G = mu0 * w * (g1 + g2) * green_scal
    # Attention!
    # En vue de l'utilisation de la fonction de Green pour la transformation CP-->CL
    # pour lequel il n'y a pas de kz, et puisque on définit des courants sur une surface dans le plan
    # (Oxy) pour lequel il n'y a pas non plus de kz, on supprime la 3e ligne et la 3e colonne de la
    # matrice de Green
    G = G[:2, :2]
    return G