# ALGORITHME PLANE TO PLANE
import numpy as np


repertSOURCE = r'images'
zSource = 0.13
npx = 201   # nbre points selon l'axe Y (horizontal)
npy = 201  # nbre points selon l'axe Z (vertical)
polar = 'V'
px = 0.001   # taille pixel
f = 15000    # fréquence en MHz
fic1 = 'CornetKu18_15GHz_z139'  # fic sortie HFSS plan 1
fic2 = 'CornetKu18_15GHz_z150'  # fic sortie HFSS plan 2
fic3 = 'CornetKu18_15GHz_z200'  # fic sortie HFSS plan 3
Nit = 300    # Nbre itération PtP
x = np.arange(-(npx-1)/2*px, px*(npx-1)/2, px)
x1 = x
y = np.arange(-(npy-1)/2*px, px*(npy-1)/2, px)
y1 = y
PPM = 1/px
[d1, Ex1, Ey1, Ez1, E1, Etg1] = LectureHFSSplanCPLX(repertSOURCE, fic1, x, y, npx, npy, zSource)
[d2, Ex2, Ey2, Ez2, E2, Etg2] = LectureHFSSplanCPLX(repertSOURCE, fic2, x, y, npx, npy, zSource)
[d3, Ex3, Ey3, Ez3, E3, Etg3] = LectureHFSSplanCPLX(repertSOURCE, fic3, x, y, npx, npy, zSource)

z1 = fic1[end-2:end]
z1 = str2num(z1)/1000
z2 = fic2[end-2:end]
z2 = str2num(z2)/1000
z3 = fic3[end-2:end]
z3 = str2num(z3)/1000
Etit1='E1'
Etit2='E2'
Etit3='E3'