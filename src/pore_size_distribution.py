# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:41:36 2021

according to:
Zhou, J., A. Putz, and M. Secanell. “A Mixed Wettability Pore Size 
Distribution Based Mathematical Model for Analyzing Two-Phase Flow in Porous 
Electrodes: I. Mathematical Model.” Journal of The Electrochemical Society 164, 
no. 6 (2017): F530–39. https://doi.org/10.1149/2.0381706jes.

@author: feierabend
"""

import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import special

# parameters for SGL 34BA (5% PTFE)
thickness = 260e-6
porosity = 0.74
permeability_abs = 1.88e-11

# mixed wettability model parameters
r_k = np.asarray([[14.20e-6, 34.00e-6], [14.20e-6, 34.00e-6]])
F_HI = 0.3
F = np.asarray([F_HI, 1.0 - F_HI])
# f_k = np.asarray([[0.28, 0.72], [0.28, 0.72]])
# f_k = np.asarray([[0.0, 1.0], [0.28, 0.72]])
f_k = np.asarray([[0.28, 0.72], [0.0, 1.0]])


s_k = np.asarray([[1.0, 0.35], [1.0, 0.35]])
contact_angle = np.asarray([70.0, 122.0])


r = np.logspace(-7, -3, 1000)

sqrt_2 = math.sqrt(2.0)
sqrt_2pi = math.sqrt(2.0 * math.pi)

dx_dr = np.zeros(r.shape)
dx_dr_i = np.array([np.zeros(dx_dr.shape), np.zeros(dx_dr.shape)])
r_dx_dr_i = np.zeros(dx_dr_i.shape)

r_c = np.zeros(F.shape)
for i in range(F.shape[0]):
    for j in range(r_k.shape[0]):
        E_i_j = np.exp(- (np.log(r) - np.log(r_k[i, j])) ** 2.0 \
                   / (s_k[i, j] * sqrt_2))
        dx_dr_i[i] += F[i] * f_k[i, j] / (r * s_k[i, j] * sqrt_2pi) * E_i_j
        r_c[i] += f_k[i, j] * r_k[i, j]
        r_dx_dr_i[i] += r_c[i] * dx_dr_i[i]
    dx_dr += dx_dr_i[i]


# r_dx_dr = r_c[0] * dx_dr_i[0] + r_c[1] * dx_dr_i[1]
# r_c_ = r_k[0, 0] * f_k[0, 0] + r_k[0, 1] + f_k[0, 1]

# r_c_ = 30e-6
# plt.plot(r, r_dx_dr)
# plt.plot(r, r_dx_dr_i[0])
# plt.plot(r, r_dx_dr_i[1])
# plt.grid()
# plt.xscale("log")
# plt.show()




