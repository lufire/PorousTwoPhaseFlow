# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:40:01 2021

@author: lukas

according to:
Zhou, J., A. Putz, and M. Secanell. “A Mixed Wettability Pore Size
Distribution Based Mathematical Model for Analyzing Two-Phase Flow in Porous
Electrodes I. Mathematical Model.” Journal of The Electrochemical Society 164,
no. 6 (2017): F530–39.
"""

import numpy as np
import scipy as sp
import math
from matplotlib import pyplot as plt

# psd function
r_ch = [14.2e-6, 34e-6]
s_ch = [1.0, 0.35]
f_ch = [0.28, 0.72]
F = [0.08, 1.0 - 0.08]



def dX_dr(r):
    sqrt_2 = np.sqrt(2.0)
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    outer_sum = 0.0
    for i in range(len(F)):
        inner_sum = 0.0
        for j in range(len(f_ch)):
            E = np.exp(-((np.log(r) - np.log(r_ch[j])) 
                         / (s_ch[j] * sqrt_2)) ** 2.0)
            inner_sum += f_ch[j] / (r_ch[j] * s_ch[j] * sqrt_2pi) * E
        outer_sum += F[i] * inner_sum
    return outer_sum

# def dX_dr(r):
#     sqrt_2 = np.sqrt(2.0)
#     sqrt_2pi = np.sqrt(2.0 * np.pi)
#     outer_sum = 0.0
#     for i in range(len(F)):
#         inner_sum = 0.0
#         for j in range(len(f_ch)):
#             E = np.exp(-((np.log(r) - np.log(r_ch[j])) 
#                          / (s_ch[j] * sqrt_2)) ** 2.0)
#             inner_sum += f_ch[j] / (r_ch[j] * s_ch[j] * sqrt_2pi) * E
#         outer_sum += F[i] * inner_sum
#     return outer_sum
        
pore_radius = np.logspace(-7, -3, 100)

psd = np.asarray(dX_dr(pore_radius))

fig, ax = plt.subplots()
ax.plot(pore_radius, np.average(pore_radius) * psd)
ax.set_xscale('log')
plt.show()