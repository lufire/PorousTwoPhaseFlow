# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:41:36 2021

@author: feierabend

according to:
Pasaogullari, Ugur, and C. Y. Wang. “Liquid Water Transport in Gas Diffusion
Layer of Polymer Electrolyte Fuel Cells.” Journal of The Electrochemical
Society 151, no. 3 (2004): A399. https://doi.org/10.1149/1.1646148.
"""

import math
import numpy as np
from scipy import optimize
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve
# import sympy as sy
from matplotlib import pyplot as plt
import saturation as sat

# s = sy.symbols('s')

# boundary conditions
current_density = np.linspace(100.0, 30000.0, 100)
current_density = [20000.0]
temp = 343.15

# parameters
faraday = 96485.3329
rho_water = 977.8
mu_water = 0.4035e-3
mm_water = 0.018
sigma_water = 0.07275 * (1.0 - 0.002 * (temp - 291.0))

# comparison SGG
thickness = 260e-6
porosity = 0.74
permeability_abs = 1.88e-11
contact_angles = np.asarray([70.0, 130.0])

# numerical discretization
nz = 100
z = np.linspace(0, thickness, nz)
dz = thickness / nz

# saturation bc
s_chl = 0.001

# initial saturation
s_0 = np.ones(z.shape) * s_chl

# channel pressure
p_chl = 101325.0
nu_water = mu_water / rho_water
water_flux = current_density[0] / (2.0 * faraday) * mm_water


def saturation_func(s, theta):
    if theta < 90.0:
        return s ** 4.0 * (-0.2415 + 0.66765 * s - 0.6135 * s ** 2.0)
    else:
        return s ** 4.0 * (0.35425 - 0.8480 * s + 0.6135 * s ** 2.0)


# get constants
constants = []
for contact_angle in contact_angles:
    theta = contact_angle * math.pi / 180.0
    c = saturation_func(s_chl, contact_angle) \
        - water_flux * nu_water * z[-1] / \
        (sigma_water * math.cos(theta) * math.sqrt(porosity * permeability_abs))
    constants.append(c)
saturation_avg = []
saturations = []

start_time = time.time()
for j in range(1):

    for i in range(len(contact_angles)):
        theta = contact_angles[i] * math.pi / 180.0

        def der_root_saturation_hi(s):
            return - 4.0 * 0.2415 * s ** 3.0 + 5.0 * 0.66765 * s ** 4.0 - 6.0 * \
                   0.6135 * s ** 5.0

        def der_root_saturation_ho(s):
            return 4.0 * 0.35425 * s ** 3.0 - 5.0 * 0.8480 * s ** 4.0 + 6.0 * \
                   0.6135 * s ** 5.0

        def der_root_saturation(s):
            if contact_angles[i] < 90.0:
                return - 0.966 * s ** 3.0 + 3.33825 * s ** 4.0 - \
                       3.681 * s ** 5.0
            else:
                return 1.417 * s ** 3.0 - 4.24 * s ** 4.0 + 3.681 * s ** 5.0

        def root_saturation_hi(s):
            return s ** 4.0 * (-0.2415 + 0.66765 * s - 0.6135 * s ** 2.0) \
                - water_flux * nu_water * z / \
                (sigma_water * math.cos(theta)
                 * math.sqrt(porosity * permeability_abs)) \
                - constants[i]

        def root_saturation_ho(s):
            return s ** 4.0 * (0.35425 - 0.8480 * s + 0.6135 * s ** 2.0) \
                - water_flux * nu_water * z / \
                (sigma_water * math.cos(theta)
                 * math.sqrt(porosity * permeability_abs)) \
                - constants[i]


        def root_saturation(s):
            return saturation_func(s, contact_angles[i]) \
                   - water_flux * nu_water * z / \
                   (sigma_water * math.cos(theta)
                    * math.sqrt(porosity * permeability_abs)) \
                   - constants[i]
        s_0 = np.ones(nz) * 0.01
        # s_0 = 0.001

        # if contact_angles[i] < 90.0:
        #     # solution = optimize.newton(root_saturation_hi, s_0,
        #     #                            fprime=der_root_saturation_hi)
        #     solution = optimize.newton(root_saturation_hi, s_0)
        # else:
        #     # solution = optimize.newton(root_saturation_ho, s_0,
        #     #                            fprime=der_root_saturation_ho)
        #     solution = optimize.newton(root_saturation_ho, s_0)
        # saturation = optimize.newton(root_saturation, s_0)
        saturation = optimize.newton(root_saturation, s_0,
                                     fprime=der_root_saturation)
        # saturation = solution

        # print(solution.message)
        saturations.append(saturation)
        # print('Current density (A/m²): ', current_density[0])
        #
        # print('Average saturation (-): ', np.average(saturation))
        # print('GDL-channel interface saturation (-): ', saturation[-1])

        saturation_avg.append(np.average(saturation))

end_time = time.time()

print(end_time - start_time)

saturation_avg = np.asarray(saturation_avg)

# create plots
# fig, ax = plt.subplots(dpi=150)
#
# linestyles = ['solid', 'solid', 'solid']
# markers = ['.', '.', '.']
# colors = ['k', 'r', 'b']
# labels = ['Leverett-J {}°'.format(str(int(item))) for item in contact_angles]
# labels.append('PSD')
#
# for i in range(len(contact_angles)):
#     ax.plot(z * 1e6, saturations[i], linestyle=linestyles[i], marker=markers[i],
#             color=colors[i], label=labels[i])
# ax.legend()
#
# # ax.set_xlim([0.0, 1.0])
# # ax.set_ylim([-2000, 2000.0])
# # s = np.linspace(0.0, 1.0, 100)
# # j = leverett_j(s, contact_angle)
#
# # fig, ax = plt.subplots(dpi=150)
# # ax.plot(s, j)
# plt.tight_layout()
# plt.show()






