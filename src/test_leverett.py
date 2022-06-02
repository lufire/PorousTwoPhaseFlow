# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:41:36 2021

@author: feierabend
"""

import math
import numpy as np
from scipy import special
from scipy import optimize
from matplotlib import pyplot as plt
import saturation as sat


def leverett_hi(s):
    return 1.417 * (1.0 - s) - 2.120 * (1.0 - s) ** 2 \
        + 1.263 * (1.0 - s) ** 3
    
def leverett_ho(s):
    return 1.417 * s - 2.120 * s ** 2 + 1.263 * s ** 3

def leverett_j(s, theta):
    if theta < 90.0:
        return leverett_hi(s)
    else:
        return leverett_ho(s)
    
def leverett_p_s(saturation, surface_tension, contact_angle, 
                 porosity, permeability):
    factor = - surface_tension * np.cos(contact_angle * np.pi / 180.0) \
        * np.sqrt(porosity / permeability)
    return factor * leverett_j(saturation, contact_angle)

def leverett_s_p(capillary_pressure, surface_tension, contact_angle, 
                 porosity, permeability):
    factor = - surface_tension * np.cos(contact_angle * np.pi / 180.0) \
        * np.sqrt(porosity / permeability)
    def root_leverett_p_s(saturation):
        return factor * leverett_j(saturation, contact_angle) - capillary_pressure
    s_in = np.zeros(np.asarray(capillary_pressure).shape) + 0.0
    saturation = optimize.root(root_leverett_p_s, s_in).x
    return saturation

# parameters
temp = 343.15
surface_tension = 0.07275 * (1.0 - 0.002 * (temp - 291.0))
porosity = 0.78
permeability = 6.2e-12
contact_angles = [70.0, 130.0]

# psd specific parameters
r_k = np.asarray([[14.20e-6, 34.00e-6], [14.20e-6, 34.00e-6]])
F_HI = 0.08
F = np.asarray([F_HI, 1.0 - F_HI])
f_k = np.asarray([[0.28, 0.72], [0.28, 0.72]])
s_k = np.asarray([[0.35, 1.0], [0.35, 1.0]])

# calculate saturation from given capillariy pressures
capillary_pressure = np.linspace(-10000.0, 10000.0, 500)
# capillary_pressure = np.linspace(11, 13.0, 100)
saturations = []
for contact_angle in contact_angles:
    params_leverett = [surface_tension, contact_angle, porosity, permeability]
    saturations.append(sat.get_saturation(
        capillary_pressure, [], params_leverett, model_type='leverett'))

params_psd = [surface_tension, contact_angles, F, f_k, r_k, s_k]
saturations.append(sat.get_saturation(
        capillary_pressure, params_psd, [], model_type='psd'))

# create plots
fig, ax = plt.subplots(dpi=150)

linestyles = ['solid', 'dotted', 'dashed']
colors = ['k', 'r', 'b']
labels = ['Leverett-J {}°'.format(str(int(item))) for item in contact_angles]
labels.append('PSD')
for i in range(len(saturations)):
# ax.plot(capillary_pressure, saturation)
    ax.plot(saturations[i], capillary_pressure, linestyle=linestyles[i],
            color=colors[i], label=labels[i])
ax.legend()

# ax.set_xlim([0.0, 1.0])
# ax.set_ylim([-2000, 2000.0])
# s = np.linspace(0.0, 1.0, 100)
# j = leverett_j(s, contact_angle)

# fig, ax = plt.subplots(dpi=150)
# ax.plot(s, j)
plt.tight_layout()
plt.show()
