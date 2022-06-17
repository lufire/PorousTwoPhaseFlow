# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:35:42 2021

@author: lukas

according to: 
Ferreira, Rui B, et al. “A One-Dimensional and Two-Phase Flow Model of a
Proton Exchange Membrane Fuel Cell: A 1-D and Two-Phase Flow Model of a PEM
Fuel Cell.” Journal of Chemical Technology & Biotechnology 90, no. 9
(September 2015): 1547–51. https://doi.org/10.1002/jctb.4651.

"""
import math
import numpy as np
from matplotlib import pyplot as plt

def leverett_hi(s):
    return 1.47 * (1.0 - s) - 2.12 * (1.0 - s) ** 2 + 1.263 * (1.0 - s) ** 3

def leverett_ho(s):
    return 1.47 * s - 2.12 * s ** 2 + 1.263 * s ** 3

def leverett_j(s, theta):
    if theta < 90.0:
        return leverett_hi(s)
    else:
        return leverett_ho(s)

def perm_rel(s):
    return s ** 3.0


# operating conditions
temp = 273.15 + 70.0
current_density = 50000.0


# parameters
th_gdl = 2e-4
perm_gdl = 5e-6
rho_water = 977.75
mu_water = 0.405e-3
mm_water = 18e-3
faraday = 96485.3415
epsilon = 0.8
theta_gdl = 120.0
sigma_water = 0.07275 * (1.0 - 0.002 * (temp - 291))

# boundary condition at channel
s_chl = 1e-5

# initial condition
s_0 = 1e-2

# discretization
n_z = 10
dz = th_gdl / n_z
z = np.linspace(0, th_gdl, n_z)

# precalculations
material_constant = rho_water * perm_gdl * sigma_water \
    * math.cos(theta_gdl * math.pi / 180.0) \
    / (mu_water * mm_water * math.sqrt(perm_gdl / epsilon))

water_flux = current_density / (2.0 * faraday)


# setup linear system
# setup matrix center diagonals
center_diag = np.zeros(n_z)
center_diag[:-1] = - 1.0 / dz

# Dirichlet bc at last node
center_diag[-1] = 1.0

# upper diagonal
upper_diag = np.zeros(n_z - 1)
upper_diag[:] = 1.0 / dz

# coefficient matrix
A = np.diag(center_diag) + np.diag(upper_diag, k=1)

# setup constant coefficient vector of right hand side
rhs_const = np.zeros(n_z)
rhs_const[:-1] = - water_flux / material_constant
rhs_const[-1] = s_chl

# initial condition vector
s_in = np.ones(n_z) * s_0

s = s_in

n_iter = 1000.0
tol = 1e-10
error = 1e5
i = 0
while i < n_iter and error > tol:
    J_s = leverett_j(s, theta_gdl)
    rhs = np.copy(rhs_const)
    rhs[:-1] /= (perm_rel(s)[:-1] * J_s[:-1])
    s_old = np.copy(s)
    s = np.linalg.solve(A, rhs)
    error = s - s_old
    error = np.dot(error, error)
    error /= 2 * len(s)
    # print(error)
    i += 1

plt.plot(z * 1e6, s)
plt.show()    
    
 




