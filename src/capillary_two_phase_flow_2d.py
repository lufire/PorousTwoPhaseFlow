# Solve a two-dimensional diffusion problem in a square domain.

# This example solves a diffusion problem and demonstrates the use of
# applying boundary condition patches.

# .. index::
#    single: Grid2D
import numpy as np
from matplotlib import pyplot as plt
from fipy import CellVariable, FaceVariable, Grid2D, Viewer, TransientTerm, \
    DiffusionTerm
from fipy.tools import numerix
from fipy import input
import saturation as sat
import matplotlib
matplotlib.use('TkAgg')

# Physical boundary conditions and parameters
# Operating conditions
current_density = 30000.0
temp_bc = 343.15
operating_voltage = 0.5

# Saturation at channel gdl interace
s_chl = 0.001

# Constant gas pressure
p_gas = 101325.0

# Physical parameters
thermo_neutral_voltage = 1.482
faraday = 96485.3329
rho_water = 977.8
mu_water = 0.4035e-3
mm_water = 0.018
sigma_water = 0.07275 * (1.0 - 0.002 * (temp_bc - 291.0))

# Water flux due to current density
water_flux = current_density / (2.0 * faraday) * mm_water

# Heat flux due to current density
cathode_heat_flux_fraction = 0.7
heat_flux = (thermo_neutral_voltage - operating_voltage) * current_density \
            * cathode_heat_flux_fraction

# Parameters for SGL 34BA (5% PTFE)
thickness = 260e-6
width = 2e-3
porosity = 0.74
permeability_abs = 1.88e-11
thermal_conductivity = np.asarray([28.4, 2.8]) * porosity

# psd specific parameters
r_k = np.asarray([[14.20e-6, 34.00e-6], [14.20e-6, 34.00e-6]])
F_HI = 0.08
F = np.asarray([F_HI, 1.0 - F_HI])
f_k = np.asarray([[0.28, 0.72], [0.28, 0.72]])
s_k = np.asarray([[0.35, 1.0], [0.35, 1.0]])

contact_angles = np.asarray([70.0, 130.0])
contact_angle = contact_angles[1]
saturation_model = 'leverett'

# Collect parameters in lists for each model
params_leverett = \
    [sigma_water, contact_angle, porosity, permeability_abs]
params_psd = [sigma_water, contact_angles, F, f_k, r_k, s_k]

# Numerical resolution
nx = 100
ny = 10

dx = width / nx
dy = thickness / ny

L = dx * nx
W = dy * ny
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
X, Y = mesh.faceCenters

# Select parameter set according to saturation model
if saturation_model == 'leverett':
    params = params_leverett
elif saturation_model == 'psd':
    params = params_psd
else:
    raise NotImplementedError

# Constant factor for saturation "diffusion" coefficient
D_s_const = rho_water / mu_water * permeability_abs

# Initialize mesh variables
# Saturation diffusion coefficient
D_s = CellVariable(mesh=mesh, value=D_s_const)
D_s_f = FaceVariable(mesh=mesh, value=D_s.arithmeticFaceValue())

D_c = CellVariable(mesh=mesh, value=[[X**2 * 1000, 0],
                                     [0, -Y**2 * 1000]])

K_th = CellVariable(mesh=mesh, value=thermal_conductivity)

# D_c = CellVariable(mesh=mesh, value=0.0)

# Liquid pressure
p_liq = CellVariable(name="Liquid pressure",
                     mesh=mesh,
                     value=p_gas,
                     hasOld=True)

# Saturation
s = CellVariable(mesh=mesh, value=0.0, hasOld=True)

# Temperature
temp = CellVariable(mesh=mesh, value=temp_bc)

# Species concentration, last species will not be solved for
species_fractions = \
    [{'name': 'O2', 'value': 0.21},
     {'name': 'H2O', 'value': 0.0},
     {'name': 'N2', 'value': 0.79}]

x = [CellVariable(name='x_' + species_fractions[i]['name'],
                  mesh=mesh,
                  value=species_fractions[i]['value'])
     for i, species in enumerate(species_fractions)]

c = [CellVariable(name='c_' + species_fractions[i]['name'],
                  mesh=mesh,
                  value=species_fractions[i]['value'])
     for i, species in enumerate(species_fractions)]

# Set boundary conditions
# top: fixed Dirichlet condition (fixed liquid pressure according to saturation
# boundary condition)
# bottom: Neumann flux condition (according to reaction water flux)
# facesTopLeft = ((mesh.facesLeft & (Y > L / 2))
#                 | (mesh.facesTop & (X < L / 2)))
# facesBottomRight = ((mesh.facesRight & (Y < L / 2))
#                     | (mesh.facesBottom & (X > L / 2)))
# Specify boundary patches
facesTopLeft = (mesh.facesTop & (X < L / 2.0))
facesTopRight = (mesh.facesTop & (X >= L / 2.0))
facesTop = mesh.facesTop
facesBottom = mesh.facesBottom

# Boundary conditions for liquid pressure
p_capillary_top = sat.get_capillary_pressure(s_chl, params, saturation_model)
p_liquid_top = p_capillary_top + p_gas
p_liq.setValue(p_liquid_top)
# p_liq.constrain(p_liquid_top, facesTop)
p_liq.constrain(p_liquid_top, facesTopRight)
# p_liq_bot = p_liquid_top + 200.0
# p_liq.constrain(p_liq_bot, facesBottom)
# p_liq.faceGrad.constrain(water_flux, facesBottom)
D_s.constrain(0.0, facesBottom)

# Setup saturation diffusion equation
eq_s = DiffusionTerm(coeff=D_s_f) - (facesBottom * water_flux).divergence

# Boundary conditions for temperature
temp.constrain(temp_bc, facesTopLeft)
K_th.constrain(0.0, facesBottom)

# Setup saturation diffusion equation
eq_t = DiffusionTerm(K_th) \
       - (facesBottom * heat_flux).divergence

# We can solve the steady-state problem
iter_max = 1000
iter_min = 10
error_tol = 1e-7
urf = 0.5
urfs = [0.5]
s_value = np.ones(nx * ny) * s_chl
s_old = np.copy(s_value)
p_cap = np.ones(s_value.shape) * 1000.0
p_cap_old = np.copy(p_cap)
residual = np.inf

iter = 0

residuals = []

while True:
    if iter > iter_min and residual <= error_tol:
        print('Solution converged with {} steps and residual = {}'.format(
            iter, residual))
        break
    if iter >= iter_max:
        print('Solution did not converge within {} steps and residual = {}'
              ''.format(iter, residual))
        break

    # update diffusion values with previous saturation values
    D_s.setValue(D_s_const * sat.k_s(s))
    D_s_f.setValue(D_s.arithmeticFaceValue())

    # p_liq.faceGrad.constrain(water_flux, facesBottom)

    # solve liquid pressure transport equation
    residual_s = eq_s.sweep(var=p_liq) #, underRelaxation=urfs[i])
    residual_t = eq_t.sweep(var=temp)

    p_cap_old = np.copy(p_cap)
    # calculate capillary pressure
    p_cap[:] = p_liq.value - p_gas

    # calculate new saturation values
    s_old = np.copy(s.value)
    s_new = sat.get_saturation(p_cap, params, saturation_model)
    s_value = urf * s_new + (1.0 - urf) * s_old
    s.setValue(s_value)

    s_diff = (s_value - s_old) / s_value
    p_diff = (p_cap - p_cap_old) / p_cap
    eps_s = np.dot(s_diff.transpose(), s_diff) / (2.0 * len(s_diff))
    eps_p = np.dot(p_diff.transpose(), p_diff) / (2.0 * len(p_diff))

    eps = eps_s + eps_p
    residual = residual_t + residual_s  # + eps
    # update iteration counter
    residuals.append(residual)
    iter += 1

if __name__ == '__main__':
    viewer = Viewer(vars=s) #, datamin=0., datamax=1.)
    viewer.plot()
    input("Saturation. Press <return> to proceed...")

    viewer = Viewer(vars=temp) #, datamin=0., datamax=1.)
    viewer.plot()
    input("Temperature. Press <return> to proceed...")

    fig, ax = plt.subplots()
    ax.plot(np.asarray(list(range(len(residuals)))), np.asarray(residuals))

    # for i in range(len(urfs)):
    #     ax.plot(list(range(len(residuals[i]))), residuals[i],
    #             label='urf = ' + str(urfs[i]))
    ax.set_yscale('log')
    # plt.legend()
    plt.show()

# .. image:: mesh20x20steadyState.*
#    :width: 90%
#    :align: center
#    :alt: stead-state solution to diffusion problem on a 2D domain with some
#    Dirichlet boundaries

# and test the value of the bottom-right corner cell.
#

# print(numerix.allclose(phi(((L,), (0,))), valueBottom, atol=1e-2))
