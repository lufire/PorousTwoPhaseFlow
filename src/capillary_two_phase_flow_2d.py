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
import pemfc
from pemfc import constants
from pemfc.src.fluid import fluid
from pemfc.src.fluid import diffusion_model
import saturation_model as sm
import porous_layer as pl
import settings
from parameters import fluid_dict, porous_dict, electrode_dict

matplotlib.use('TkAgg')

# Physical boundary conditions and parameters
# Operating conditions
current_density = settings.boundary_conditions['current_density']
temp_bc = settings.boundary_conditions['channel_temperature']
operating_voltage = settings.boundary_conditions['operating_voltage']

# Humidity in channel
h_chl = settings.boundary_conditions['channel_humidity']
# Fraction of inlet oxygen concentration
# (at simulated position along the channel)
f_O2 = settings.boundary_conditions['oxygen_fraction']
# Saturation at channel gdl interace
s_chl = settings.boundary_conditions['gdl_channel_saturation']
# Constant gas pressure
p_gas = settings.boundary_conditions['channel_pressure']
# Liquid fraction (molar) of produced water at CL-GDL interface
liquid_water_fraction_cl = \
    settings.boundary_conditions['cl_gdl_liquid_water_fraction']

# Physical parameters
thermo_neutral_voltage = electrode_dict["thermoneutral_voltage"]
faraday = constants.FARADAY
gas_constant = constants.GAS_CONSTANT
# rho_water = 977.8
# mu_water = 0.4035e-3

# Electrochemical reaction parameters
n_charge = electrode_dict['charge_number']
n_stoich = electrode_dict['reaction_stoichiometry']

# Heat flux due to current density
cathode_heat_flux_fraction = \
    settings.boundary_conditions['cathode_heat_flux_fraction']
heat_flux = (thermo_neutral_voltage - operating_voltage) * current_density \
            * cathode_heat_flux_fraction

# Parameters for SGL 34BA (5% PTFE)
width = settings.domain['width']
porosity = porous_dict['porosity']
permeability_abs = porous_dict['permeability'][0]
thermal_conductivity_eff = \
    np.asarray(porous_dict['thermal_conductivity']) * porosity

# Numerical resolution
nx = settings.domain['nx']
ny = settings.domain['ny']

thickness = porous_dict['thickness']

dx = width / nx
dy = thickness / ny

L = dx * nx
W = dy * ny
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
X, Y = mesh.faceCenters

# Create fluid object
n_cells = mesh.numberOfCells
fluid_dict['nodes'] = n_cells
fluid_dict['temperature'] = temp_bc
fluid_dict['pressure'] = p_gas
fluid_dict['humidity'] = h_chl
fluid_dict['components']['O2']['molar_fraction'] = \
    fluid_dict['components']['O2']['molar_fraction'] * f_O2
fluid_dict['components']['N2']['molar_fraction'] = \
    1.0 - fluid_dict['components']['O2']['molar_fraction']
humid_air = fluid.factory(fluid_dict, backend='pemfc')
humid_air.update()

# Initialize porous layer
porous_layer = pl.PorousTwoPhaseLayer(porous_dict)

# Initialize saturation model
# saturation_model = 'leverett'
# saturation_dict = porous_dict['saturation_model'][saturation_model]
# saturation_dict['porosity'] = porosity
# saturation_dict['permeability'] = permeability_abs
saturation_model = porous_layer.saturation_model

# Find specie to not explicitly solve for
id_inert = np.where(np.asarray(n_stoich) == 0.0)[-1][0]
name_inert = humid_air.species_names[id_inert]

# Constant factor for saturation "diffusion" coefficient
# D_s_const = rho_water / mu_water * permeability_abs
D_s_const = humid_air.liquid.density / humid_air.liquid.viscosity \
    * permeability_abs

# Initialize mesh variables
# Saturation diffusion coefficient
D_s = CellVariable(mesh=mesh, value=D_s_const)
D_s_f = FaceVariable(mesh=mesh, value=D_s.arithmeticFaceValue())

# Concentration diffusion coefficients (only solve for n-1 species)
solution_species = [name for i, name in enumerate(humid_air.species_names)
                    if n_stoich[i] != 0.0]
solve_species = \
    dict(zip(humid_air.species_names,
             [item in solution_species for item in humid_air.species_names]))
diff_model = diffusion_model.MixtureAveragedDiffusionModel(humid_air.gas)
diff_model.update(humid_air.temperature, humid_air.pressure,
                  humid_air.mole_fraction, update_names=solution_species)
D_c = {name: CellVariable(mesh=mesh,
                          value=diff_model.d_eff[humid_air.species_id[name]])
       for name in solution_species}
D_c_f = {name: FaceVariable(mesh=mesh, value=D_c[name].arithmeticFaceValue())
         for name in solution_species}

# Thermal diffusion coefficient (conductivity)
K_th = CellVariable(mesh=mesh, value=thermal_conductivity_eff)

# Liquid pressure
p_liq = CellVariable(name="Liquid pressure",
                     mesh=mesh,
                     value=p_gas,
                     hasOld=True)

# Saturation
s = CellVariable(mesh=mesh, value=0.0, hasOld=True)

# Temperature
temp = CellVariable(mesh=mesh, value=temp_bc)

# Species concentration, inert specie will not be solved for
c = {name: CellVariable(name='c_' + name, mesh=mesh,
                        value=humid_air.gas.concentration[
                            humid_air.species_id[name]])
     for name in solution_species}
# Species mole fractions, including inert specie
x = {name: CellVariable(name='x_' + name, mesh=mesh,
                        value=humid_air.gas.mole_fraction[i])
     for i, name in enumerate(humid_air.species_names)}

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
sigma_water_bc = humid_air.phase_change_species.calc_surface_tension(temp_bc)[0]
p_capillary_top = \
    saturation_model.calc_capillary_pressure(s_chl, sigma_water_bc)
p_liquid_top = p_capillary_top + p_gas
p_liq.setValue(p_liquid_top)
# p_liq.constrain(p_liquid_top, facesTop)
p_liq.constrain(p_liquid_top, facesTopRight)
# p_liq_bot = p_liquid_top + 200.0
# p_liq.constrain(p_liq_bot, facesBottom)
# p_liq.faceGrad.constrain(water_flux, facesBottom)
D_s.constrain(0.0, facesBottom)
for name in solution_species:
    D_c[name].constrain(0.0, facesBottom)

# Flux boundary conditions
gas_mole_flux_cl = \
    {name: current_density * n_stoich[humid_air.species_id[name]]
     / (n_charge * faraday)
     for name in solution_species}
# Setup saturation diffusion equation
# Water flux due to current density
liquid_mole_flux_cl = gas_mole_flux_cl['H2O'] * liquid_water_fraction_cl
gas_mole_flux_cl['H2O'] *= (1.0 - liquid_water_fraction_cl)
water_id = humid_air.species_id['H2O']
liquid_mass_flux_cl = \
    liquid_mole_flux_cl * humid_air.species.mw[humid_air.species_id['H2O']]

# Boundary conditions for temperature
temp.constrain(temp_bc, facesTopLeft)
K_th.constrain(0.0, facesBottom)

# Boundary conditions for molar concentrations
c_bc = {name: humid_air.gas.concentration[humid_air.species_id[name], 0]
        for name in solution_species}
for name in solution_species:
    c[name].constrain(c_bc[name], facesTopRight)

# Setup discretized transport equations
eq_s = DiffusionTerm(coeff=D_s_f) \
       - (facesBottom * liquid_mass_flux_cl).divergence
eq_t = DiffusionTerm(K_th) \
       - (facesBottom * heat_flux).divergence
eq_c = {name: DiffusionTerm(coeff=D_c_f[name])
        - (facesBottom * gas_mole_flux_cl[name]).divergence
        for name in solution_species}

# Setup numerical parameters
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
iter_count = 0
residuals = []

# Start iteration loop
while True:
    # Check convergence criteria
    if iter_count > iter_min and residual <= error_tol:
        print('Solution converged with {} steps and residual = {}'.format(
            iter_count, residual))
        break
    if iter_count >= iter_max:
        print('Solution did not converge within {} steps and residual = {}'
              ''.format(iter_count, residual))
        break

    # Calculate inert specie
    c_array = np.zeros((humid_air.n_species, n_cells))
    for i in range(len(c_array)):
        if i != id_inert:
            name = humid_air.species_names[i]
            c_array[i] = c[name].value
    c_total = p_gas / (gas_constant * humid_air.temperature)
    c_array[id_inert] = c_total - np.sum(c_array, axis=0)

    # Update fluid properties
    humid_air.update(temp.value.ravel(),
                     np.ones(temp.value.ravel().shape) * p_gas,
                     mole_composition=c_array)
    t = temp.value.ravel()
    sigma = humid_air.phase_change_species.calc_surface_tension(
        temp.value.ravel())[0]
    # Update diffusion model
    diff_model.update(humid_air.temperature, humid_air.pressure,
                      humid_air.mole_fraction, update_names=solution_species)

    # Update diffusion coefficients
    # Saturation transport coefficient
    D_s.setValue(D_s_const * porous_layer.calc_relative_permeability(s))
    D_s_f.setValue(D_s.arithmeticFaceValue())
    # Concentration diffusion coefficients
    for name in solution_species:
        D_c[name].setValue(diff_model.d_eff[humid_air.species_id[name]])
        D_c_f[name].setValue(D_c[name].arithmeticFaceValue())

    # Solve Transport equations
    residual_s = eq_s.sweep(var=p_liq)  # , underRelaxation=urfs[i])
    residual_t = eq_t.sweep(var=temp)
    residual_c = [eq_c[name].sweep(var=c[name]) for name in solution_species]

    # Save old capillary pressure
    p_cap_old = np.copy(p_cap)
    # Calculate capillary pressure
    p_cap[:] = p_liq.value - p_gas


    # Calculate new saturation values using under-relaxation
    s_old = np.copy(s.value)
    s_new = saturation_model.calc_saturation(p_cap, sigma)
    s_value = urf * s_new + (1.0 - urf) * s_old
    s.setValue(s_value)

    s_diff = (s_value - s_old) / s_value
    p_diff = (p_cap - p_cap_old) / p_cap
    eps_s = np.dot(s_diff.transpose(), s_diff) / (2.0 * len(s_diff))
    eps_p = np.dot(p_diff.transpose(), p_diff) / (2.0 * len(p_diff))

    eps = eps_s + eps_p
    residual = residual_t + residual_s + np.sum(residual_c) + eps

    # Update iteration counter
    residuals.append(residual)
    iter_count += 1

# Update mole fractions
for name in humid_air.species_names:
    x[name].setValue(humid_air.gas.mole_fraction[humid_air.species_id[name]])

if __name__ == '__main__':
    viewer = Viewer(vars=s)  # , datamin=0., datamax=1.)
    viewer.plot()
    input("Saturation. Press <return> to proceed...")

    viewer = Viewer(vars=temp)  # , datamin=0., datamax=1.)
    viewer.plot()
    input("Temperature. Press <return> to proceed...")

    viewer = Viewer(vars=x['O2'])  # , datamin=0., datamax=1.)
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
