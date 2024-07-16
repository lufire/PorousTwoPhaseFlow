# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:41:36 2021

@author: feierabend
"""

import numpy as np
from matplotlib import pyplot as plt
from pemfc.src.fluid import fluid
from src.settings import porous_dict, fluid_dict
import src.porous_layer as pl

# Create fluid object
fluid_dict['nodes'] = 1
fluid_dict['temperature'] = 343.15
fluid_dict['pressure'] = 101325
fluid_dict['humidity'] = 1.0
fluid_dict['components']['O2']['molar_fraction'] = \
    fluid_dict['components']['O2']['molar_fraction'] * 0.75
fluid_dict['components']['N2']['molar_fraction'] = \
    1.0 - fluid_dict['components']['O2']['molar_fraction']
humid_air = fluid.factory(fluid_dict, backend='pemfc')
humid_air.update()


# Boundary conditions for liquid pressure
sigma_water_bc = humid_air.phase_change_species.calc_surface_tension(
    humid_air.temperature)[0]
# Initialize porous layer
porous_layer = pl.PorousTwoPhaseLayer(porous_dict)

# Get saturation model
sat_model = porous_layer.saturation_model

saturation = np.linspace(0.001, 1.0, 1000)

capillary_pressure = sat_model.calc_capillary_pressure(saturation,
                                                       sigma_water_bc)

dpc_ds = np.diff(capillary_pressure) / np.diff(saturation)

plt.plot(saturation[:-1], dpc_ds)
plt.show()