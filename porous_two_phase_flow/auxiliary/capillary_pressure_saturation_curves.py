import numpy as np
import json
from porous_two_phase_flow.saturation_model import SaturationModel
from porous_two_phase_flow.porous_layer import PorousLayer
from pemfc.src.fluid import fluid
import matplotlib.pyplot as plt
import matplotlib
np.set_printoptions(legacy="1.25")


settings_path = (r'D:\Software\Python\PycharmProjects\PorousTwoPhaseFlow\data'
                 r'\saturation_model_settings.json')

with open(settings_path) as file:
    settings = json.load(file)


fluid = fluid.create(settings['fluid'])
porous_layer = PorousLayer(settings['porosity_model'])
saturation_models = \
    {key: SaturationModel(settings['saturation_model'][key], porous_layer,
                          fluid)
     for key in settings['saturation_model'] if key not in ('type', 'PSD')}

saturation = np.linspace(0.01, 0.99, 100)
capillary_pressures = {key: value.calc_capillary_pressure(saturation)
                       for key, value in saturation_models.items()}

fig, ax = plt.subplots(figsize=(12, 8))
for key, value in capillary_pressures.items():
    ax.plot(value, saturation, label=key)
ax.legend()
type = 'Leverett'
interfacial_area = porous_layer.calc_specific_interfacial_area(
    saturation, capillary_pressures[type], saturation_models[type])
plt.show()
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(saturation, interfacial_area)
ax.legend()
plt.show()
