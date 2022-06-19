import math
import numpy as np
from scipy import special
from scipy import optimize
from abc import ABC, abstractmethod


class PorousLayer(ABC):

    def __new__(cls, model_dict):
        model_type = model_dict.get('type', 'CarbonPaper')
        if model_type == 'CarbonPaper':
            return super(PorousLayer, cls).\
                __new__(CarbonPaper)

        else:
            raise NotImplementedError

    def __init__(self, model_dict):
        self.dict = model_dict
        self.model_type = model_dict['type']

    @abstractmethod
    def calc_effective_property(self, transport_property,
                                matrix_property=True):
        pass

    @abstractmethod
    def calc_two_phase_interfacial_area(self, saturation):
        pass


class CarbonPaper(PorousLayer):
    def __init__(self, model_dict):
        super().__init__(model_dict)
        self.bruggemann_coeff = model_dict['bruggemann_coefficient']
        self.porosity = model_dict['porosity']
        self.contact_angle = model_dict['contact_angle']
        self.pore_radius = model_dict['pore_radius']
        # pore volume base on cubic approximation
        self.pore_volume = (2.0 * self.pore_radius) ** 3.0

    def calc_effective_property(self, transport_property, matrix_property=True):
        if matrix_property:
            return transport_property * (1.0 - self.porosity)
        else:
            return transport_property * self.porosity ** self.bruggemann_coeff

    def calc_two_phase_interfacial_area(self, saturation):
        # Assuming single spherical droplet per pore
        radius_liquid = \
            (3.0 * self.pore_volume * saturation / (4.0 * np.pi)) ** (1.0 / 3.0)
        return 4.0 * np.pi * radius_liquid ** 2.0
