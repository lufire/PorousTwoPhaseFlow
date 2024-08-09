from __future__ import annotations
import numpy as np
from scipy import integrate
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

# if TYPE_CHECKING:
from porous_two_phase_flow import saturation_model as sm


class PorousLayer(ABC):

    def __new__(cls, model_dict):
        model_type = model_dict.get('type', 'CarbonPaper')
        if model_type == 'CarbonPaper':
            return super(PorousLayer, cls).__new__(CarbonPaper)

        else:
            raise NotImplementedError

    def __init__(self, model_dict):
        self.porosity = model_dict['porosity']
        self.permeability = model_dict['permeability']
        self.dict = model_dict
        self.model_type = model_dict['type']

    @abstractmethod
    def calc_effective_property(self, transport_property,
                                matrix_property=True):
        pass


class PorousTwoPhaseLayer(PorousLayer, ABC):
    def __init__(self, model_dict):
        super().__init__(model_dict)

        self.n_rel = model_dict.get('relative_permeability_exponent', 3.0)
        self.pore_radius = model_dict['pore_radius']
        self.pore_volume = (2.0 * self.pore_radius) ** 3.0
        self.pore_density = self.porosity / self.pore_volume  # 1 / m³

    @abstractmethod
    def calc_specific_interfacial_area(self, saturation, *args, **kwargs):
        pass

    def calc_relative_permeability(self, saturation: np.ndarray,
                                   saturation_min: float = None,
                                   saturation_model: sm.SaturationModel = None):
        saturation = np.copy(saturation)
        if saturation_min is not None:
            saturation_min = saturation_min
        elif saturation_model is not None:
            saturation_min = saturation_model.s_min
        else:
            saturation_min = 1e-6
        saturation[saturation <= 0.0] = saturation_min
        return saturation ** self.n_rel


class CarbonPaper(PorousTwoPhaseLayer):
    def __init__(self, model_dict):
        super().__init__(model_dict)
        self.bruggemann_coeff = model_dict['bruggemann_coefficient']
        # self.contact_angle = model_dict['contact_angle']
        # pore volume base on cubic approximation

    def calc_effective_property(self, transport_property, matrix_property=True):
        if matrix_property:
            return transport_property * (1.0 - self.porosity)
        else:
            return transport_property * self.porosity ** self.bruggemann_coeff

    def calc_specific_interfacial_area(
            self, saturation: np.ndarray,
            capillary_pressure: np.ndarray = None,
            saturation_model:  sm.SaturationModel = None,
            **kwargs):
        """
        Joekar-Niasar, V., S. M. Hassanizadeh, und A. Leijnse. „Insights into
        the Relationships Among Capillary Pressure, Saturation, Interfacial Area
        and Relative Permeability Using Pore-Network Modeling“. Transport in
        Porous Media 74, Nr. 2 (September 2008): 201–19.
        https://doi.org/10.1007/s11242-007-9191-7.

        :param saturation: non-wetting phase saturation
        :param capillary_pressure: capillary pressure as difference between
        non-wetting and wetting phase [Pa]
        :param saturation_model: instance of type SaturationModel
        :return: specific interfacial area [m²/m³]
        """
        if not isinstance(capillary_pressure, np.ndarray):
            raise TypeError('capillary_pressure argument must be a numpy array')
        if not isinstance(saturation_model, sm.SaturationModel):
            raise TypeError('saturation_model must be of type SaturationModel')
        # Wetting phase saturation
        s_w = 1.0 - saturation

        # # Capillary pressure is defined positively as difference between
        # # non-wetting and wetting fluid in kPa
        # p_c = np.abs(capillary_pressure) / 1000.0
        # interfacial_area = (6.462 * 1.0 * (1.0 - s_w) ** 1.244 *
        #                     p_c ** -0.963)

        p_c_max = saturation_model.calc_capillary_pressure(0.99)
        p_c = np.copy(capillary_pressure)
        p_c[p_c > p_c_max] = p_c_max
        a1 = -1.6e-1
        a2 = 1.43e-5
        a3 = 1.91e-1
        interfacial_area = (a1 * (p_c_max - p_c) * (1.0 - s_w)
                            + a2 * (p_c_max - p_c) ** 2.0 * (1.0 - s_w)
                            + a3 * (p_c_max - p_c) * (1.0 - s_w) ** 2.0)
        if 'a_vol_max' in kwargs:
            a_vol_max = kwargs.get('a_vol_max', 265.0)
            # a_vol_max = 265.0
            interfacial_area = a_vol_max * np.pi * (saturation * s_w ** 2.0) ** 0.6
        return interfacial_area
