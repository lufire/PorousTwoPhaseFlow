import numpy as np
from scipy import special
from scipy import optimize
from abc import ABC, abstractmethod
from src.constants import SQRT_2
import porous_layer as pl


class SaturationModel(ABC):

    def __new__(cls, model_dict, porous_layer):
        if not isinstance(porous_layer, pl.PorousTwoPhaseLayer):
            raise TypeError('porous_layer must be of type PorousTwoPhaseLayer')
        model_type = model_dict.get('type', 'leverett')
        if model_type == 'leverett':
            return super(SaturationModel, cls).\
                __new__(LeverettModel)
        elif model_type == 'psd':
            return super(SaturationModel, cls).\
                __new__(PSDModel)
        else:
            raise NotImplementedError

    def __init__(self, model_dict, porous_layer):
        self.dict = model_dict
        self.model_type = model_dict['type']
        self.s_min = model_dict.get('minimum_saturation', 1e-4)


    @abstractmethod
    def calc_saturation(self, capillary_pressure, surface_tension, *args,
                        **kwargs):
        pass

    @abstractmethod
    def calc_capillary_pressure(self, saturation, surface_tension, *args,
                                **kwargs):
        pass

    @staticmethod
    def young_laplace(capillary_pressure, sigma, contact_angle):
        return - 1.0 * 2.0 * sigma * np.cos(contact_angle * np.pi / 180.0) \
               / capillary_pressure


class LeverettModel(SaturationModel):
    def __init__(self, model_dict, porous_layer):
        super().__init__(model_dict, porous_layer)
        self.contact_angle = model_dict['contact_angle']
        self.contact_angle_rad = self.contact_angle * np.pi / 180.0
        if isinstance(porous_layer.permeability, (list, tuple, np.ndarray)):
            self.permeability = porous_layer.permeability[0]
        else:
            self.permeability = porous_layer.permeability
        # model_dict['permeability']
        self.porosity = porous_layer.porosity  # model_dict['porosity']

    def calc_capillary_pressure(self, saturation, surface_tension, *args,
                                **kwargs):

        return self.leverett_p_s(saturation, surface_tension)

    @staticmethod
    def leverett_hi(saturation):
        return 1.417 * (1.0 - saturation) - 2.120 * (1.0 - saturation) ** 2.0 \
            + 1.263 * (1.0 - saturation) ** 3.0

    @staticmethod
    def leverett_ho(saturation):
        return 1.417 * saturation - 2.120 * saturation ** 2.0 \
               + 1.263 * saturation ** 3.0

    def leverett_j(self, saturation):
        if self.contact_angle < 90.0:
            return self.leverett_hi(saturation)
        else:
            return self.leverett_ho(saturation)

    def leverett_p_s(self, saturation, surface_tension):
        factor = - surface_tension \
            * np.cos(self.contact_angle_rad) \
            * np.sqrt(self.porosity / self.permeability)
        return factor * self.leverett_j(saturation)

    def leverett_s_p(self, capillary_pressure, surface_tension,
                     saturation_prev=None):
        factor = - surface_tension * np.cos(self.contact_angle_rad) \
            * np.sqrt(self.porosity / self.permeability)

        min_pressure = self.leverett_p_s(0.0, surface_tension)
        max_pressure = self.leverett_p_s(1.0, surface_tension)
        capillary_pressure[capillary_pressure < min_pressure] = min_pressure
        capillary_pressure[capillary_pressure > max_pressure] = max_pressure

        def root_leverett_p_s(s):
            return factor * self.leverett_j(s) \
                   - capillary_pressure
        if saturation_prev is not None:
            s_in = saturation_prev
        else:
            s_in = np.zeros(np.asarray(capillary_pressure).shape) + self.s_min
        solution = optimize.newton(root_leverett_p_s, s_in)
        saturation = solution
        return saturation

    def calc_saturation(self, capillary_pressure, surface_tension, *args,
                        saturation_prev=None, **kwargs):

        saturation = \
            self.leverett_s_p(capillary_pressure, surface_tension,
                              saturation_prev=saturation_prev)
        # return saturation
        return np.where(saturation < 0.0, 0.0,
                        np.where(saturation > 1.0, 1.0, saturation))


class PSDModel(SaturationModel):
    def __init__(self, model_dict, porous_layer):
        super().__init__(model_dict, porous_layer)
        self.f = np.asarray(model_dict['f'])
        self.s = np.asarray(model_dict['s'])
        self.F = np.asarray(model_dict['F'])
        self.r = np.asarray(model_dict['r'])
        self.contact_angle = porous_layer.contact_angle
        # np.asarray(model_dict['contact_angle'])

    def get_critical_radius(self, capillary_pressure, sigma):
        critical_radius_hydrophilic = \
            np.where(capillary_pressure < 0.0,
                     self.young_laplace(capillary_pressure, sigma,
                                        self.contact_angle[0]),
                     np.inf)
        critical_radius_hydrophobic = \
            np.where(capillary_pressure < 0.0, np.inf,
                     self.young_laplace(capillary_pressure, sigma,
                                        self.contact_angle[1]))

        # critical_radius_hydrophilic[critical_radius_hydrophilic < 0.0] = SMALL
        # critical_radius_hydrophobic[critical_radius_hydrophobic < 0.0] = SMALL

        return np.asarray([critical_radius_hydrophilic,
                           critical_radius_hydrophobic])

    def calc_saturation(self, capillary_pressure, surface_tension, *args,
                        **kwargs):
        critical_radius = self.get_critical_radius(
            capillary_pressure, surface_tension)
        saturation = np.zeros(critical_radius.shape[-1])
        phi = [1, -1]
        for i in range(self.f.shape[0]):
            for j in range(self.f.shape[1]):
                saturation += self.F[i] * self.f[i, j] * 0.5 \
                              * (1.0 + phi[i] * special.erf(
                                 (np.log(critical_radius[i])
                                  - np.log(self.r[i, j]))
                                 / (self.s[i, j] * SQRT_2)))
        return saturation

    def calc_capillary_pressure(self, saturation, surface_tension,
                                capillary_pressure_prev=None, **kwargs):

        saturation[saturation < self.s_min] = self.s_min
        saturation[saturation > 1.0] = 1.0

        def root_saturation_psd(capillary_pressure):
            return saturation - \
                   self.calc_saturation(capillary_pressure, surface_tension)

        if capillary_pressure_prev is not None:
            p_c_in = capillary_pressure_prev
        else:
            p_c_in = np.ones(np.asarray(saturation).shape)
        return optimize.root(root_saturation_psd, p_c_in).x
