import numpy as np
import pemfc.src.fluid.fluid as fl
from scipy import special
from scipy import optimize
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod
from src.constants import SQRT_2
import porous_layer as pl


class SaturationModel(ABC):

    def __new__(cls, model_dict, porous_layer, fluid):
        if not isinstance(porous_layer, pl.PorousTwoPhaseLayer):
            raise TypeError('porous_layer must be of type PorousTwoPhaseLayer')
        if not isinstance(fluid, fl.TwoPhaseMixture):
            raise TypeError('fluid must be of type TwoPhaseMixture')

        model_type = model_dict.get('type', 'leverett')
        if model_type == 'leverett':
            return super(SaturationModel, cls).__new__(LeverettModel)
        elif model_type == 'psd':
            return super(SaturationModel, cls).__new__(PSDModel)
        elif model_type == 'gostick_correlation':
            return super(SaturationModel, cls).__new__(GostickCorrelation)
        elif model_type == 'imbibition_drainage':
            return super(SaturationModel, cls).__new__(ImbibitionDrainageCurve)
        else:
            raise NotImplementedError

    def __init__(self, model_dict, porous_layer, fluid):
        self.fluid = fluid
        self.dict = model_dict
        self.model_type = model_dict['type']
        self.s_min = model_dict.get('minimum_saturation', 1e-3)

    @abstractmethod
    def calc_saturation(self, capillary_pressure, *args, **kwargs):
        pass

    @abstractmethod
    def calc_capillary_pressure(self, saturation, *args, **kwargs):
        pass

    @staticmethod
    def young_laplace(capillary_pressure, sigma, contact_angle):
        return - 1.0 * 2.0 * sigma * np.cos(contact_angle * np.pi / 180.0) \
               / capillary_pressure

    def calc_dpc_ds(self, saturation, *args, **kwargs):
        s_range = np.linspace(self.s_min, 1.0, len(saturation))
        ds = np.diff(s_range)[0]
        pc_range = self.calc_capillary_pressure(s_range, *args, **kwargs)
        dpc_ds_range = np.gradient(pc_range, ds)
        dpc_ds = interp1d(s_range, dpc_ds_range, kind='linear')
        return dpc_ds(saturation)


class LeverettModel(SaturationModel):
    def __init__(self, model_dict, porous_layer, fluid):
        super().__init__(model_dict, porous_layer, fluid)
        self.contact_angle = model_dict['contact_angle']
        self.contact_angle_rad = self.contact_angle * np.pi / 180.0
        if isinstance(porous_layer.permeability, (list, tuple, np.ndarray)):
            self.permeability = porous_layer.permeability[0]
        else:
            self.permeability = porous_layer.permeability
        # model_dict['permeability']
        self.porosity = porous_layer.porosity  # model_dict['porosity']

    def calc_capillary_pressure(self, saturation, *args, **kwargs):
        surface_tension = kwargs.get('surface_tension',
                                     self.fluid.surface_tension)
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

        min_pressure = self.leverett_p_s(self.s_min, np.min(surface_tension))
        max_pressure = self.leverett_p_s(1.0, np.max(surface_tension))
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

    def calc_saturation(self, capillary_pressure, *args,
                        saturation_prev=None, **kwargs):
        surface_tension = kwargs.get('surface_tension',
                                     self.fluid.surface_tension)
        saturation = \
            self.leverett_s_p(capillary_pressure,surface_tension,
                              saturation_prev=saturation_prev)
        # return saturation
        return np.where(saturation < self.s_min, self.s_min,
                        np.where(saturation > 1.0, 1.0, saturation))


class PSDModel(SaturationModel):
    def __init__(self, model_dict, porous_layer, fluid):
        super().__init__(model_dict, porous_layer, fluid)
        self.f = np.asarray(model_dict['f'])
        self.s = np.asarray(model_dict['s'])
        self.F = np.asarray(model_dict['F'])
        self.r = np.asarray(model_dict['r'])
        self.contact_angle = np.asarray(model_dict['contact_angle'])
        # np.asarray(model_dict['contact_angle'])

    def get_critical_radius(self, capillary_pressure, surface_tension):
        critical_radius_hydrophilic = \
            np.where(capillary_pressure < 0.0,
                     self.young_laplace(capillary_pressure, surface_tension,
                                        self.contact_angle[0]),
                     np.inf)
        critical_radius_hydrophobic = \
            np.where(capillary_pressure < 0.0, np.inf,
                     self.young_laplace(capillary_pressure, surface_tension,
                                        self.contact_angle[1]))

        # critical_radius_hydrophilic[critical_radius_hydrophilic < 0.0] = SMALL
        # critical_radius_hydrophobic[critical_radius_hydrophobic < 0.0] = SMALL

        return np.asarray([critical_radius_hydrophilic,
                           critical_radius_hydrophobic])

    def calc_saturation(self, capillary_pressure, *args, **kwargs):
        surface_tension = kwargs.get('surface_tension',
                                     self.fluid.surface_tension)
        critical_radius = self.get_critical_radius(capillary_pressure,
                                                   surface_tension)
        saturation = np.zeros(critical_radius.shape[-1])
        phi = [1, -1]
        for i in range(self.f.shape[0]):
            for j in range(self.f.shape[1]):
                saturation += self.F[i] * self.f[i, j] * 0.5 \
                              * (1.0 + phi[i] * special.erf(
                                 (np.log(critical_radius[i])
                                  - np.log(self.r[i, j]))
                                 / (self.s[i, j] * SQRT_2)))
        return np.where(saturation < self.s_min, self.s_min,
                        np.where(saturation > 1.0, 1.0, saturation))

    def calc_capillary_pressure(self, saturation,
                                capillary_pressure_prev=None, **kwargs):
        if isinstance(saturation, np.ndarray):
            saturation[saturation < self.s_min] = self.s_min
            saturation[saturation > 1.0] = 1.0

        def root_saturation_psd(capillary_pressure):
            return saturation - \
                   self.calc_saturation(capillary_pressure)

        if capillary_pressure_prev is not None:
            p_c_in = capillary_pressure_prev
        else:
            p_c_in = np.ones(np.asarray(saturation).shape)
        return optimize.root(root_saturation_psd, p_c_in).x


class GostickCorrelation(SaturationModel):
    """
    According to:
    Gostick, Jeff T., Marios A. Ioannidis, Michael W. Fowler,
    and Mark D. Pritzker. “Wettability and Capillary Behavior of Fibrous Gas
    Diffusion Media for Polymer Electrolyte Membrane Fuel Cells.” Journal of
    Power Sources 194, no. 1 (October 2009): 433–44.
    https://doi.org/10.1016/j.jpowsour.2009.04.052.
    """
    def __init__(self, model_dict, porous_layer, fluid):
        super().__init__(model_dict, porous_layer, fluid)
        self.s_w_m = model_dict.get('maximum_saturation', 1.0)
        self.s_w_r = model_dict.get('residual_saturation', 0.0)
        self.f = model_dict['f']
        self.m = model_dict['m']
        self.n = model_dict['n']
        self.p_c_b = model_dict['P_C_b']
        try:
            assert len(self.f) == len(self.m) == len(self.n) == len(self.n)
        except AssertionError:
            raise AssertionError(
                'Parameters for saturation model of type {} must be lists with '
                'equal length'.format(model_dict['type']))
        self.p_atm = model_dict.get('atmospheric_pressure', 101325.0)

    def calc_saturation(self, capillary_pressure, *args, **kwargs):
        p_c = capillary_pressure
        s_w = np.zeros(capillary_pressure.shape)
        for i in range(len(self.f)):
            s_w_i = 1.0 - (1.0 + ((p_c + self.p_atm)/self.p_c_b[i])
                           ** self.m[i]) ** -self.n[i]
            s_w += self.f[i] * (s_w_i * (self.s_w_m - self.s_w_r) + self.s_w_r)

        return s_w

    def calc_capillary_pressure(self, saturation, capillary_pressure_prev=None,
                                **kwargs):
        if isinstance(saturation, np.ndarray):
            saturation[saturation < self.s_min] = self.s_min
            saturation[saturation > 1.0] = 1.0

        def root_saturation(capillary_pressure):
            return saturation - \
                   self.calc_saturation(capillary_pressure)

        if capillary_pressure_prev is not None:
            p_c_in = capillary_pressure_prev
        else:
            p_c_in = np.ones(np.asarray(saturation).shape)
        return optimize.root(root_saturation, p_c_in).x


class ImbibitionDrainageCurve(SaturationModel):
    """
    According to section 5.3 in:
    Behrends, Hanno, Detlef Stolten, and Wolfgang Schröder. “Untersuchung des
    Wassertransportes in Gasdiffusionsmedien für Polymer-Brennstoffzellen.”
    Publikationsserver der RWTH Aachen University, 2015.
    https://publications.rwth-aachen.de/record/464442.
    """
    def __init__(self, model_dict, porous_layer, fluid):
        super().__init__(model_dict, porous_layer, fluid)
        drainage_dict = model_dict['drainage_model']
        self.drainage_model = SaturationModel(drainage_dict, porous_layer,
                                              fluid)
        imbibition_dict = model_dict['imbibition_model']
        self.imbibition_model = SaturationModel(imbibition_dict,
                                                porous_layer, fluid)

    def calc_saturation(self, capillary_pressure, *args, **kwargs):
        sat_imb = self.imbibition_model.calc_saturation(capillary_pressure,
                                                        *args, **kwargs)
        sat_drain = self.drainage_model.calc_saturation(capillary_pressure,
                                                        *args, **kwargs)
        humidity = kwargs.get('humidity', self.fluid.humidity)
        sat = sat_imb * np.heaviside(1.0 - humidity, 0.0) \
            + sat_drain * np.heaviside(humidity - 1.0, 1.0)
        return sat

    def calc_capillary_pressure(self, saturation, capillary_pressure_prev=None,
                                **kwargs):
        if isinstance(saturation, np.ndarray):
            saturation[saturation < self.s_min] = self.s_min
            saturation[saturation > 1.0] = 1.0

        def root_saturation(capillary_pressure):
            return saturation - \
                   self.calc_saturation(capillary_pressure)

        if capillary_pressure_prev is not None:
            p_c_in = capillary_pressure_prev
        else:
            p_c_in = np.ones(np.asarray(saturation).shape)
        return optimize.root(root_saturation, p_c_in).x
