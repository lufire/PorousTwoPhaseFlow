import numpy as np
import string

import pemfc.src.fluid.fluid as fl
from scipy import special
from scipy import optimize
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod
from .constants import SQRT_2
from . import porous_layer as pl


class SaturationModel(ABC):

    def __new__(cls, model_dict, porous_layer,
                fluid: fl.TwoPhaseMixture = None):
        if not isinstance(porous_layer, pl.PorousTwoPhaseLayer):
            raise TypeError('porous_layer must be of type PorousTwoPhaseLayer')
        # TODO: isinstance check fails because the same type, however
        #  separately imported. Possible solution is to make fluid an
        #  independent package. As a temporary solution, the type checking
        #  will be done differently
        # if not isinstance(fluid, fl.TwoPhaseMixture):
        #     raise TypeError('fluid must be of type TwoPhaseMixture')
        if not all(hasattr(fluid, attr) for attr in ["liquid", "gas"]):
            raise TypeError('fluid must be of type TwoPhaseMixture')

        model_type = model_dict.get('type', 'Leverett')
        model_type = model_type.replace('_', ' ')
        model_type = ''.join([item[0].upper() + item[1:]
                              for item in model_type.split(' ')])

        model_dict.update(model_dict.get(model_type, {}))
        if model_type == 'Leverett':
            return super(SaturationModel, cls).__new__(LeverettModel)
        elif model_type == 'PSD':
            return super(SaturationModel, cls).__new__(PSDModel)
        elif model_type == 'GostickCorrelation':
            return super(SaturationModel, cls).__new__(GostickCorrelation)
        elif model_type == 'DataTable':
            return super(SaturationModel, cls).__new__(DataTable)
        elif model_type == 'ImbibitionDrainage':
            return super(SaturationModel, cls).__new__(ImbibitionDrainageCurve)
        elif model_type == 'VanGenuchten':
            return super(SaturationModel, cls).__new__(VanGenuchtenModel)
        else:
            raise NotImplementedError

    def __init__(self, model_dict, porous_layer, fluid=None):
        self.fluid = fluid
        self.dict = model_dict
        self.model_type = model_dict['type']
        self.s_min = model_dict.get('minimum_saturation', 1e-3)
        precalculate_pressure = model_dict.get('precalculate_pressure', False)
        if precalculate_pressure:
            self.func_pc = self.interpolate_pressure()
        else:
            self.func_pc = None
        if model_dict.get('precalculate_gradient', False):
            self.func_grad = self.interpolate_gradient()
        else:
            self.func_grad = None

    @abstractmethod
    def calc_saturation(self, capillary_pressure, *args, **kwargs):
        pass

    @abstractmethod
    def calc_capillary_pressure(self, saturation, *args, **kwargs):
        if hasattr(self, "func_pc") and self.func_pc is not None:
            return self.func_pc(saturation)
        else:
            return None

    def implicit_capillary_pressure(self, saturation,
                                    capillary_pressure_prev=None, *args,
                                    **kwargs):
        saturation = np.copy(np.asarray(saturation))
        shape = saturation.shape
        saturation[saturation < self.s_min] = self.s_min
        saturation[saturation > 1.0] = 1.0
        saturation = saturation.ravel(order='F')

        def root_saturation(pressure):
            return saturation - \
                self.calc_saturation(pressure)

        if capillary_pressure_prev is not None:
            p_c_in = capillary_pressure_prev
        else:
            p_c_in = np.ones(np.asarray(saturation).shape)
        p_c_in_flat = p_c_in.ravel(order='F')
        # try:
        solution = optimize.root(root_saturation, p_c_in_flat).x
        # except Exception:
        #     raise ValueError
        capillary_pressure = np.reshape(solution, shape, order='F')
        return capillary_pressure

    @staticmethod
    def young_laplace(capillary_pressure, sigma, contact_angle):
        return - 1.0 * 2.0 * sigma * np.cos(contact_angle * np.pi / 180.0) \
               / capillary_pressure

    def interpolate_pressure(self, resolution=100):
        s_range = np.linspace(self.s_min, 1.0, resolution)
        surface_tension = np.average(self.fluid.surface_tension)
        pc_range = self.calc_capillary_pressure(
            s_range, surface_tension=surface_tension)
        return interp1d(s_range, pc_range, kind='linear')

    def interpolate_gradient(self, resolution=1000):
        s_range = np.linspace(self.s_min, 1.0, resolution)
        ds = np.diff(s_range)[0]
        surface_tension = np.average(self.fluid.surface_tension)
        pc_range = self.calc_capillary_pressure(
            s_range, surface_tension=surface_tension)
        dpc_ds_range = np.gradient(pc_range, ds)
        return interp1d(s_range, dpc_ds_range, kind='linear')

    def calc_gradient(self, saturation, use_precalculated=True, *args, **kwargs):
        if use_precalculated and self.func_grad is not None:
            return self.func_grad(saturation)
        else:
            dpc_ds = self.interpolate_gradient(resolution=len(saturation))
            return dpc_ds(saturation)


class LeverettModel(SaturationModel):
    def __init__(self, model_dict, porous_layer, fluid=None):
        self.contact_angle = model_dict['contact_angle']
        self.contact_angle_rad = self.contact_angle * np.pi / 180.0
        if isinstance(porous_layer.permeability, (list, tuple, np.ndarray)):
            self.permeability = porous_layer.permeability[0]
        else:
            self.permeability = porous_layer.permeability
        # model_dict['permeability']
        self.porosity = porous_layer.porosity  # model_dict['porosity']
        super().__init__(model_dict, porous_layer, fluid=fluid)

    def calc_capillary_pressure(self, saturation, *args, **kwargs):
        surface_tension = kwargs.get('surface_tension',
                                     self.fluid.surface_tension)
        if isinstance(saturation, np.ndarray):
            if (isinstance(surface_tension, np.ndarray)
                    and len(surface_tension) > 1):
                surface_tension = surface_tension.reshape(
                    saturation.shape, order='F')
        else:
            surface_tension = np.average(surface_tension)
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
        if isinstance(factor, np.ndarray):
            if factor.size != capillary_pressure.size:
                factor = np.average(factor)
        capillary_pressure = np.copy(capillary_pressure)
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
        try:
            solution = optimize.newton(root_leverett_p_s, s_in)
        except Exception:
            raise ValueError
        saturation = solution
        return saturation

    def calc_saturation(self, capillary_pressure, *args,
                        saturation_prev=None, **kwargs):
        surface_tension = kwargs.get('surface_tension',
                                     self.fluid.surface_tension)
        if isinstance(capillary_pressure, np.ndarray):
            if (isinstance(surface_tension, np.ndarray)
                    and surface_tension.size == capillary_pressure.size):
                surface_tension = surface_tension.ravel(order='F')
                surface_tension = surface_tension.reshape(
                    capillary_pressure.shape, order='F')
            else:
                surface_tension = np.average(surface_tension)
        else:
            surface_tension = np.average(surface_tension)
        saturation = \
            self.leverett_s_p(capillary_pressure, surface_tension,
                              saturation_prev=saturation_prev)
        # return saturation
        return np.where(saturation < self.s_min, self.s_min,
                        np.where(saturation > 1.0, 1.0, saturation))


class PSDModel(SaturationModel):
    def __init__(self, model_dict, porous_layer, fluid=None):
        self.f = np.asarray(model_dict['f'])
        self.s = np.asarray(model_dict['s'])
        self.F = np.asarray(model_dict['F'])
        self.r = np.asarray(model_dict['r'])
        self.contact_angle = np.asarray(model_dict['contact_angle'])
        super().__init__(model_dict, porous_layer, fluid=fluid)

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
        if isinstance(capillary_pressure, np.ndarray):
            if (isinstance(surface_tension, np.ndarray)
                    and surface_tension.size == capillary_pressure.size):
                surface_tension = surface_tension.reshape(
                    capillary_pressure.shape, order='F')
            else:
                surface_tension = np.average(surface_tension)
        else:
            surface_tension = np.average(surface_tension)
        critical_radius = self.get_critical_radius(capillary_pressure,
                                                   surface_tension)
        saturation = np.zeros(capillary_pressure.shape)
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
        saturation = np.array(saturation)
        shape = saturation.shape
        saturation[saturation < self.s_min] = self.s_min
        saturation[saturation > 1.0] = 1.0
        saturation = saturation.ravel(order='F')

        precalc_capillary_pressure = super().calc_capillary_pressure(saturation)
        if precalc_capillary_pressure is not None:
            return precalc_capillary_pressure

        def root_saturation_psd(capillary_pressure):
            return saturation - \
                   self.calc_saturation(capillary_pressure)

        if capillary_pressure_prev is not None:
            p_c_in = capillary_pressure_prev
        else:
            p_c_in = np.ones(np.asarray(saturation).shape)
        solution = optimize.root(root_saturation_psd, p_c_in).x
        return solution.reshape(shape, order='F')


class GostickCorrelation(SaturationModel):
    """
    According to:
    Gostick, Jeff T., Marios A. Ioannidis, Michael W. Fowler,
    and Mark D. Pritzker. “Wettability and Capillary Behavior of Fibrous Gas
    Diffusion Media for Polymer Electrolyte Membrane Fuel Cells.” Journal of
    Power Sources 194, no. 1 (October 2009): 433–44.
    https://doi.org/10.1016/j.jpowsour.2009.04.052.
    """
    def __init__(self, model_dict, porous_layer, fluid=None):
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
        model_dict['precalculate_pressure'] = model_dict.get(
            'precalculate_pressure', True)
        model_dict['precalculate_gradient'] = model_dict.get(
            'precalculate_gradient', True)
        super().__init__(model_dict, porous_layer, fluid=fluid)

    def calc_saturation(self, capillary_pressure, *args, **kwargs):
        p_c = capillary_pressure
        s_w = np.zeros(capillary_pressure.shape)
        for i in range(len(self.f)):
            s_w_i = 1.0 - (1.0 + ((p_c + self.p_atm)/self.p_c_b[i])
                           ** self.m[i]) ** -self.n[i]
            # s_w_i = ((1.0 + ((-p_c + self.p_atm)/self.p_c_b[i]) ** self.m[i])
            #          ** -self.n[i])
            s_w += self.f[i] * (s_w_i * (self.s_w_m - self.s_w_r) + self.s_w_r)

        return s_w

    def calc_capillary_pressure(self, saturation, capillary_pressure_prev=None,
                                **kwargs):
        saturation = np.copy(np.asarray(saturation))
        shape = saturation.shape
        s_min = self.s_w_r + 1e-3
        s_max = self.s_w_m - 1e-3
        saturation[saturation < s_min] = s_min
        saturation[saturation > s_max] = s_max
        saturation = saturation.ravel(order='F')

        precalc_capillary_pressure = super().calc_capillary_pressure(saturation)
        if precalc_capillary_pressure is not None:
            return precalc_capillary_pressure

        def root_saturation(capillary_pressure):
            return saturation - \
                   self.calc_saturation(capillary_pressure)

        if capillary_pressure_prev is not None:
            p_c_in = capillary_pressure_prev
        else:
            p_c_in = np.ones(np.asarray(saturation).shape)
        solution = optimize.root(root_saturation, p_c_in).x
        return solution.reshape(shape, order='F')


class DataTable(SaturationModel):
    """
    According to:
    Gostick, Jeff T., Marios A. Ioannidis, Michael W. Fowler,
    and Mark D. Pritzker. “Wettability and Capillary Behavior of Fibrous Gas
    Diffusion Media for Polymer Electrolyte Membrane Fuel Cells.” Journal of
    Power Sources 194, no. 1 (October 2009): 433–44.
    https://doi.org/10.1016/j.jpowsour.2009.04.052.
    """
    def __init__(self, model_dict, porous_layer, fluid):
        data_format = model_dict['data_format']
        if data_format == 'file':
            file_path = model_dict['file_path']
            with open(file_path, 'r') as file:
                data = np.loadtxt(file, delimiter=';')
            data = np.asarray(data).transpose()
        elif data_format == 'table':
            data = model_dict['data']
        else:
            raise NotImplementedError('saturation-capillary pressure data must'
                                      'be provided in file or as list in '
                                      'in put dictionary')
        # Sort data from lowest to highest saturation values
        sorted_indices = data[0].argsort()
        self.s_data = data[0][sorted_indices]
        self.pc_data = data[1][sorted_indices]
        model_dict['precalculate_pressure'] = \
            model_dict.get('precalculate_pressure', True)
        model_dict['precalculate_gradient'] = \
            model_dict.get('precalculate_gradient', True)
        self.s_func = interp1d(self.pc_data, self.s_data, kind='linear',
                               fill_value=(self.s_data[0], self.s_data[-1]),
                               bounds_error=False)
        self.pc_func = interp1d(self.s_data, self.pc_data, kind='linear',
                                fill_value=(self.pc_data[0], self.pc_data[-1]),
                                bounds_error=False)
        super().__init__(model_dict, porous_layer, fluid)

    def calc_saturation(self, capillary_pressure, *args, **kwargs):
        return self.s_func(capillary_pressure)

    def calc_capillary_pressure(self, saturation, *args, **kwargs):
        return self.pc_func(saturation)


class ImbibitionDrainageCurve(SaturationModel):
    """
    According to section 5.3 in:
    Behrends, Hanno, Detlef Stolten, and Wolfgang Schröder. “Untersuchung des
    Wassertransportes in Gasdiffusionsmedien für Polymer-Brennstoffzellen.”
    Publikationsserver der RWTH Aachen University, 2015.
    https://publications.rwth-aachen.de/record/464442.
    """
    def __init__(self, model_dict, porous_layer, fluid=None):
        drainage_dict = model_dict['drainage_model']
        self.drainage_model = SaturationModel(drainage_dict, porous_layer,
                                              fluid)
        imbibition_dict = model_dict['imbibition_model']
        self.imbibition_model = SaturationModel(imbibition_dict,
                                                porous_layer, fluid)
        super().__init__(model_dict, porous_layer, fluid=fluid)

    def calc_saturation(self, capillary_pressure, *args, **kwargs):
        sat_imb = self.imbibition_model.calc_saturation(capillary_pressure,
                                                        *args, **kwargs)
        sat_drain = self.drainage_model.calc_saturation(capillary_pressure,
                                                        *args, **kwargs)
        humidity = kwargs.get('humidity', self.fluid.humidity)
        if isinstance(capillary_pressure, np.ndarray):
            if (isinstance(humidity, np.ndarray)
                    and humidity.size == capillary_pressure.size):
                humidity = humidity.reshape(
                    capillary_pressure.shape, order='F')
            else:
                humidity = np.average(humidity)
        else:
            humidity = np.average(humidity)
        sat = sat_imb * np.heaviside(1.0 - humidity, 0.0) \
            + sat_drain * np.heaviside(humidity - 1.0, 1.0)
        return sat

    def calc_capillary_pressure(self, saturation, capillary_pressure_prev=None,
                                **kwargs):
        saturation = np.copy(np.asarray(saturation))
        shape = saturation.shape
        saturation[saturation < self.s_min] = self.s_min
        saturation[saturation > 1.0] = 1.0
        saturation = saturation.ravel(order='F')

        def root_saturation(pressure):
            return saturation - \
                   self.calc_saturation(pressure)

        if capillary_pressure_prev is not None:
            p_c_in = capillary_pressure_prev
        else:
            p_c_in = np.ones(np.asarray(saturation).shape)
        p_c_in_flat = p_c_in.ravel(order='F')
        # try:
        solution = optimize.root(root_saturation, p_c_in_flat).x
        # except Exception:
        #     raise ValueError
        capillary_pressure = np.reshape(solution, shape, order='F')
        return capillary_pressure


class VanGenuchtenModel(SaturationModel):
    def __init__(self, model_dict, porous_layer, fluid=None):
        self.s_r_n = model_dict.get('residual_non-wetting_saturation', 1.0)
        self.s_r_w = model_dict.get('residual_wetting_saturation', 0.0)
        self.wetting = model_dict.get('wetting', False)
        self.alpha = model_dict['alpha']
        self.m = model_dict['m']
        self.n = model_dict['n']
        # self.gamma = model_dict['gamma']
        super().__init__(model_dict, porous_layer, fluid=fluid)

    def calc_capillary_pressure(self, saturation, wetting=None,
                                capillary_pressure_prev=None, *args, **kwargs):
        if wetting is None:
            wetting = self.wetting
        if wetting:
            s_w = np.copy(saturation)
        else:
            s_w = 1.0 - saturation
        s_w = np.asarray(s_w)
        s_w[s_w < self.s_r_w] = self.s_r_w
        s_w_max = 1.0 - self.s_r_n
        s_w[s_w > s_w_max] = s_w_max
        s_e = np.asarray((s_w - self.s_r_w) / (1.0 - self.s_r_w - self.s_r_n))
        s_e[s_e < 0.0] = 0.0
        try:
            p_c = (np.abs(np.emath.power((s_e ** (1.0/self.m) - 1.0),
                                          (1.0 / self.n)))
                   / self.alpha)
        except FloatingPointError:
            p_c = self.implicit_capillary_pressure(
                saturation,
                capillary_pressure_prev=capillary_pressure_prev,
                wetting=wetting)
        return p_c

    def calc_saturation(self, capillary_pressure, wetting=None,
                        saturation_prev=None,
                        **kwargs):
        # capillary_pressure = np.copy(capillary_pressure)
        # # min_pressure = self.calc_capillary_pressure(self.s_min)
        # # max_pressure = self.calc_capillary_pressure(1.0)
        # #
        # # capillary_pressure[capillary_pressure < min_pressure] = min_pressure
        # # capillary_pressure[capillary_pressure > max_pressure] = max_pressure
        #
        # def root_capillary_pressure(sat):
        #     return capillary_pressure - self.calc_capillary_pressure(sat)
        # if saturation_prev is not None:
        #     s_in = saturation_prev
        # else:
        #     s_in = np.zeros(np.asarray(capillary_pressure).shape) + self.s_min
        # solution = optimize.newton(root_capillary_pressure, s_in)
        # saturation = solution
        try:
            s_e = (1.0 + np.abs(np.emath.power(self.alpha * capillary_pressure,
                                               self.n)) ** self.m)
        except FloatingPointError:
            raise FloatingPointError
        s_w = s_e * (1.0 - self.s_r_w - self.s_r_n) + self.s_r_w
        if wetting is None:
            wetting = self.wetting
        if wetting:
            saturation = s_w
        else:
            saturation = 1.0 - s_w
        return saturation
