from . import xray_compounds as xc
import numpy as np
import pint
unit = pint.UnitRegistry()
from .utils import extend_bounds


class LayerSpec:
    """
    Represents a single layer in a multilayer reflectivity model.

    Each LayerSpec defines physical and optical properties of a layer, including:
    - Thickness (fixed or fit)
    - Interface roughness (fixed or fit)
    - Energy-dependent refractive index components (n and k), either from material data or user-defined arrays

    The class supports both fixed values and parameter fitting, with bounds and initial guesses.
    Substrate layers are treated specially and cannot have thickness parameters.

    Attributes:
        name (str): Identifier for the layer.
        is_substrate (bool): Flag indicating whether the layer is a substrate.
        params (dict): Dictionary storing parameter specifications for thickness, roughness, n, and k.
        _nk_set (bool): Internal flag to ensure n and k arrays are defined before validation.

    Methods:
        fit_thickness(x0, bounds): Enable thickness fitting with initial guess and bounds.
        fixed_thickness(value): Set a fixed thickness value.
        fit_roughness(x0, bounds): Enable roughness fitting.
        fixed_roughness(value): Set a fixed roughness value.
        fit_nk_from_material(material, energy_uni, bounds_n, bounds_k, density): Fit n/k from material database.
        fixed_nk_from_material(material, energy_uni, density): Fix n/k from material database.
        fit_nk_array(n_array, k_array, bounds_n, bounds_k): Fit n/k from user-defined arrays.
        fixed_nk(n_array, k_array): Fix n/k from user-defined arrays.
        validate(energy_count): Ensure n/k arrays are properly defined and match energy resolution.
    """

    def __init__(self, name, is_substrate=False):
        self.name = name
        self.is_substrate = is_substrate
        self.params = {}
        self._nk_set = False

    def fit_thickness(self, x0, bounds=None, delta=None):
        if self.is_substrate:
            raise ValueError(f"Layer '{self.name}' is marked as substrate and cannot have thickness.")
        if delta is not None:
            bounds = (x0 - delta, x0 + delta)
        if bounds is None:
            raise ValueError("Either bounds or delta must be provided for thickness fitting.")
        self.params['thickness'] = {'fit': True, 'x0': x0, 'bounds': bounds}
        return self


    def fixed_thickness(self, value):
        if self.is_substrate:
            raise ValueError(f"Layer '{self.name}' is marked as substrate and cannot have thickness.")
        self.params['thickness'] = {'fit': False, 'value': value}
        return self

    def fit_roughness(self, x0, bounds=None, delta=None):
        if delta is not None:
            bounds = (x0 - delta, x0 + delta)
        if bounds is None:
            raise ValueError("Either bounds or delta must be provided for roughness fitting.")
        self.params['roughness'] = {'fit': True, 'x0': x0, 'bounds': bounds}
        return self


    def fixed_roughness(self, value):
        self.params['roughness'] = {'fit': False, 'value': value}
        return self
    
    def fit_nk_from_material(self, material, energy_pol_uni, bounds_n=None, bounds_k=None,
                            delta_n=None, delta_k=None, density=None):
        self._nk_set = True
        energy_uni = []
        for label in energy_pol_uni:
            energy_str, pol = label.split('_')
            energy = float(energy_str)
            energy_uni.append(energy)
        nk_complex = np.conjugate(xc.refractive_index(material, energy_uni * unit.eV, density=density))
        n_array = 1 - np.real(nk_complex)
        k_array = np.imag(nk_complex)



        bounds_n_extended = [extend_bounds(n_array[i], bounds_n, delta_n) for i in range(len(n_array))]
        bounds_k_extended = [extend_bounds(k_array[i], bounds_k, delta_k) for i in range(len(k_array))]

        self.params['n'] = {'fit': True, 'x0': n_array, 'bounds': bounds_n_extended}
        self.params['k'] = {'fit': True, 'x0': k_array, 'bounds': bounds_k_extended}
        return self


    def fit_nk_array(self, n_array, k_array, bounds_n=None, bounds_k=None,delta_n=None, delta_k=None):
        self._nk_set = True

        bounds_n_extended = [extend_bounds(n_array[i], bounds_n, delta_n) for i in range(len(n_array))]
        bounds_k_extended = [extend_bounds(k_array[i], bounds_k, delta_k) for i in range(len(k_array))]

        self.params['n'] = {'fit': True, 'x0': n_array, 'bounds': bounds_n_extended}
        self.params['k'] = {'fit': True, 'x0': k_array, 'bounds': bounds_k_extended}
        return self
        
    def fixed_nk_from_material(self, material, energy_pol_uni, density=None):
        energy_uni = []
        for label in energy_pol_uni:
            energy_str, pol = label.split('_')
            energy = float(energy_str)
            energy_uni.append(energy)

        nk_complex = np.conjugate(xc.refractive_index(material, energy_uni * unit.eV, density=density))
        n_array = 1-np.real(nk_complex)
        k_array = np.imag(nk_complex)

        return self.fixed_nk(n_array, k_array)


    def fixed_nk(self, n_array, k_array):
        self._nk_set = True
        self.params['n'] = {'fit': False, 'value': n_array}
        self.params['k'] = {'fit': False, 'value': k_array}
        return self

    def validate(self, energy_count):
        if not self._nk_set:
            raise ValueError(f"Layer '{self.name}' must define n and k arrays.")
        if self.params.get('n', {}).get('fit') and len(self.params['n']['x0']) != energy_count:
            raise ValueError(f"Layer '{self.name}' has mismatched n array length.")
        if self.params.get('k', {}).get('fit') and len(self.params['k']['x0']) != energy_count:
            raise ValueError(f"Layer '{self.name}' has mismatched k array length.")