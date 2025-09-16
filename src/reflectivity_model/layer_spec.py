from . import xray_compounds as xc
import numpy as np
import pint
unit = pint.UnitRegistry()
import ast
import pandas as pd
from .utils import extend_bounds,load_nk_from_file


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
        self.params['thickness'] = {'fit': False, 'x0': value}
        return self

    def fit_roughness(self, x0, bounds=None, delta=None):
        if delta is not None:
            bounds = (x0 - delta, x0 + delta)
        if bounds is None:
            raise ValueError("Either bounds or delta must be provided for roughness fitting.")
        self.params['roughness'] = {'fit': True, 'x0': x0, 'bounds': bounds}
        return self


    def fixed_roughness(self, value):
        self.params['roughness'] = {'fit': False, 'x0': value}
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
        self.params['n'] = {'fit': False, 'x0': n_array}
        self.params['k'] = {'fit': False, 'x0': k_array}
        return self

    def validate(self, energy_count):
        if not self._nk_set:
            raise ValueError(f"Layer '{self.name}' must define n and k arrays.")
        if self.params.get('n', {}).get('fit') and len(self.params['n']['x0']) != energy_count:
            raise ValueError(f"Layer '{self.name}' has mismatched n array length.")
        if self.params.get('k', {}).get('fit') and len(self.params['k']['x0']) != energy_count:
            raise ValueError(f"Layer '{self.name}' has mismatched k array length.")

    def describe(self):
        lines = [f"Layer: {self.name}"]

        if getattr(self, "is_substrate", False):
            lines.append("Declared as substrate (no thickness fitting)")
        else:
            if "thickness" in self.params:
                mode = "fit" if self.params["thickness"].get("fit") else "fixed"
                lines.append(f"Thickness: {mode} → {self.params['thickness']}")

        if "roughness" in self.params:
            mode = "fit" if self.params["roughness"].get("fit") else "fixed"
            lines.append(f"Roughness: {mode} → {self.params['roughness']}")
            # Optical constants
        if "n" in self.params:
            mode = "fit" if self.params["n"].get("fit") else "fixed"
            lines.append(f"delta: {mode} → {self.params['n']}")
        if "k" in self.params:
            mode = "fit" if self.params["k"].get("fit") else "fixed"
            lines.append(f"beta: {mode} → {self.params['k']}")

        return "\n".join(lines)


    @classmethod
    def from_row(cls, row, energy_pol_uni):
        def parse_bounds_or_delta(value):
            if pd.isna(value):
                return None, None
            if isinstance(value, str):
                value = ast.literal_eval(value)
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return 'bounds', tuple(value)
            elif isinstance(value, (int, float)):
                return 'delta', value
            return None, None

        layer = cls(row['name'])

        # ─── Substrate ────────────────────────────────────────────────────────
        thickness = row.get('thickness')
        if pd.isna(thickness):
            layer.is_substrate = True
        else:   
            # ─── Thickness ────────────────────────────────────────────────────────
            fit_thickness_type, fit_thickness_val = parse_bounds_or_delta(row.get('fit_thickness'))
            if fit_thickness_type == 'delta':
                layer = layer.fit_thickness(thickness, delta=fit_thickness_val)
            elif fit_thickness_type == 'bounds':
                layer = layer.fit_thickness(thickness, bounds=fit_thickness_val)
            else:
                layer = layer.fixed_thickness(thickness)

        # ─── Roughness ────────────────────────────────────────────────────────
        roughness = row.get('roughness', 0.01)
        fit_roughness_type, fit_roughness_val = parse_bounds_or_delta(row.get('fit_roughness'))
        if fit_roughness_type == 'delta':
            layer = layer.fit_roughness(roughness, delta=fit_roughness_val)
        elif fit_roughness_type == 'bounds':
            layer = layer.fit_roughness(roughness, bounds=fit_roughness_val)
        else:
            layer = layer.fixed_roughness(roughness)

        # ─── Optical Constants (n/k) ──────────────────────────────────────────
        fit_n_type, fit_n_val = parse_bounds_or_delta(row.get('fit_n'))
        fit_k_type, fit_k_val = parse_bounds_or_delta(row.get('fit_k'))

        nk_source = row.get('nk')
        density = row.get('density')

        if pd.isna(row.get('fit_n')) and pd.isna(row.get('fit_k')):
            if pd.isna(density):
                n_arr, k_arr = load_nk_from_file(nk_source, energy_pol_uni)
                layer = layer.fixed_nk_array(n_arr, k_arr)
            else:
                layer = layer.fixed_nk_from_material(nk_source, energy_pol_uni, density=density)
        else:
            delta_n = fit_n_val if fit_n_type == 'delta' else None
            bounds_n = fit_n_val if fit_n_type == 'bounds' else None
            delta_k = fit_k_val if fit_k_type == 'delta' else None
            bounds_k = fit_k_val if fit_k_type == 'bounds' else None

            if pd.isna(density):
                n_arr, k_arr = load_nk_from_file(nk_source, energy_pol_uni)
                layer = layer.fit_nk_array(
                    n_arr, k_arr,
                    delta_n=delta_n, delta_k=delta_k,
                    bounds_n=bounds_n, bounds_k=bounds_k
                )
            else:
                layer = layer.fit_nk_from_material(
                    nk_source, energy_pol_uni, density=density,
                    delta_n=delta_n, delta_k=delta_k,
                    bounds_n=bounds_n, bounds_k=bounds_k
                )

        return layer
