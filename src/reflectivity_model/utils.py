import scipy.constants as const
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erf
import ast
import pandas as pd

# Precompute hc in nm·eV
hc = const.h * const.c / const.e * 1e9  # Planck × speed of light / charge × 1e9

def eVnm_converter(value):
    """Convert photon energy in eV to wavelength in nm."""
    return hc / value


def extend_bounds(x0_val, bound=None, delta=None):
    if delta is not None:
        return (x0_val - delta, x0_val + delta)
    if bound is not None:
        lower, upper = bound
        return (min(lower, x0_val), max(upper, x0_val))
    raise ValueError("Either bounds or delta must be provided.")

def fmt_bounds(bounds, precision=3):
    if isinstance(bounds, tuple):
        return f"[{bounds[0]:.{precision}f}, {bounds[1]:.{precision}f}]"
    return str(bounds)
def fmt(val, precision=3):
    return f"{val:.{precision}f}" if isinstance(val, (int, float)) else str(val)

def safe_serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_serialize(v) for v in obj]
    else:
        return obj
    
def extract_nk_arrays(nk_E, energy_pol_uni, fitted_nk_layers):
    nk_dict = {lname: {"n_array": [], "k_array": []} for lname in fitted_nk_layers}

    for e_idx in range(len(energy_pol_uni)):
        nk_vals = nk_E[e_idx]

        for i, lname in enumerate(fitted_nk_layers):
            n_val = nk_vals[2 * i]
            k_val = nk_vals[2 * i + 1]

            nk_dict[lname]["n_array"].append(n_val)
            nk_dict[lname]["k_array"].append(k_val)

    return nk_dict

def load_nk_from_file(filepath,energy_pol_uni):
    energy_uni = []
    for label in energy_pol_uni:
        energy_str, pol = label.split('_')
        energy = float(energy_str)
        energy_uni.append(energy)
    df = pd.read_csv(filepath)
    e_arr = np.array(df['Energy'])
    n_arr = np.array(df['delta'])
    k_arr = np.array(df['beta'])
    n_interp = interp1d(e_arr, n_arr, fill_value="extrapolate")
    k_interp = interp1d(e_arr, k_arr, fill_value="extrapolate")

    n_array_extended = n_interp(energy_uni)
    k_array_extended = k_interp(energy_uni)
    return n_array_extended, k_array_extended

def normalize_polarization(pol_entry):
    """
    Normalize polarization input to binary:
    - Returns 1 for s-polarization
    - Returns 0 for p-polarization
    - Defaults to s if unclear
    """
    entry = str(pol_entry).strip().lower()
    if any(key in entry for key in ['100', 's', '1']):
        return 1
    elif any(key in entry for key in ['190', 'p', '0']):
        return 0
    return 1  # default to s-pol if ambiguous

def validate_positive(value, field_name="value"):
    if value is None:
        raise ValueError(f"{field_name} is missing.")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive. Got {value}.")
    
def smooth_transition(z, z0, width, val1, val2):
    """Smooth transition between val1 and val2 centered at z0 with roughness width."""
    return val1 + (val2 - val1) * 0.5 * (1 + erf((z - z0) / (np.sqrt(2) * width)))

def get_thick_rough(layer):
    thickness = 0
    roughness = 0
    if "fit_thickness" in layer and "x0" in layer["fit_thickness"]:
        thickness = layer["fit_thickness"]["x0"]
    elif "fixed_thickness" in layer:
        thickness = layer["fixed_thickness"]

    if "fit_roughness" in layer and "x0" in layer["fit_roughness"]:
        roughness = layer["fit_roughness"]["x0"]
    elif "fixed_roughness" in layer:
        roughness = layer["fixed_roughness"]
    return thickness, roughness

def get_nk(layer,target_energy_index=0):
    if "fit_nk_array" in layer:
        n = layer["fit_nk_array"]["n_array"][target_energy_index]
        k = layer["fit_nk_array"]["k_array"][target_energy_index]
    else:
        n = layer["fixed_nk"]["n_array"][target_energy_index]
        k = layer["fixed_nk"]["k_array"][target_energy_index]
    return n, k
    


def build_centered_erf_profile(layers, energies, target_energy_index=0, resolution=0.1):
    # Parameters
    vacuum_thickness = 10
    substrate_extension = 10
    vacuum_n, vacuum_k = 0.0, 0.0

    # Compute total stack thickness
    stack_thickness = sum(
        layer.get("fit_thickness", {}).get("x0", layer.get("fixed_thickness", 0))
        for layer in layers
    )
    total_depth = vacuum_thickness + stack_thickness + substrate_extension
    z_grid = np.arange(-vacuum_thickness, stack_thickness + substrate_extension, resolution)

    n_profile = np.full_like(z_grid, vacuum_n, dtype=float)
    k_profile = np.full_like(z_grid, vacuum_k, dtype=float)



    z_current = 0
    prev_n, prev_k = vacuum_n, vacuum_k

    for i, layer in enumerate(layers[:-1]):
        #print(layer)
        #thickness = layer.get("fit_thickness").get("x0", layer.get("fixed_thickness"))
        #roughness = layer.get("fit_roughness").get("x0", layer.get("fixed_roughness"))
        thickness, roughness = get_thick_rough(layer)
        n_val, k_val = get_nk(layer,target_energy_index=target_energy_index)

        # Fill interior of layer
        for j, z in enumerate(z_grid):
            if z_current <= z < z_current + thickness:
                n_profile[j] = n_val
                k_profile[j] = k_val

        # Apply smooth transition centered at interface
        z_interface = z_current
        width = roughness / 2
        for j, z in enumerate(z_grid):
            if z_interface - 3 * width < z < z_interface + 3 * width:
                alpha = 0.5 * (1 + erf((z - z_interface) / (np.sqrt(2) * width)))
                n_profile[j] = (1 - alpha) * prev_n + alpha * n_val
                k_profile[j] = (1 - alpha) * prev_k + alpha * k_val

        z_current += thickness
        prev_n, prev_k = n_val, k_val

    # Final transition to substrate extension
    last_layer = layers[-1]
    rough_substrate = last_layer.get("fit_roughness", {}).get("x0", last_layer.get("fixed_roughness", 0.01))
    #print(rough_substrate)
    sub_n, sub_k = get_nk(last_layer,target_energy_index=target_energy_index)
    z_interface = z_current
    width = rough_substrate / 2

    for j, z in enumerate(z_grid):
        if z_interface - 3 * width < z < z_interface + 3 * width:
            alpha = 0.5 * (1 + erf((z - z_interface) / (np.sqrt(2) * width)))
            n_profile[j] = (1 - alpha) * prev_n + alpha * sub_n
            k_profile[j] = (1 - alpha) * prev_k + alpha * sub_k
        elif z >= z_interface + 3 * width:
            n_profile[j] = sub_n
            k_profile[j] = sub_k

    return z_grid, n_profile, k_profile


