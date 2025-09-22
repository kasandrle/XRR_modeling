import scipy.constants as const
import numpy as np
from scipy.interpolate import interp1d
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
    n_arr = np.array(df['n'])
    k_arr = np.array(df['k'])
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


