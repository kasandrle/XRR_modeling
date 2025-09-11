import scipy.constants as const

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