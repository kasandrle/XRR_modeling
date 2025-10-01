import numpy as np
#import sys
#from os.path import join, expanduser
#home = expanduser("~")
#sys.path.append(join(home,'Projects/'))
import matrixmethod.mm_numba as mm #https://github.com/mikapfl/matrixmethod
from scipy.optimize import minimize
from reflectivity_model import eVnm_converter

def objective_inner(x, aoi, xrr, sigma_sq, E, model, pol,E_pol, return_R=False):
    """
    Computes the chi-squared error between simulated and experimental reflectivity
    for a single energy-polarization combination.

    This function updates the model with energy-specific fit parameters, constructs
    the multilayer stack, simulates reflectivity using the transfer matrix method,
    and compares it to experimental data using a weighted error model.

    Args:
        x (array-like): Fit parameters specific to this energy-polarization slice.
        aoi (array-like): Array of angles of incidence (in radians), adjusted for offset.
        xrr (array-like): Experimental reflectivity values.
        sigma_sq (array-like): Error variance for each reflectivity point.
        E (float): Energy value (e.g., in eV) for this slice.
        model (ReflectivityModel): Instance of the reflectivity model.
        pol (int): Polarization index (0 = p-pol, 1 = s-pol).
        E_pol (str): Unique identifier for energy-polarization combination.
        return_R (bool): If True, also returns the simulated reflectivity array.

    Returns:
        float or tuple:
            - If return_R is False: returns chi-squared error (float)
            - If return_R is True: returns (chi-squared, simulated reflectivity)
    """

    # Update model keys with current fit parameters
    for i, param in enumerate([p for p in model.domain_E_init if f'_{E_pol}' in p['name']]):
        model.keys[param['name']] = x[i]

    # Build layer arrays for this energy
    layer, rough, n_stack = model.build_layer_arrays(E_pol)

    # Simulate reflectivity
    wl = eVnm_converter(E)
    rm, tm = mm.reflec_and_trans(n_stack, wl, aoi, layer, rough, pol=pol)
    rm = np.square(np.abs(np.asarray(rm)))

    chi = np.sum(np.square(rm - xrr) / sigma_sq)
    return (chi, rm) if return_R else chi

def objective_inner_safe(x, *args):
    if not np.all(np.isfinite(x)):
        print("Non-finite x detected:", x)
        return np.inf
        #raise ValueError("x contains nan or inf")
    result = objective_inner(x, *args)
    if not np.isfinite(result):
        print("Objective returned non-finite value for x =", x)
        raise ValueError("Objective returned nan or inf")
    return result



def objective_model_fit(x, model, exp_reflectivity, return_loglikelihood=False, return_all=False): 
    """
    Computes the objective function for multilayer reflectivity fitting across multiple energies.

    This function supports both global and per-energy fitting strategies. It evaluates the
    chi-squared error between simulated and experimental reflectivity data, optionally returning
    log-likelihood or detailed outputs for analysis and visualization.

    Args:
        x (array-like): Flattened array of fit parameters. Includes global parameters and,
            if strategy is "global", energy-dependent parameters.
        model (ReflectivityModel): Instance of the reflectivity model containing layer definitions,
            parameter domains, and fitting configuration.
        exp_reflectivity (pd.DataFrame): DataFrame containing experimental reflectivity data.
            Must include columns: 'energy_pol', 'Theta', 'R', and optionally a sigma column.
        return_loglikelihood (bool): If True, returns the log-likelihood instead of chi-squared.
        return_all (bool): If True, returns detailed outputs including nk arrays, simulated R,
            experimental R, per-energy chi values, and total chi.

    Returns:
        float or tuple:
            - If return_loglikelihood is True: returns total log-likelihood.
            - If return_all is True: returns (nk_E, R_E, chi_E, chi_total)
            - Otherwise: returns total chi-squared error (float)

    Raises:
        ValueError: If input parameter lengths do not match expected counts based on fit strategy.
        ValueError: If sigma_mode is invalid or required sigma inputs are missing.
    """

    # Validate input length based on fit strategy
    global_keys = [k for k, v in model.global_params.items() if v.get('fit')]
    n_global = len(global_keys)
    n_energy = len(model.domain_E_init)  # total energy-dependent params

    if model.fit_strategy == "per_energy":
        if len(x) != n_global:
            raise ValueError(f"Expected {n_global} global parameters for 'per_energy' strategy, got {len(x)}")
    elif model.fit_strategy == "global":
        expected_total = n_global + n_energy
        if len(x) != expected_total:
            raise ValueError(f"Expected {expected_total} parameters (global + energy-dependent) for 'global' strategy, got {len(x)}")
    else:
        raise ValueError(f"Unknown fit strategy: {model.fit_strategy}")

    x_global = x[:n_global]
    x_energy = x[n_global:]

    # Update global parameters
    for i, key in enumerate(global_keys):
        model.keys[key] = x_global[i]

    # Initialize outputs
    chi_total = 0
    loglikelihood = 0
    if return_all:
        nk_E, R_E, R_exp_E, chi_E = [], [], [], []

    # Loop over energies
    last_successful_x = None
    for E_pol in model.energy_pol_uni:

        i = model.energy_index_map[E_pol]['index']
        E = model.energy_index_map[E_pol]['energy']
        pol = model.energy_index_map[E_pol]['pol_number']  #1 = s_plo, 0 = p_pol

        xrr_all = exp_reflectivity[exp_reflectivity['energy_pol'] == E_pol]
        aoi = np.deg2rad(xrr_all['Theta'].values) - np.deg2rad(model.keys['aoi_offset'])
        xrr = xrr_all['R'].values
        if model.sigma_mode == "model":
            sigma_sq = np.square(model.keys['a'] * xrr) + np.square(model.keys['b'])
        elif model.sigma_mode == "column":
            if model.sigma_column is None:
                raise ValueError("sigma_column must be specified when sigma_mode='column'")
            if model.sigma_column not in xrr_all.columns:
                raise ValueError(f"Column '{model.sigma_column}' not found in exp_reflectivity")
            sigma_sq = xrr_all[model.sigma_column].values
        elif model.sigma_mode == "function":
            if not callable(model.sigma_function):
                raise ValueError("sigma_function must be a callable when sigma_mode='function'")
            sigma_sq = model.sigma_function(xrr_all['R'].values, model)
        else:
            raise ValueError(f"Unknown sigma_mode: {model.sigma_mode}")


        if model.fit_strategy == "per_energy":
            x0 = [d['x0'] for d in model.domain_E_init if f'_{E_pol}' in d['name']]
            bounds = [d['domain'] for d in model.domain_E_init if f'_{E_pol}' in d['name']]

            res = minimize(
                objective_inner_safe, x0, bounds=bounds,
                args=(aoi, xrr, sigma_sq, E, model, pol, E_pol),
                method='L-BFGS-B'
            )

            # Evaluate result
            if res.success:
                last_successful_x = res.x
            else:
                if last_successful_x is not None and np.all(np.isfinite(last_successful_x)):
                    res = minimize(
                        objective_inner_safe, last_successful_x, bounds=bounds,
                        args=(aoi, xrr, sigma_sq, E, model, pol, E_pol),
                        method='L-BFGS-B'
                    )
                    if res.success:
                        last_successful_x = res.x
                else:
                    x_random = [np.random.uniform(low, high) for (low, high) in bounds]
                    if not np.all(np.isfinite(x_random)):
                        print("Non-finite x_random detected:", x_random)
                        raise ValueError("x_random contains nan or inf")
                    res = minimize(
                        objective_inner_safe, x_random, bounds=bounds,
                        args=(aoi, xrr, sigma_sq, E, model, pol, E_pol),
                        method='L-BFGS-B'
                    )
                    if res.success:
                        last_successful_x = res.x

            chi_E_val, rm = objective_inner(res.x, aoi, xrr, sigma_sq, E, model, pol, E_pol, return_R=True)

            if return_all:
                nk_E.append(res.x)


        elif model.fit_strategy == "global":
            # Use energy-dependent parameters from x
            energy_params = [p for p in model.domain_E_init if f'_{E_pol}' in p['name']]
            for i, param in enumerate(energy_params):
                model.keys[param['name']] = x_energy[i]

            wl = eVnm_converter(E)
            layer, rough, n_stack = model.build_layer_arrays(E_pol)
            rm, _ = mm.reflec_and_trans(n_stack, wl, aoi, layer, rough, pol=pol)
            rm = np.square(np.abs(np.asarray(rm)))
            chi_E_val = np.sum(np.square(rm - xrr) / sigma_sq)

            if return_all:
                nk_E.append(x_energy)

        else:
            raise ValueError(f"Unknown fit_strategy: {model.fit_strategy}")

        chi_total += chi_E_val#/len(xrr)

        if return_loglikelihood:
            loglikelihood += np.sum(np.log(1 / np.sqrt(2 * np.pi * sigma_sq)) - chi_E_val / 2)

        if return_all:
            R_E.append(rm)
            R_exp_E.append(xrr)
            chi_E.append(chi_E_val)

    if return_loglikelihood:
        return loglikelihood
    if return_all:
        return np.asarray(nk_E), R_E, chi_E, chi_total

    return chi_total