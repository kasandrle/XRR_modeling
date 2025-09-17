import numpy as np
import sys
from os.path import join, expanduser
home = expanduser("~")
sys.path.append(join(home,'Projects/'))
import matrixmethod.mm_numba as mm #https://github.com/mikapfl/matrixmethod
from scipy.optimize import minimize
from reflectivity_model import eVnm_converter

def objective_inner(x, aoi, xrr, sigma_sq, E, model, pol,E_pol, return_R=False):
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

def objective_model_fit(x, model, combined_df, return_loglikelihood=False, return_all=False): 
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

        xrr_all = combined_df[combined_df['energy_pol'] == E_pol]
        aoi = np.deg2rad(xrr_all['Theta'].values) - np.deg2rad(model.keys['aoi_offset'])
        xrr = xrr_all['R'].values
        if model.sigma_mode == "model":
            sigma_sq = np.square(model.keys['a'] * xrr) + np.square(model.keys['b'])
        elif model.sigma_mode == "column":
            if model.sigma_column is None:
                raise ValueError("sigma_column must be specified when sigma_mode='column'")
            if model.sigma_column not in xrr_all.columns:
                raise ValueError(f"Column '{model.sigma_column}' not found in combined_df")
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
                objective_inner, x0, bounds=bounds,
                args=(aoi, xrr, sigma_sq, E, model, pol, E_pol),
                method='TNC'
            )

            # Evaluate result
            if res.success:
                last_successful_x = res.x
            else:
                if last_successful_x is not None:
                    res = minimize(
                        objective_inner, last_successful_x, bounds=bounds,
                        args=(aoi, xrr, sigma_sq, E, model, pol, E_pol),
                        method='TNC'
                    )
                    if res.success:
                        last_successful_x = res.x
                else:
                    x_random = [np.random.uniform(low, high) for (low, high) in bounds]
                    res = minimize(
                        objective_inner, x_random, bounds=bounds,
                        args=(aoi, xrr, sigma_sq, E, model, pol, E_pol),
                        method='TNC'
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

        chi_total += chi_E_val

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