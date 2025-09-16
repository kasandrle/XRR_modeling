import sys
sys.path.append("../src")  # relative path from notebooks/ to src/
from reflectivity_model import LayerSpec, ReflectivityModel, eVnm_converter,extract_nk_arrays,load_nk_from_file
import reflectivity_model.utils as utils
import reflectivity_model.xray_compounds as xc
import json

import numpy as np
from os import listdir
from os.path import isfile, join, expanduser
import pandas as pd
import matplotlib.pyplot as plt
import pint
unit = pint.UnitRegistry()
from scipy.optimize import differential_evolution, minimize
import argparse
from pathlib import Path


home = expanduser("~")
sys.path.append(join(home,'Projects/'))
import matrixmethod.mm_numba as mm #https://github.com/mikapfl/matrixmethod

#ToDO: Check for polairzation if nothing is given, set it to s pol 100=s 190=p

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

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# Sample and path configuration
parser = argparse.ArgumentParser(description="Run reflectivity fit for a sample")
parser.add_argument("--sample_name", type=str, required=True, help="Name of the sample (e.g. SOG1)")
parser.add_argument("--path", type=str, required=True, help="Base path to data directory")
parser.add_argument("--input_csv", type=str, required=False, help="Path to input CSV file")
parser.add_argument("--config", type=str, required=False, help="Use a json config file")

args = parser.parse_args()

sample_name = args.sample_name
path = Path(args.path)
#input_file = pd.read_csv(args.input_csv)

print(f"🧪 Running fit for sample: {sample_name}")
print(f"📂 Data path: {path}")
#print(f"📄 Loaded input file with {len(input_file)} rows")

#sample_name = 'SOG1'
#path = '/home/kas/Projects/XRR_photoresist/Underlayer/XRR/reduced/'
#input_file = pd.read_csv('../fit_input/sample1_set_up.csv')

# Filter relevant CSV files
onlyfiles_keys = [
    f for f in listdir(path)
    if isfile(join(path, f)) and f.endswith('.csv') and f.startswith(sample_name)
]
print("Found files:", onlyfiles_keys)

labels = []
exp_data_all = []

# Process each file
for filename in onlyfiles_keys:
    label = filename[:-4]  # strip .csv
    full_path = join(path, filename)
    df = pd.read_csv(full_path)

    # Normalize column names
    df.rename(columns={
        'energy': 'Energy',
        'sam_th': 'Theta'
    }, inplace=True)

    # Ensure polarization column exists
    if 'pol' not in df.columns:
        df['pol'] = 1  # default to s-polarization

    df['pol'] = df['pol'].apply(normalize_polarization)

    # Create combined energy/polarization label
    df['energy_pol'] = df.apply(
        lambda row: f"{row['Energy']}_{'p' if row['pol'] == 0 else 's'}",
        axis=1
    )

    # Filter theta range
    df = df[(df['Theta'] > 2.5) & (df['Theta'] < 30)]

    # Add metadata
    df['file_name'] = label
    df['E_round'] = np.round(df['Energy'], 1)

    labels.append(label)
    exp_data_all.append(df)

# Combine all dataframes
combined_df = pd.concat(exp_data_all, ignore_index=True)

# Extract unique energy values and labels
energy_uni = np.unique(combined_df['Energy'])
energy_pol_uni = np.unique(combined_df['energy_pol'])

# Derive ceremonial file name
split_file = labels[0].split('_')
file_name = '_'.join(split_file[:3])

# Optional: Display unique energy/polarization combinations
print("Unique energy/polarization labels:", energy_pol_uni)

if args.input_csv:
    input_file = pd.read_csv(args.input_csv)
    input_file_col = input_file.columns
    layers = []

    for _, row in input_file.iterrows():
        layer_spec = LayerSpec.from_row(row, energy_pol_uni)
        # print(layer_spec.describe())
        layers.append(layer_spec)

    model = ReflectivityModel(
        energy_pol_uni=energy_pol_uni,
        layers=layers,
        global_params={
            'aoi_offset': {'fit': True, 'x0': 0.0, 'bounds': (-1, 1)},
            'darkcurrent': {'fit': False, 'x0': 0.0},
            'a': {'fit': False, 'x0': 0.045},
            'b': {'fit': False, 'x0': 2.5e-5},
            # 'c': {'fit': False, 'value': 2.5e-5}
        },
        fit_strategy="per_energy",
        sigma_mode="model",
        # sigma_function=custom_sigma
    )

    model.initialize_keys_from_x0()
    model.summarize_stack(energy_pol_uni[0])
    model.describe_sigma()

elif args.config:
    config = load_config(args.config)
    model = ReflectivityModel.from_config(config)

else:
    raise ValueError("You must provide either --input_csv or --config.")


def objective_inner(x, aoi, xrr, sigma_sq, E, model, pol,E_pol, return_R=False):
    # Update model keys with current fit parameters
    for i, param in enumerate([p for p in model.domain_E_init if f'_{E_pol}' in p['name']]):
        model.keys[param['name']] = x[i]

    # Build layer arrays for this energy
    layer, rough, n_stack = model.build_layer_arrays(E_pol)

    # Simulate reflectivity
    wl = utils.eVnm_converter(E)
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

x0_global, bounds_global, global_keys = model.get_global_fit_params()
#objective_model_fit(x0_global,model, combined_df)

x0_energy = [d['x0'] for d in model.domain_E_init]
bounds_energy = [d['domain'] for d in model.domain_E_init]

#model.fit_strategy="per_energy"
x0 =x0_global# + x0_energy
bounds = bounds_global #+ bounds_energy

import multiprocessing

num_workers = multiprocessing.cpu_count()

res = differential_evolution(
    func=objective_model_fit,
    bounds=bounds,
    args=(model, combined_df),
    strategy='best1bin',
    maxiter=500,
    popsize=15,
    tol=1e-6,
    mutation=(0.5, 1.0),
    recombination=0.7,
    polish=False,
    disp=True,
    workers=num_workers
)


x0_global, bounds_global, global_keys = model.get_global_fit_params()
nk_E, R_E, chi_E, chi_total = objective_model_fit(res.x, model, combined_df, return_all=True)

fitted_nk_layers = [
    layer.name
    for layer in model.layers
    if (
        ('n' in layer.params and layer.params['n'].get('fit', False)) or
        ('k' in layer.params and layer.params['k'].get('fit', False))
    )
]

fitted_globals = {
    name: value
    for name, value in zip(global_keys, res.x[:len(global_keys)])
}

fitted_nk = extract_nk_arrays(nk_E, model.energy_pol_uni, fitted_nk_layers)
fit_para = {**fitted_globals, **fitted_nk}

config = model.to_config(fit_para=fit_para)#,x_global = res.x[:len(global_keys)],nk_E=nk_E)
with open(sample_name + "_reflectivity_model_config.json", "w") as f:
    json.dump(config, f, indent=2)

x0_global, bounds_global, global_keys = model.get_global_fit_params()
nk_E, R_E, chi_E, chi_total = objective_model_fit(res.x, model, combined_df, return_all=True)

model.save_all_fit_outputs(
    combined_df=combined_df,
    R_E=R_E,
    nk_E=nk_E,
    x_global=res.x[:len(global_keys)],
    folder_path="../fit_outputs",
    sample_name=sample_name
)