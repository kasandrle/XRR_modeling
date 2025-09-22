import sys
sys.path.append("../src")  # relative path from notebooks/ to src/
from reflectivity_model import LayerSpec, ReflectivityModel, eVnm_converter,extract_nk_arrays,load_nk_from_file, normalize_polarization
import reflectivity_model.utils as utils
import reflectivity_model.xray_compounds as xc
import json
from datetime import datetime
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import pint
unit = pint.UnitRegistry()
from scipy.optimize import differential_evolution
import argparse
from pathlib import Path

import objective


#home = expanduser("~")
#sys.path.append(join(home,'Projects/'))
#import matrixmethod.mm_numba as mm #https://github.com/mikapfl/matrixmethod

#ToDO: Check for polairzation if nothing is given, set it to s pol 100=s 190=p


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# Sample and path configuration
parser = argparse.ArgumentParser(description="Run reflectivity fit for a sample")
parser.add_argument("--sample_name", type=str, required=True, help="Name of the sample (e.g. UL)")
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
    func=objective.objective_model_fit,
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
nk_E, R_E, chi_E, chi_total = objective.objective_model_fit(res.x, model, combined_df, return_all=True)

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
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"../fit_outputs/{sample_name}_reflectivity_model_config_{timestamp}.json", "w") as f:
    json.dump(config, f, indent=2)

x0_global, bounds_global, global_keys = model.get_global_fit_params()
nk_E, R_E, chi_E, chi_total = objective.objective_model_fit(res.x, model, combined_df, return_all=True)

model.save_all_fit_outputs(
    combined_df=combined_df,
    R_E=R_E,
    nk_E=nk_E,
    x_global=res.x[:len(global_keys)],
    folder_path="../fit_outputs",
    sample_name=sample_name
)