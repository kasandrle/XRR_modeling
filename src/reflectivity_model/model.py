import numpy as np
import pandas as pd
import os
from datetime import datetime
from .utils import fmt,fmt_bounds

class ReflectivityModel:
    """
    Constructs and manages a multilayer reflectivity model for energy-dependent fitting.

    This class organizes layer specifications, global parameters, and energy-resolved domains
    to support inverse modeling of reflectivity data (e.g. RSoXS, XRR). It supports both
    global and per-energy fitting strategies, and allows flexible error modeling via sigma_mode.

    Attributes:
        energy_points (list of float): Discrete energy values (e.g. in eV) used for modeling.
        layers (list of LayerSpec): Ordered list of layer definitions, including substrate.
        global_params (dict): Dictionary of global fit parameters (e.g. thickness, roughness, offsets).
        fit_strategy (str): Either "global" or "per_energy", controlling how energy-dependent parameters are fit.
        sigma_mode (str): Defines how error variance (σ²) is computed—either "model" or "column".
        sigma_column (str or None): Name of the column in combined_df to use if sigma_mode="column".
        sigma_function (function of R, model): User specific function of sigma , needs refelctivity and model.
        keys (dict): Centralized parameter dictionary used during fitting and simulation.
        domain_E_init (list): Energy-dependent parameter definitions (name, x0, bounds).
        x0_dict (dict): Mapping of energy to initial parameter values.
        energy_index_map (dict): Maps energy values to their index for array slicing.

    Methods:
        _validate_layers(): Ensures layer definitions are complete and stack-level consistency is maintained.
        _inject_layer_params(): Transfers layer-level fit parameters into the global parameter dictionary.
        _build_energy_domains(): Constructs energy-dependent parameter domains for n and k.
        _build_global_keys(): Initializes the keys dictionary from global and energy-dependent parameters.
        _validate_sigma_model(): checks if a and b are in global parameters
        initialize_keys_from_x0(): Populates keys from initial values for fitting.
        build_layer_arrays(E): Constructs thickness, roughness, and complex refractive index stack for a given energy.
        describe_sigma(): Returns a human-readable description of the current sigma error model.
        summarize_stack(E): prints out the values for one energy
        get_global_fit_params(): Returns x0, bounds, and keys for all global parameters marked for fitting.
    """

    def __init__(self, energy_pol_uni, layers, global_params, fit_strategy="global", sigma_mode="model", sigma_column=None, sigma_function=None):
        self.energy_pol_uni = energy_pol_uni
        self.layers = layers
        self.global_params = global_params
        self.fit_strategy = fit_strategy
        self.sigma_mode = sigma_mode
        self.sigma_column = sigma_column
        self.sigma_function = sigma_function

        self.keys = {}
        self.domain_E_init = []
        self.x0_dict = {}
        self.energy_index_map = {
            energy_pol: {
                "index": i,
                "energy": float(energy_pol.split('_')[0]),
                "pol": energy_pol.split('_')[1],
                "pol_number": 1 if energy_pol.split('_')[1] == 's' else 0
            }
            for i, energy_pol in enumerate(energy_pol_uni)
        }


        self._validate_layers()
        self._inject_layer_params()
        self._build_energy_domains()
        self._build_global_keys()   
        self._validate_sigma_model()     

    def _validate_layers(self):
        energy_count = len(self.energy_pol_uni)

        # Validate individual layers
        for layer in self.layers:
            layer.validate(energy_count)

        # Stack-level consistency checks
        n_layers = len(self.layers)
        n_thickness = sum('thickness' in layer.params for layer in self.layers if not layer.is_substrate)
        n_roughness = sum('roughness' in layer.params for layer in self.layers)
        n_n_arrays = sum('n' in layer.params for layer in self.layers)
        n_k_arrays = sum('k' in layer.params for layer in self.layers)

        if n_thickness != n_layers - 1:  # substrate excluded
            raise ValueError(f"Expected {n_layers - 1} thickness entries, found {n_thickness}.")

        if n_roughness != n_layers:
            raise ValueError(f"Expected {n_layers} roughness entries (interfaces), found {n_roughness}.")

        if n_n_arrays != n_layers or n_k_arrays != n_layers:
            raise ValueError(f"Expected n/k arrays for all {n_layers} layers, found {n_n_arrays} n and {n_k_arrays} k.")
        
    def _validate_sigma_model(self):
        if self.sigma_mode == "model":
            missing = [p for p in ['a', 'b'] if p not in self.global_params]
            if missing:
                raise ValueError(f"Missing global parameters for sigma_mode='model': {missing}")
        elif self.sigma_mode == "column":
            forbidden = [p for p in ['a', 'b'] if p in self.global_params]
            if forbidden:
                raise ValueError(f"Global parameters {forbidden} should not be defined when sigma_mode='column'")
        elif self.sigma_mode == "function":
                if not callable(self.sigma_function):
                    raise ValueError("sigma_function must be a callable when sigma_mode='function'")

        else:
            raise ValueError(f"Unknown sigma_mode: {self.sigma_mode}")
        
    def _inject_layer_params(self):
        for layer in self.layers:
            lname = layer.name

            # Thickness
            if 'thickness' in layer.params and layer.params['thickness']['fit']:
                self.global_params[f'thickness_{lname}'] = {
                    'fit': True,
                    'x0': layer.params['thickness']['x0'],
                    'bounds': layer.params['thickness']['bounds']
                }

            # Roughness
            if 'roughness' in layer.params and layer.params['roughness']['fit']:
                self.global_params[f'rough_{lname}'] = {
                    'fit': True,
                    'x0': layer.params['roughness']['x0'],
                    'bounds': layer.params['roughness']['bounds']
                }


    def _build_energy_domains(self):
        for layer in self.layers:
            lname = layer.name
            for i, E in enumerate(self.energy_pol_uni):
                if layer.params.get('n', {}).get('fit'):
                    self.domain_E_init.append({
                        'name': f'n_{lname}_{E}',
                        'x0': layer.params['n']['x0'][i],
                        'domain': layer.params['n']['bounds'][i]
                    })
                if layer.params.get('k', {}).get('fit'):
                    self.domain_E_init.append({
                        'name': f'k_{lname}_{E}',
                        'x0': layer.params['k']['x0'][i],
                        'domain': layer.params['k']['bounds'][i]
                    })
        for E in self.energy_pol_uni:
            self.x0_dict[E] = [d['x0'] for d in self.domain_E_init if f'_{E}' in d['name']]

    def _build_global_keys(self):
        for param, spec in self.global_params.items():
            if spec.get('fit'):
                self.keys[param] = spec['x0']
            else:
                self.keys[param] = spec['value']
        self.keys['x0'] = self.x0_dict
        self.keys['E_pol_uni'] = self.energy_pol_uni

    def initialize_keys_from_x0(self):
        # Energy-dependent parameters
        for param in self.domain_E_init:
            self.keys[param['name']] = param['x0']

        # Global parameters
        for key, spec in self.global_params.items():
            if spec.get('fit'):
                self.keys[key] = spec['x0']
            else:
                self.keys[key] = spec['value']

        # Layer-level global parameters (thickness, roughness)
        for layer in self.layers:
            lname = layer.name

            # Thickness
            if 'thickness' in layer.params and layer.params['thickness']['fit']:
                self.keys[f'thickness_{lname}'] = layer.params['thickness']['x0']

            # Roughness
            if 'roughness' in layer.params and layer.params['roughness']['fit']:
                self.keys[f'rough_{lname}'] = layer.params['roughness']['x0']



    def build_layer_arrays(self, E):
        layer = []
        rough = []
        n_stack = [1 + 0j]  # ambient

        i = self.energy_index_map[E]['index']

        for layer_spec in self.layers:
            lname = layer_spec.name

            # Thickness
            if 'thickness' in layer_spec.params:
                if layer_spec.params['thickness']['fit']:
                    layer.append(self.keys[f'thickness_{lname}'])
                else:
                    layer.append(layer_spec.params['thickness']['value'])

            # Roughness
            if 'roughness' in layer_spec.params:
                if layer_spec.params['roughness']['fit']:
                    rough.append(self.keys[f'rough_{lname}'])
                else:
                    rough.append(layer_spec.params['roughness']['value'])

            # Optical constants
            if layer_spec.params['n']['fit']:
                n_real = self.keys[f'n_{lname}_{E}']
            else:
                n_real = layer_spec.params['n']['value'][i]

            if layer_spec.params['k']['fit']:
                n_imag = self.keys[f'k_{lname}_{E}']
            else:
                n_imag = layer_spec.params['k']['value'][i]


            n_stack.append(1 - n_real + n_imag * 1j)

        return np.array(layer), np.array(rough), np.array(n_stack)
    
    def summarize_stack(self, E_pol):
        i = self.energy_index_map[E_pol]['index']
        print(f"\n📐 Stack Summary at {E_pol} eV")
        print(f"{'Layer':<14} | {'Thick.':>8} : {'Bounds':>17} | {'Rough.':>8} : {'Bounds':>17} | {'Δ (n)':>8} : {'Bounds':>17} | {'β (k)':>8} : {'Bounds':>17}")
        print("-" * 120)

        for layer_spec in self.layers:
            lname = layer_spec.name

            # Thickness
            if 'thickness' in layer_spec.params:
                if layer_spec.params['thickness']['fit']:
                    thickness = self.keys[f'thickness_{lname}']
                    bounds_thick = layer_spec.params['thickness']['bounds']
                else:
                    thickness = layer_spec.params['thickness']['value']
                    bounds_thick = '—'
            else:
                thickness = '—'
                bounds_thick = '—'

            # Roughness
            if 'roughness' in layer_spec.params:
                if layer_spec.params['roughness']['fit']:
                    roughness = self.keys[f'rough_{lname}']
                    bounds_rough = layer_spec.params['roughness']['bounds']
                else:
                    roughness = layer_spec.params['roughness']['value']
                    bounds_rough = '—'
            else:
                roughness = '—'
                bounds_rough = '—'

            # Optical constants
            if layer_spec.params['n']['fit']:
                n_val = self.keys[f'n_{lname}_{E_pol}']
                bounds_n = layer_spec.params['n']['bounds'][i]
            else:
                n_val = layer_spec.params['n']['value'][i]
                bounds_n = '—'

            if layer_spec.params['k']['fit']:
                k_val = self.keys[f'k_{lname}_{E_pol}']
                bounds_k = layer_spec.params['k']['bounds'][i]
            else:
                k_val = layer_spec.params['k']['value'][i]
                bounds_k = '—'

            print(f"🔹 {lname:<14} | {fmt(thickness):>8} : {fmt_bounds(bounds_thick):>17} | {fmt(roughness):>8} : {fmt_bounds(bounds_rough):>17} | {fmt(n_val, 4):>8} : {fmt_bounds(bounds_n, 4):>17} | {fmt(k_val, 4):>8} : {fmt_bounds(bounds_k, 4):>17}")

    def describe_sigma(self):
        if self.sigma_mode == "model":
            return "σ² = (a·R)² + b² (model-based error)"
        elif self.sigma_mode == "column":
            return f"σ² from column '{self.sigma_column}' in experimental data"
        elif self.sigma_mode == "function":
            func_name = getattr(self.sigma_function, '__name__', 'custom_sigma')
            return f"σ² computed via custom function '{func_name}' — may depend on global parameters like a, b, c..."
        else:
            return "Unknown sigma mode"


    def get_global_fit_params(self):
        """
        Returns x0, bounds, and keys for all global parameters marked for fitting.
        """
        x0 = []
        bounds = []
        keys = []

        for key, spec in self.global_params.items():
            if spec.get('fit', False):
                x0.append(spec['x0'])
                bounds.append(spec['bounds'])
                keys.append(key)

        return x0, bounds, keys

    
    #____save to dataframe_____________

    def get_reflectivity_dataframe(self, combined_df, R_E, save_path=None):
        records = []
        for e_idx, E_pol in enumerate(self.energy_pol_uni):
            E = self.energy_index_map[E_pol]['energy']
            pol = self.energy_index_map[E_pol]['pol']
            xrr_all = combined_df[combined_df['energy_pol'] == E_pol]
            aoi_deg = xrr_all['Theta'].values
            xrr = xrr_all['R'].values
            rm = R_E[e_idx]
            for i, theta in enumerate(aoi_deg):
                records.append({
                    'energy': E,
                    'pol':pol,
                    'aoi': theta,
                    'R_sim': rm[i],
                    'R_exp': xrr[i],
                    'residual': rm[i] - xrr[i]
                })
        df = pd.DataFrame(records)
        if save_path:
            df.to_csv(save_path, index=False)
        return df

    def get_nk_long_dataframe(self, nk_E, save_path=None):
        records = []

        for e_idx, E_pol in enumerate(self.energy_pol_uni):
            E = self.energy_index_map[E_pol]['energy']
            pol = self.energy_index_map[E_pol]['pol']
            nk_vals = nk_E[e_idx]

            # Track fitted values
            energy_params = [p for p in self.domain_E_init if f'_{E_pol}' in p['name']]
            layer_nk = {}

            for i, param in enumerate(energy_params):
                name = param['name']
                if 'n_' in name:
                    lname = name.split('n_')[1]
                    layer_nk.setdefault(lname, {})['n'] = nk_vals[i]
                elif 'k_' in name:
                    lname = name.split('k_')[1]
                    layer_nk.setdefault(lname, {})['k'] = nk_vals[i]

            # Add fitted values — guaranteed to be present
            for lname, nk in layer_nk.items():
                n_val = nk.get('n')
                k_val = nk.get('k')

                if n_val is None or k_val is None:
                    raise ValueError(f"Missing fitted n or k for layer '{lname}' at energy '{E_pol}'")

                records.append({
                    'energy': E,
                    'pol': pol,
                    'layer': lname,
                    'n': n_val,
                    'k': k_val,
                    'fit': True
                })

            # Add fixed values — only if not already added
            for layer_spec in self.layers:
                lname = layer_spec.name
                if lname in layer_nk:
                    continue  # already added as fitted

                # Only add if n/k are fixed
                n_val = layer_spec.params['n']['value'][e_idx] if 'n' in layer_spec.params and not layer_spec.params['n'].get('fit', False) else None
                k_val = layer_spec.params['k']['value'][e_idx] if 'k' in layer_spec.params and not layer_spec.params['k'].get('fit', False) else None

                if n_val is not None or k_val is not None:
                    records.append({
                        'energy': E,
                        'pol': pol,
                        'layer': lname,
                        'n': n_val if n_val is not None else '—',
                        'k': k_val if k_val is not None else '—',
                        'fit': False
                    })

        df = pd.DataFrame(records)

        if save_path:
            df.to_csv(save_path, index=False)

        return df

    def get_nk_wide_dataframe(self, nk_E, save_path=None):
        records = []
        for e_idx, E_pol in enumerate(self.energy_pol_uni):
            E = self.energy_index_map[E_pol]['energy']
            pol = self.energy_index_map[E_pol]['pol']
            energy_params = [p for p in self.domain_E_init if f'_{E_pol}' in p['name']]
            nk_vals = nk_E[e_idx]
            row = {'energy': E}
            for i, param in enumerate(energy_params):
                name = param['name']
                if 'n_' in name:
                    lname = name.split('n_')[1]#.rsplit('_', 1)[0]
                    row[f'n_{lname}'] = nk_vals[i]
                elif 'k_' in name:
                    lname = name.split('k_')[1]#.rsplit('_', 1)[0]
                    row[f'k_{lname}'] = nk_vals[i]
            records.append(row)
        df = pd.DataFrame(records)
        if save_path:
            df.to_csv(save_path, index=False)
        return df

    def get_global_param_dataframe(self, x_global, save_path=None):
        records = []
        global_index = 0  # index into x_global

        for key, spec in self.global_params.items():
            if spec.get('fit', False):
                value = x_global[global_index]
                global_index += 1
            else:
                value = spec['value']

            records.append({
                'param_name': key,
                'value': value,
                'fit': spec.get('fit', False)
            })

        for layer_spec in self.layers:
            lname = layer_spec.name

            # Thickness (only if not fitted)
            if 'thickness' in layer_spec.params:
                thick_spec = layer_spec.params['thickness']
                if not thick_spec.get('fit', False):  # Only include if fit is False
                    value = thick_spec.get('value', '—')
                    records.append({
                        'param_name': f'thickness_{lname}',
                        'value': value,
                        'fit': False,
                    })

            # Roughness (only if not fitted)
            if 'roughness' in layer_spec.params:
                rough_spec = layer_spec.params['roughness']
                if not rough_spec.get('fit', False):  # Only include if fit is False
                    value = rough_spec.get('value', '—')
                    records.append({
                        'param_name': f'roughness_{lname}',
                        'value': value,
                        'fit': False,
                    })



        df = pd.DataFrame(records)

        if save_path:
            df.to_csv(save_path, index=False)

        return df


    def save_all_fit_outputs(self, combined_df, R_E, nk_E, x_global, folder_path="fit_outputs", sample_name="sample"):
        """
        Saves reflectivity, nk (long and wide), and global parameters to CSV files.
        Filenames include sample_name and timestamp.
        """


        os.makedirs(folder_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def fname(label):
            return os.path.join(folder_path, f"{sample_name}_{label}_{timestamp}.csv")

        # Reflectivity
        df_reflectivity = self.get_reflectivity_dataframe(combined_df, R_E)
        df_reflectivity.to_csv(fname("reflectivity"), index=False)

        # nk long format
        df_nk_long = self.get_nk_long_dataframe(nk_E)
        df_nk_long.to_csv(fname("nk_long"), index=False)

        # nk wide format
        #df_nk_wide = self.get_nk_wide_dataframe(nk_E)
        #df_nk_wide.to_csv(fname("nk_wide"), index=False)

        # global parameters
        df_global = self.get_global_param_dataframe(x_global)
        df_global.to_csv(fname("global_params"), index=False)

        print(f"✅ Fit outputs saved to '{folder_path}' with sample '{sample_name}' and timestamp {timestamp}")
