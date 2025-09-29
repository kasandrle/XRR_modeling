# XRR_modeling  
**Accessible multilayer reflectivity fitting across energies**

Welcome to **XRR_modeling**, a Python toolkit for modeling and fitting reflectivity data (e.g., XRR, RSoXS) across multiple photon energies. This package provides a modular framework for defining multilayer stacks, managing global and energy-dependent parameters, and exporting results with ceremonial clarity.

---

## Purpose

XRR_modeling offers:

- A flexible objective function for inverse modeling of reflectivity data  
- Support for both global and per-energy fitting strategies  
- Customizable error modeling via `sigma_mode`  
- Export pipelines with timestamped, sample-named outputs for traceable documentation

>  **Optical Constants Convention**  
> In this fitting framework, the complex refractive index is defined as:  
> `n = 1 - δ + i·β`  
> where `δ` (delta) and `β` (beta) are the dispersive and absorptive components, respectively.  

---

## Core Components

### `reflectivity_model`

Contains two primary classes:

#### `LayerSpec`
Defines individual layer properties such as thickness, roughness, and optical constants.

#### `ReflectivityModel`
Constructs and manages a multilayer reflectivity model for energy-dependent fitting.

This class organizes layer specifications, global parameters, and energy-resolved domains to support inverse modeling. It supports both global and per-energy fitting strategies and allows flexible error modeling via `sigma_mode`.

---

## Class Overview: `ReflectivityModel`

### Attributes

- `energy_points`: List of discrete energy values (e.g., in eV)  
- `layers`: Ordered list of `LayerSpec` objects, including substrate  
- `global_params`: Dictionary of global fit parameters (e.g., thickness, roughness, offsets)  
- `fit_strategy`: `"global"` or `"per_energy"` — controls how energy-dependent parameters are fit  
- `sigma_mode`: `"model"` or `"column"` — defines how error variance (σ²) is computed  
- `sigma_column`: Column name in `combined_df` if using `"column"` mode  
- `sigma_function`: User-defined function of reflectivity and model for custom error modeling  
- `keys`: Centralized parameter dictionary used during fitting and simulation  
- `domain_E_init`: Initial energy-dependent parameter definitions (name, x₀, bounds)  
- `x0_dict`: Maps energy values to initial parameter guesses  
- `energy_index_map`: Maps energy values to their index for array slicing

---

### Methods

- `_validate_layers()` — Ensures layer definitions are complete and consistent  
- `_inject_layer_params()` — Transfers layer-level parameters into the global dictionary  
- `_build_energy_domains()` — Constructs energy-dependent domains for `n` and `k`  
- `_build_global_keys()` — Initializes `keys` from global and energy-dependent parameters  
- `_validate_sigma_model()` — Checks presence of required parameters in `global_params`  
- `initialize_keys_from_x0()` — Populates `keys` from initial values  
- `build_layer_arrays(energy_pol_uni)` — Constructs stack arrays for transfer matrix method  
- `describe_sigma()` — Returns a human-readable description of the sigma error model  
- `summarize_stack(energy_pol_uni)` — Prints layer values for a given energy  
- `get_global_fit_params()` — Returns x₀, bounds, and keys for global parameters  
- `get_reflectivity_dataframe(exp_reflectivity, simulated_reflectivity)` — Saves experimental and simulated reflectivity to a DataFrame  
- `get_nk_long_dataframe(nk_E)` — Saves optical constants in long-format DataFrame  
- `get_global_param_dataframe(x_global)` — Saves global fit parameters to DataFrame  
- `save_all_fit_outputs(...)` — Saves reflectivity, optical constants, and global parameters to CSV files with sample name and timestamp  
- `from_config(config)` — Initializes model from a JSON config  
- `to_config(fit_para=None)` — Saves model to JSON; updates initial values if `fit_para` is provided

---

## Class Overview: `LayerSpec`

Represents a single layer in a multilayer reflectivity model.

Each `LayerSpec` defines the physical and optical properties of a layer, including:

- **Thickness** — fixed or fit  
- **Interface roughness** — fixed or fit  
- **Energy-dependent refractive index components (`n` and `k`)** — from material data or user-defined arrays

The class supports both fixed values and parameter fitting, with bounds and initial guesses. Substrate layers are treated specially and cannot have thickness parameters.

---

### Attributes

- `name` (`str`) — Identifier for the layer  
- `is_substrate` (`bool`) — Flag indicating whether the layer is a substrate  
- `params` (`dict`) — Stores parameter specifications for thickness, roughness, `n`, and `k`  
- `_nk_set` (`bool`) — Internal flag to ensure `n` and `k` arrays are defined before validation

---

### Methods

- `fit_thickness(x0, bounds)` — Enable thickness fitting with initial guess and bounds  
- `fixed_thickness(value)` — Set a fixed thickness value  
- `fit_roughness(x0, bounds)` — Enable roughness fitting  
- `fixed_roughness(value)` — Set a fixed roughness value  
- `fit_nk_from_material(material, energy_uni, bounds_n, bounds_k, density)` — Fit `n`/`k` from material database  
- `fixed_nk_from_material(material, energy_uni, density)` — Fix `n`/`k` from material database  
- `fit_nk_array(n_array, k_array, bounds_n, bounds_k)` — Fit `n`/`k` from user-defined arrays  
- `fixed_nk(n_array, k_array)` — Fix `n`/`k` from user-defined arrays  
- `validate(energy_count)` — Ensure `n`/`k` arrays are properly defined and match energy resolution  
- `describe()` — Print out a description of the layer  
- `from_row(row, energy_pol_uni)` — Configure layer from a row in an input CSV file (see README for format)

---

## Input csv file

See [`fit_input/sample1_et_up.csv`](fit_input/sample1_et_up.csv) for an example.

---

### Required Columns

The input CSV must contain the following columns:
name,thickness,fit_thickness,roughness,fit_roughness,deltabeta,density,fit_delta,fit_beta

Each row represents a single layer in the stack. **Layer order matters**:

- The **top layer** is closest to the surface.
- The **last layer** is the substrate, and its `thickness` must be left empty.

---

### Parameter Guidelines

- **`thickness` and `roughness`**  
  - If the corresponding `fit_*` field is empty → treated as fixed values.  
  - If fitting is enabled → used as starting parameters.

- **Valid entries for `fit_thickness`, `fit_roughness`, `fit_delta`, or `fit_beta`**  
  - A single float (± around the starting value)  
  - A pair of values in brackets `[lower_bound, upper_bound]`  
  - Leave empty to fix the parameter during fitting

---

### Optical Constants (`deltabeta` Field)

You can specify optical constants in one of two ways:

#### Option 1: External CSV File

Provide a CSV file containing optical constants for the layer with columns:
Energy, delta, beta

**Requirements:**
- The energy range must fully cover the range of the fitted data.
- Leave `density` empty.

#### Option 2: Henke Database Reference

Provide:
- A chemical formula (e.g., `SiO2`) in the `deltabeta` field  
- A numeric `density` value  

This will use the Henke database to retrieve optical constants.

---


## Installation

