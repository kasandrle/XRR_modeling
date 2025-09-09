"""Calculate x-ray properties (mainly from periodictable and henke data)."""

import numpy as np

import periodictable as pt
import periodictable.xsf


def delta_beta(compound, energy, density=None, relative_density=1.0, desperate_density_lookup=False):
    n = refractive_index(compound, energy, density, relative_density, desperate_density_lookup)
    delta = 1 - np.real(n)
    beta = -np.imag(n)
    return delta, beta


def refractive_index(compound, energy, density=None, relative_density=1.0, desperate_density_lookup=False):
    """Returns the refractive n = 1 - delta - i beta for the given compound at the given energies. energy has to
    be given in keV or in any pint quantity convertible to keV in spectroscopic convention (so for example wavelength
    in nm is fine). The density should be given in g/cm^3 and will be taken from tables if it is None (see
    compound_density for desperate_density_lookup and further information).
    If the relative_density is given, delta and beta will be scaled by the relative density.
    """
    try:
        energy_keV = energy.to('keV', 'sp').magnitude
    except AttributeError:
        energy_keV = energy
    if density is None:
        density = compound_density(compound, desperate_lookup=desperate_density_lookup)
    if density is None:
        raise ValueError('Please specify density or desperate_density_lookup=True')
    try:
        density_g_cm3 = density.to('g/cm**3').magnitude
    except AttributeError:
        density_g_cm3 = density

    n = pt.xsf.index_of_refraction(compound, energy=energy_keV, density=density_g_cm3)
    delta = 1 - np.real(n)
    beta = -np.imag(n)
    n = 1 - (1j * beta + delta) * relative_density
    return n


def compound_density(compound, desperate_lookup=False):
    """Returns the density of the compound in g/cm^3. Elemental densities are taken from periodictable, which gets
    the densities from "The ILL Neutron Data Booklet, Second Edition."
    For compound densities, the values from the henke database at http://henke.lbl.gov/cgi-bin/density.pl are used
    if available.
    If the compound density is not found for the given compound, None is returned, unless desperate_lookup is True,
    in which case the elemental density of the first element in the compound is returned.
    """
    for d in henke_densities:
        if compound in (d[0], d[1]):
            return d[2]
    comp = pt.formula(compound)
    if comp.density is not None:
        return comp.density
    if desperate_lookup:
        return comp.structure[0][1].density
    return None

henke_densities = [
    ['', 'AgBr', 6.473],
    ['', 'AlAs', 3.81],
    ['', 'AlN', 3.26],
    ['Sapphire', 'Al2O3', 3.97],
    ['', 'AlP', 2.42],
    ['', 'B4C', 2.52],
    ['', 'BeO', 3.01],
    ['', 'BN', 2.25],
    ['Polyimide', 'C22H10N2O5', 1.43],
    ['Polypropylene', 'C3H6', 0.90],
    ['PMMA', 'C5H8O2', 1.19],
    ['Polycarbonate', 'C16H14O3', 1.2],
    ['Kimfol', 'C16H14O3', 1.2],
    ['Mylar', 'C10H8O4', 1.4],
    ['Teflon', 'C2F4', 2.2],
    ['Parylene-C', 'C8H7Cl', 1.29],
    ['Parylene-N', 'C8H8', 1.11],
    ['Fluorite', 'CaF2', 3.18],
    ['', 'CdWO4', 7.9],
    ['', 'CdS', 4.826],
    ['', 'CoSi2', 5.3],
    ['', 'Cr2O3', 5.21],
    ['', 'CsI', 4.51],
    ['', 'CuI', 5.63],
    ['', 'InN', 6.88],
    ['', 'In2O3', 7.179],
    ['', 'InSb', 5.775],
    ['', 'IrO2', 11.66],
    ['', 'GaAs', 5.316],
    ['', 'GaN', 6.10],
    ['', 'GaP', 4.13],
    ['', 'HfO2', 9.68],
    ['', 'LiF', 2.635],
    ['', 'LiH', 0.783],
    ['', 'LiOH', 1.43],
    ['', 'MgF2', 3.18],
    ['', 'MgO', 3.58],
    ['', 'Mg2Si', 1.94],
    ['Mica', 'KAl3Si3O12H2', 2.83],
    ['', 'MnO', 5.44],
    ['', 'MnO2', 5.03],
    ['', 'MoO2', 6.47],
    ['', 'MoO3', 4.69],
    ['', 'MoSi2', 6.31],
    ['Salt', 'NaCl', 2.165],
    ['', 'NbSi2', 5.37],
    ['', 'NbN', 8.47],
    ['', 'NiO', 6.67],
    ['', 'Ni2Si', 7.2],
    ['', 'Ru2Si3', 6.96],
    ['', 'RuO2', 6.97],
    ['', 'SiC', 3.217],
    ['', 'Si3N4', 3.44],
    ['Silica', 'SiO2', 2.2],
    ['Quartz', 'SiO2', 2.65],
    ['', 'TaN', 16.3],
    ['', 'TiN', 5.22],
    ['', 'Ta2Si', 14.],
    ['Rutile', 'TiO2', 4.26],
    ['ULE', 'Si.925Ti.075O2', 2.205],
    ['', 'UO2', 10.96],
    ['', 'VN', 6.13],
    ['Water', 'H2O', 1.0],
    ['', 'WC', 15.63],
    ['YAG', 'Y3Al5O12', 4.55],
    ['Zerodur', 'Si.56Al.5P.16Li.04Ti.02Zr.02Zn.03O2.46', 2.53],
    ['', 'ZnO', 5.675],
    ['', 'ZnS', 4.079],
    ['', 'ZrN', 7.09],
    ['Zirconia', 'ZrO2', 5.68],
    ['', 'ZrSi2', 4.88],
]
