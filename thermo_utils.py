# -*- coding: utf-8 -*-
"""
thermo_utils.py
Helper and thermodynamic functions for the LIQLEV simulation.
"""

import numpy as np
from CoolProp.CoolProp import PropsSI

def dPdTsat(fluid, press_kpa):
    """Evaluates slope of the saturated P-T curve (kPa/K)."""
    t_plus = Tsat(fluid, press_kpa + 0.1)
    t_minus = Tsat(fluid, press_kpa - 0.1)
    if t_plus == t_minus: return 0
    return 0.2 / (t_plus - t_minus)

def Cpsat(fluid, phase, press_kpa):
    """Evaluates saturation specific heat at constant pressure (kJ/kg-K)."""
    quality = 1 if phase == "vapor" else 0
    return PropsSI("C", "P", press_kpa * 1000, "Q", quality, fluid) / 1000

def DensitySat(fluid, phase, press_kpa):
    """Evaluates saturation density (kg/m^3)."""
    quality = 1 if phase == "vapor" else 0
    return PropsSI("Dmass", "P", press_kpa * 1000, "Q", quality, fluid)

def EnthalpySat(fluid, phase, press_kpa):
    """Evaluates saturation enthalpy (kJ/kg)."""
    quality = 1 if phase == "vapor" else 0
    return PropsSI("H", "P", press_kpa * 1000, "Q", quality, fluid) / 1000

def LHoV(fluid, press_kpa):
    """Evaluates latent heat of vaporization (kJ/kg)."""
    h_vap = EnthalpySat(fluid, "vapor", press_kpa)
    h_liq = EnthalpySat(fluid, "liquid", press_kpa)
    return h_vap - h_liq

def Psat(fluid, temp_sat_k):
    """Evaluates saturation pressure (kPa)."""
    return PropsSI("P", "T", temp_sat_k, "Q", 1, fluid) / 1000

def Tsat(fluid, press_kpa):
    """Evaluates saturation temperature (K)."""
    return PropsSI("T", "P", press_kpa * 1000, "Q", 1, fluid)

def sli(arg, x, y):
    """Simple Linear Interpolation using numpy."""
    return np.interp(arg, x, y)