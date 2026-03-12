# -*- coding: utf-8 -*-
"""
config.py
Configuration and inputs for the LIQLEV simulation.

This module acts as the central interface for the user. Adjust the variables 
in the `get_config()` dictionary to set up your specific cryogenic venting scenarios.
"""

import numpy as np
import pandas as pd
from thermo_utils import Tsat, Psat, DensitySat

def get_config():
    """
    Central configuration function for the LIQLEV simulation.
    Users should modify the values in this dictionary to set up their specific test cases.
    """
    config = {}
    
    # =========================================================================
    # 1. FILE PATHS
    # =========================================================================
    # Directory where the output CSV files will be saved.
    config['OUTPUT_FILE'] = r"./results/python_results.csv"
    # Path to the CSV containing transient gravity data (time vs. acceleration in g's).
    config['GRAVITY_FILE'] = r'./data/5s_drop_tower_extracted_az_positive_data.csv'

    # =========================================================================
    # 2. FLUID & THERMODYNAMIC SETTINGS
    # =========================================================================
    # Working fluid (e.g., "Nitrogen", "Hydrogen", "Oxygen", "Methane"). Uses CoolProp.
    config['FLUID'] = "Nitrogen" 
    
    # Initial tank pressure (psia). The bulk liquid is assumed saturated at this pressure.
    config['PINIT_PSIA'] = 14.7 
    # Final target tank pressure (psia). The simulation terminates when this is reached.
    config['PFINAL_PSIA'] = 5.0  

    # =========================================================================
    # 3. TANK GEOMETRY
    # =========================================================================
    # Tank internal diameter (feet).
    config['DTANK'] = 0.3 
    # Tank internal height (feet).
    config['HTANK'] = 2.0
    # Derived: Tank internal volume (ft^3).
    config['VOLT'] = (np.pi / 4) * (config['DTANK']**2) * config['HTANK']

    # =========================================================================
    # 4. SIMULATION CONTROL & SWEEP PARAMETERS
    # =========================================================================
    # Total simulation duration limit (seconds).
    config['DURATION'] = 20.0 
    # Time step for the numerical solver (seconds). Smaller = more accurate but slower.
    config['DELTA_T'] = 0.01 
    
    # List of initial vent mass flow rates to simulate (lbm/s). 
    # The script will run a separate simulation loop for each rate in this list.
    config['VENT_RATES'] = [0.003] 
    
    # List of initial liquid fill fractions by volume (0.0 to 1.0). (e.g., 0.5 = 50% full).
    config['INITIAL_FILL_FRACTIONS'] = [0.5] 
    
    # Epsilon (ε) defines the boil-off partitioning ratio. It represents the fraction 
    # of vapor generated in the wall thermal boundary layer vs. the liquid free surface.
    #   - 'height_dep': Dynamically calculates ε at each time step based on wetted wall area.
    #   - Numbers (e.g., 0.4, 50.0): Forces a constant ε. High values mimic bulk boiling.
    config['EPSILONS'] = ['height_dep']

    # --- Vent Valve Ramping Logic ---
    # Time (in seconds) over which the vent valve physically opens/ramps down to a holding value.
    config['VENT_RAMP_DURATION'] = 5.0
    # The fraction of the initial vent rate to hold after the ramp duration.
    # (e.g., 1.0 = constant venting, 0.8 = rate drops linearly by 20% over the ramp duration).
    config['VENT_TARGET_FACTOR'] = 1.0

    # =========================================================================
    # 5. GRAVITY SETTINGS
    # =========================================================================
    # TOGGLE: Set to True to use a constant gravity level, or False to read from GRAVITY_FILE
    config['USE_CONSTANT_GRAVITY'] = True
    config['CONSTANT_GRAVITY_G'] = 0.001  # Constant gravity level in g's (used if True above)
    
    # If using the CSV file and the simulation runs longer than the provided data, 
    # the model will "hold" the gravity at this constant value (in g's) for the rest of the run.
    config['HOLD_G_VALUE'] = 0.0014
    config['GRAVITY_FUNCTION'] = None 
    
    
    # =========================================================================
    # INTERNAL LOGIC: GRAVITY DATA PROCESSING (Do not change unless debugging)
    # =========================================================================
    g_to_ft_s2 = 32.174

    # Branch 1: Use Constant Gravity
    if config['USE_CONSTANT_GRAVITY']:
        print(f"[*] Using constant gravity of {config['CONSTANT_GRAVITY_G']} g's.")
        config['NGGO'] = 2
        config['TGGO'] = np.array([0.0, config['DURATION']])
        # Convert user's g's to ft/s^2 for the math solver
        const_g_ft_s2 = config['CONSTANT_GRAVITY_G'] * g_to_ft_s2
        config['XGGO'] = np.array([const_g_ft_s2, const_g_ft_s2])
        
        # Save plotting references
        config['TGGO_g'] = config['TGGO']
        config['XGGO_g'] = config['XGGO'] / g_to_ft_s2
        config['LAST_ORIGINAL_GRAVITY_TIME'] = config['DURATION']

    # Branch 2: Read Transient Gravity from CSV
    else:
        try:
            print(f"[*] Reading transient gravity data from: {config['GRAVITY_FILE']}")
            g_df = pd.read_csv(config['GRAVITY_FILE'])

            # Get original data from the file
            tggo_g = g_df['normalized_time'].to_numpy()
            xggo_g = g_df['az_positive'].to_numpy()
            
            # Store the end time of the *original* data for plotting reference
            last_original_data_time = tggo_g[-1]
            config['LAST_ORIGINAL_GRAVITY_TIME'] = last_original_data_time
            
            # Extend gravity profile if duration is longer than data
            if config['DURATION'] > last_original_data_time:
                print(f"[+] Gravity data ends at {last_original_data_time:.2f}s. Holding {config['HOLD_G_VALUE']} G's until {config['DURATION']}s.")
                # Add a point just after the last data point with the new hold value to create a step change
                tggo_g = np.append(tggo_g, last_original_data_time + 1e-9)
                xggo_g = np.append(xggo_g, config['HOLD_G_VALUE'])
                # Add a final point at the simulation duration to maintain the hold
                tggo_g = np.append(tggo_g, config['DURATION'])
                xggo_g = np.append(xggo_g, config['HOLD_G_VALUE'])
            
            config['TGGO_g'] = tggo_g
            config['XGGO_g'] = xggo_g
            config['NGGO'] = len(tggo_g)
            config['TGGO'] = tggo_g
            config['XGGO'] = xggo_g * g_to_ft_s2
            
            print("[+] Gravity data processed successfully.")

        except FileNotFoundError:
            print(f"[ERROR] Gravity data file not found at: {config['GRAVITY_FILE']}")
            print(f"[!] Reverting to constant gravity baseline of {config['CONSTANT_GRAVITY_G']} g's.")
            config['NGGO'] = 2
            config['TGGO'] = np.array([0.0, config['DURATION']])
            const_g_ft_s2 = config['CONSTANT_GRAVITY_G'] * g_to_ft_s2
            config['XGGO'] = np.array([const_g_ft_s2, const_g_ft_s2])
            config['TGGO_g'] = config['TGGO']
            config['XGGO_g'] = config['XGGO'] / g_to_ft_s2
            config['LAST_ORIGINAL_GRAVITY_TIME'] = config['DURATION']
            
    return config


def get_base_inputs(vent_rate_lbm_s, fill_fraction, neps=None, teps=None, xeps=None):
    """
    Translates the user-friendly config dictionary into the rigid data structures 
    expected by the core LIQLEV solver.
    """
    config = get_config()
    
    FLUID = config['FLUID']
    DURATION = config['DURATION']
    GRAVITY_FUNCTION = config['GRAVITY_FUNCTION']
    NGGO = config['NGGO']
    TGGO = config['TGGO']
    XGGO = config['XGGO']
    DELTA_T = config['DELTA_T']
    
    # Tank calculations
    dtank = config['DTANK']
    htank_ft = config['HTANK']
    volt = (np.pi / 4) * (dtank**2) * htank_ft
    ac = 0.7854 * (dtank ** 2)
    
    INITIAL_FILL_FRACTION = fill_fraction
    Htzero = INITIAL_FILL_FRACTION * htank_ft
    
    # Thermodynamic state setup
    PSI_TO_KPA = 6.89475729
    pinit_psia = config['PINIT_PSIA']
    pfinal_psia = config['PFINAL_PSIA']
    press_kpa = pinit_psia * PSI_TO_KPA
    tinit = Tsat(FLUID, press_kpa) * 1.8 # Convert K to Rankine for solver
    
    # Compute initial saturated liquid density (lb/ft³) to get consistent initial liquid mass
    if FLUID == "Hydrogen":
        rhol = 0.1709 + 0.7454*tinit - 0.04421*tinit**2 + 0.001248*tinit**3 - 1.738e-5*tinit**4 + 9.424e-8*tinit**5
    else:
        t_k = tinit / 1.8
        ps_kpa = Psat(FLUID, t_k)
        rhol = DensitySat(FLUID, "liquid", ps_kpa) * 0.0624279606
        
    xmlzro = rhol * (Htzero * ac)

    # Setup Epsilon schedule if not dynamically calculated
    if neps is None:
        neps = 11 
        teps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, DURATION]) 
        xeps = np.array([0.0000, 0.0513, 0.1780, 0.2800, 0.3620, 0.4220, 0.4700, 0.5200, 0.5600, 0.6000, 0.6000])

    # Variable Vent Rate Logic
    ramp_duration = config['VENT_RAMP_DURATION']
    ramp_target_factor = config['VENT_TARGET_FACTOR']

    if DURATION > ramp_duration:
        # Case 1: Duration is long enough to complete the ramp and hold
        tvmdot_arr = np.array([0.0, ramp_duration, DURATION])
        xvmdot_arr = np.array([vent_rate_lbm_s, vent_rate_lbm_s * ramp_target_factor, vent_rate_lbm_s * ramp_target_factor])
    else:
        # Case 2: Duration is shorter than the ramp time (calculate linear intercept)
        slope = (vent_rate_lbm_s * ramp_target_factor - vent_rate_lbm_s) / ramp_duration
        end_rate = vent_rate_lbm_s + (slope * DURATION)
        tvmdot_arr = np.array([0.0, DURATION])
        xvmdot_arr = np.array([vent_rate_lbm_s, end_rate])
        
    nvmd_count = len(tvmdot_arr)

    # Return the dictionary that `core.py` unpacks to run the simulation
    inputs = {
        # --- Scalar Inputs ---
        "Title": "Liquid Level Rise EPS From EVOLVE",
        "Liquid": FLUID,
        "Units": "British",
        "Delta": DELTA_T,
        "Dtank": dtank,
        "Htzero": Htzero,
        "Volt": volt,
        "Xmlzro": xmlzro,
        "Pinit": pinit_psia,
        "Pfinal": pfinal_psia,
        "Tinit": tinit,
        "Thetin": 0.0,
        "Nvmd": nvmd_count,
        "Neps": neps,
        "Nlattm": 2,
        "Nvertm": 2,
        "Nggo": NGGO,
        # --- Array Inputs ---
        "Tvmdot": tvmdot_arr, 
        "Xvmdot": xvmdot_arr, 
        "Teps": teps,
        "Xeps": xeps,
        "Tspal": np.array([0.0, DURATION]),
        "Xspacl": np.array([1.0, 1.0]),
        "Tspav": np.array([0.0, DURATION]),
        "Xspacv": np.array([1.0, 1.0]),
        "Tggo": TGGO,
        "Xggo": XGGO,
        "gravity_function": GRAVITY_FUNCTION,
    }
    return inputs