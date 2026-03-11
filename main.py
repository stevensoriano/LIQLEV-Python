# -*- coding: utf-8 -*-
"""
main.py
Main execution script for the LIQLEV simulation.

This script acts as the master controller. It reads the configuration, 
iterates through the requested parameter sweeps (fill fractions, vent rates, 
and epsilon models), executes the core transient solver, saves the output 
data to CSVs, and generates the post-simulation diagnostic plots.
"""

import numpy as np
import pandas as pd
from config import get_config, get_base_inputs
from core import liqlev_simulation
from plotting import (plot_gravity_profiles, plot_vent_rate, plot_level_rise, 
                      plot_pressure, plot_level_increase_in, plot_diagnostics, 
                      plot_vapor_generation, plot_eps)

def calculate_epsilon_local(h, dtank):
    """
    Calculates the theoretical baseline epsilon (boil-off partitioning ratio) 
    based purely on tank geometry and liquid height.
    
    Epsilon (eps) = Wall Wetted Area / (Wall Wetted Area + Free Surface Area)
    
    This assumes vapor generation scales proportionally with the heat transfer area.
    """
    perim = np.pi * dtank
    a_wall = perim * h
    ac = 0.7854 * (dtank ** 2)
    a_interface = ac
    if (a_wall + a_interface) == 0: return 0
    return a_wall / (a_wall + a_interface)

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. LOAD CONFIGURATION & SWEEP PARAMETERS
    # ---------------------------------------------------------
    config = get_config()
    
    VENT_RATES = config['VENT_RATES']
    OUTPUT_FILE = config['OUTPUT_FILE']
    INITIAL_FILL_FRACTIONS = config['INITIAL_FILL_FRACTIONS']
    EPSILONS = config['EPSILONS']
    DURATION = config['DURATION']
    
    # ---------------------------------------------------------
    # 2. VERIFY & PLOT GRAVITY PROFILE
    # ---------------------------------------------------------
    # If transient gravity data was loaded, plot it before running the main loops
    # to ensure the drop tower / parabolic flight profile is correctly shaped.
    if 'TGGO_g' in config:
        last_original_time = config['LAST_ORIGINAL_GRAVITY_TIME']
        plot_gravity_profiles(config['TGGO_g'], config['XGGO_g'], config['XGGO'], last_original_time, DURATION)
        
    # ---------------------------------------------------------
    # 3. TANK GEOMETRY & REFERENCE CALCULATIONS
    # ---------------------------------------------------------
    dtank = config['DTANK']
    volt = config['VOLT']
    ac = 0.7854 * (dtank ** 2)
    htank = volt / ac
    
    # Print a reference table showing what the height-dependent epsilon 
    # will theoretically look like at standard fill fractions.
    ft_to_cm = 30.48
    print(f"\n=== Reference Epsilon Values for Tank Geometry ===")
    print(f"Tank Diameter: {dtank:.5f} ft ({dtank * ft_to_cm:.2f} cm)")
    print(f"Tank Height:   {htank:.5f} ft ({htank * ft_to_cm:.2f} cm)")
    print("-" * 45)
    print(f"{'Fill %':<10} | {'Height (ft)':<12} | {'Epsilon':<10}")
    print("-" * 45)
    
    ref_fills = [0.25, 0.50, 0.75, 1.00]
    for fill in ref_fills:
        h_current = fill * htank
        eps_val = calculate_epsilon_local(h_current, dtank)
        print(f"{fill*100:<10.0f} | {h_current:<12.4f} | {eps_val:<10.4f}")
    print("-" * 45 + "\n")
    
    # Track the peak level rise times and whether the tank overfilled for the final summary
    max_times_summary = []

    # ---------------------------------------------------------
    # 4. MAIN PARAMETRIC SWEEP (Nested Loops)
    # ---------------------------------------------------------
    for fill_fraction in INITIAL_FILL_FRACTIONS:
        print(f"\n=== Processing fill fraction: {fill_fraction} ===")
        
        for epsilon in EPSILONS:
            print(f"\n=== Processing epsilon: {epsilon} ===")
            
            # --- Epsilon Mode Logic ---
            if epsilon == 'varying':
                # Custom varying schedule (defined inside get_base_inputs if used)
                model_neps = None
                model_teps = None
                model_xeps = None
            elif epsilon == 'height_dep':
                # Calculates eps dynamically at every timestep based on current liquid height
                model_neps = 0
                model_teps = None
                model_xeps = None
            elif epsilon == 'bulk_fake':
                # "Bulk Fake" mode: Mimics distributed volumetric boiling in large tanks 
                # (like the JAXA 30m^3 tank) by multiplying the effective vapor-generating 
                # surface area by 50 to match experimental volume swelling.
                model_neps = 2
                model_teps = np.array([0.0, DURATION])
                model_xeps = np.array([50.0, 50.0])
            else:
                # Constant Epsilon mode: The user passed a specific number (e.g., 0.4, 0.8)
                model_neps = 2
                model_teps = np.array([0.0, DURATION])
                model_xeps = np.array([float(epsilon), float(epsilon)])
            
            # Store dataframes for this specific fill/epsilon combination so we can plot 
            # the different vent rates together on the same graphs.
            results_dfs = []
            
            for vent_rate in VENT_RATES:
                print("======================================================")
                print(f"### RUNNING SIMULATION FOR INITIAL VENT RATE: {vent_rate} lbm/s ###")
                print("======================================================")
                
                # Retrieve the physics inputs and run the transient solver
                model_inputs = get_base_inputs(vent_rate, fill_fraction, neps=model_neps, teps=model_teps, xeps=model_xeps)
                results_df = liqlev_simulation(model_inputs)
                results_dfs.append(results_df)
                
                # --- Post-Simulation Data Tracking ---
                exceeds = 'No'
                if not results_df.empty:
                    # Check if the liquid interface hit the tank lid
                    final_height = results_df['Height'].iloc[-1]
                    if (results_df['Height'] >= htank).any():
                        exceeds = 'Yes'

                    # Find the maximum liquid level rise and when it occurred
                    max_hratio_row = results_df.loc[results_df['Hratio'].idxmax()]
                    max_hratio_time = max_hratio_row['Time']
                    max_hratio = max_hratio_row['Hratio']
                    max_delta_h = max_hratio * model_inputs['Htzero']
                    
                    max_times_summary.append(
                        f"Time to max height for fill {fill_fraction}, vent {vent_rate}, "
                        f"epsilon {epsilon}: {max_hratio_time} s | Max Δh: {max_delta_h:.3f} ft | Exceeds tank: {exceeds}"
                    )

                # --- Export to CSV ---
                output_file = OUTPUT_FILE.replace('.csv', f'_{fill_fraction:.2f}_{vent_rate}_{epsilon}.csv')
                try:
                    print(f"\n[*] Writing results to {output_file}...")
                    results_df.to_csv(output_file, index=False)
                    print(f"[+] Successfully saved results.")
                except Exception as e:
                    print(f"[ERROR] Could not save the output file: {e}")
            
            # ---------------------------------------------------------
            # 5. GENERATE DIAGNOSTIC & RESULTS PLOTS
            # ---------------------------------------------------------
            # Generates comparative plots showing all vent rates for the current fill/epsilon combo
            plot_vent_rate(results_dfs, VENT_RATES)
            plot_level_rise(results_dfs, VENT_RATES, fill_fraction, epsilon, htank)
            plot_pressure(results_dfs, VENT_RATES, fill_fraction, epsilon, htank)
            plot_level_increase_in(results_dfs, VENT_RATES, fill_fraction, epsilon, htank)
            plot_diagnostics(results_dfs, VENT_RATES, fill_fraction, epsilon, htank) 
            plot_vapor_generation(results_dfs, VENT_RATES, fill_fraction, epsilon)
            
            if results_dfs:
                plot_eps(results_dfs, VENT_RATES, epsilon, htank)
            
    # ---------------------------------------------------------
    # 6. FINAL EXECUTION SUMMARY
    # ---------------------------------------------------------
    print(f"\nDerived tank height: {htank:.3f} ft")
    print("\n=== Summary of Times to Max Height ===")
    for summary in max_times_summary:
        print(summary)