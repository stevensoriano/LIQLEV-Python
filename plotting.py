# -*- coding: utf-8 -*-
"""
plotting.py
Matplotlib graphing functions for the LIQLEV simulation.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_gravity_profiles(time_s, grav_g, grav_ft_s2, last_original_data_time, duration):
    print("\n[*] Generating Gravity Profile plots...")
    plt.style.use('default')

    # Plot in g's
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    ax1.plot(time_s, grav_g, linestyle='-', linewidth=2, label='Gravity Profile') 
    ax1.set_title('Input Gravity Profile vs. Time', fontsize=16)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel("Gravity (g's)", fontsize=12)
    if duration > last_original_data_time:
        ax1.axvline(x=last_original_data_time, color='r', linestyle='--', label=f'End of data ({last_original_data_time:.2f}s)', alpha=0.4)
    ax1.legend(fontsize=11)
    plt.grid(True)
    plt.show()

    # Plot in ft/s^2
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)
    ax2.plot(time_s, grav_ft_s2, linestyle='-', linewidth=2, label='Gravity Profile', color='green')
    ax2.set_title('Input Gravity Profile vs. Time', fontsize=16)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Gravity (ft/s²)', fontsize=12)
    if duration > last_original_data_time:
        ax2.axvline(x=last_original_data_time, color='r', linestyle='--', label=f'End of data ({last_original_data_time:.2f}s)', alpha=0.4)
    ax2.legend(fontsize=11)
    plt.grid(True)
    plt.show()

def plot_vent_rate(dfs, vent_rates):
    print("\n[*] Generating Vent Rate Profile plot...")
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    num_rates = len(vent_rates)
    colors = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, num_rates)]
    
    for i, (df, rate) in enumerate(zip(dfs, vent_rates)):
        if not df.empty:
            ax.plot(df['Time'], df['Vent Rate'], linestyle='-', linewidth=2,
                    label=f'initial vent rate = {rate} lbm/s', color=colors[i])
            
    ax.set_title('Vent Rate Profile vs. Time', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Vent Rate (lbm/s)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.show()

def plot_level_rise(dfs, vent_rates, fill_fraction, epsilon, htank):
    print("\n[*] Generating Dimensionless Liquid Level Rise plot...")
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    num_rates = len(vent_rates)
    colors = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, num_rates)]
    
    red_x_plotted = False
    green_star_plotted = False
    
    for i, (df, rate) in enumerate(zip(dfs, vent_rates)):
        ax.plot(df['Time'], df['Hratio'], linestyle='-', linewidth=2,
                label=f'initial vent rate = {rate} lbm/s', color=colors[i])
        
        exceed_mask = df['Height'] >= htank
        if exceed_mask.any():
            idx = exceed_mask[exceed_mask].index[0]
            time_ex = df.loc[idx, 'Time']
            y_ex = df.loc[idx, 'Hratio']
            label = 'Tank Height Exceeded' if not red_x_plotted else ""
            ax.plot(time_ex, y_ex, 'rx', markersize=10, markeredgewidth=2, label=label, zorder=5)
            red_x_plotted = True
        else:
            idx_max = df['Hratio'].idxmax()
            time_max = df.loc[idx_max, 'Time']
            hratio_max = df.loc[idx_max, 'Hratio']
            label = 'Max Level Rise' if not green_star_plotted else ""
            ax.plot(time_max, hratio_max, marker='*', color='green', markersize=15, 
                    markeredgecolor='black', zorder=5, label=label)
            ax.axhline(y=hratio_max, color='green', linestyle='--', linewidth=1, alpha=0.7, label='_nolegend_')
            green_star_plotted = True
            
    ax.set_title(f'Dimensionless Liquid Level Increase vs. Time (Fill: {fill_fraction*100:.0f}%, Epsilon: {epsilon})', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Dimensionless Liquid Level Increase, Δh/h₀', fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    special_labels = ['Max Level Rise', 'Tank Height Exceeded']
    ordered_handles, ordered_labels, marker_handles, marker_labels = [], [], [], []

    for handle, label in zip(handles, labels):
        if label in special_labels:
            marker_handles.append(handle)
            marker_labels.append(label)
        else:
            ordered_handles.append(handle)
            ordered_labels.append(label)
            
    ax.legend(ordered_handles + marker_handles, ordered_labels + marker_labels, fontsize=11)
    plt.grid(True)
    plt.show()

def plot_pressure(dfs, vent_rates, fill_fraction, epsilon, htank):
    print("\n[*] Generating Tank Pressure vs. Time plot...")
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    num_rates = len(vent_rates)
    colors = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, num_rates)]

    red_x_plotted = False
    green_star_plotted = False

    for i, (df, rate) in enumerate(zip(dfs, vent_rates)):
        ax.plot(df['Time'], df['Press'], linestyle='-', linewidth=2,
                label=f'initial vent rate = {rate} lbm/s', color=colors[i])
        
        exceed_mask = df['Height'] >= htank
        if exceed_mask.any():
            idx = exceed_mask[exceed_mask].index[0]
            time_ex = df.loc[idx, 'Time']
            y_ex = df.loc[idx, 'Press']
            label = 'Tank Height Exceeded' if not red_x_plotted else ""
            ax.plot(time_ex, y_ex, 'rx', markersize=10, markeredgewidth=2, label=label, zorder=5)
            red_x_plotted = True
        else:
            idx_max = df['Hratio'].idxmax()
            time_max = df.loc[idx_max, 'Time']
            press_at_max_rise = df.loc[idx_max, 'Press']
            label = 'Max Level Rise' if not green_star_plotted else ""
            ax.plot(time_max, press_at_max_rise, marker='*', color='green', markersize=15, 
                    markeredgecolor='black', zorder=5, label=label)
            green_star_plotted = True
            
    ax.set_title(f'Tank Pressure vs. Time (Fill: {fill_fraction*100:.0f}%, Epsilon: {epsilon})', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Tank Pressure (psia)', fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    special_labels = ['Max Level Rise', 'Tank Height Exceeded']
    ordered_handles, ordered_labels, marker_handles, marker_labels = [], [], [], []

    for handle, label in zip(handles, labels):
        if label in special_labels:
            marker_handles.append(handle)
            marker_labels.append(label)
        else:
            ordered_handles.append(handle)
            ordered_labels.append(label)
            
    ax.legend(ordered_handles + marker_handles, ordered_labels + marker_labels, fontsize=11)
    plt.grid(True)
    plt.show()

def plot_level_increase_in(dfs, vent_rates, fill_fraction, epsilon, htank):
    print("\n[*] Generating Liquid Level Increase (inches) plot...")
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    num_rates = len(vent_rates)
    colors = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, num_rates)]

    red_x_plotted = False
    green_star_plotted = False

    for i, (df, rate) in enumerate(zip(dfs, vent_rates)):
        initial_height = df['Height'].iloc[0] if not df.empty else 0
        ax.plot(df['Time'], (df['Height'] - initial_height) * 12, linestyle='-', linewidth=2,
                label=f'initial vent rate = {rate} lbm/s', color=colors[i])
        
        exceed_mask = df['Height'] >= htank
        if exceed_mask.any():
            idx = exceed_mask[exceed_mask].index[0]
            time_ex = df.loc[idx, 'Time']
            y_ex = (df.loc[idx, 'Height'] - initial_height) * 12
            label = 'Tank Height Exceeded' if not red_x_plotted else ""
            ax.plot(time_ex, y_ex, 'rx', markersize=10, markeredgewidth=2, label=label, zorder=5)
            red_x_plotted = True
        else:
            idx_max = df['Hratio'].idxmax()
            time_max = df.loc[idx_max, 'Time']
            level_increase_max = (df.loc[idx_max, 'Height'] - initial_height) * 12
            label = 'Max Level Rise' if not green_star_plotted else ""
            ax.plot(time_max, level_increase_max, marker='*', color='green', markersize=15, 
                    markeredgecolor='black', zorder=5, label=label)
            ax.axhline(y=level_increase_max, color='green', linestyle='--', linewidth=1, alpha=0.7, label='_nolegend_')
            green_star_plotted = True
            
    ax.set_title(f'Liquid Level Increase vs. Time (Fill: {fill_fraction*100:.0f}%, Epsilon: {epsilon})', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Liquid Level Increase, Δh (inches)', fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    special_labels = ['Max Level Rise', 'Tank Height Exceeded']
    ordered_handles, ordered_labels, marker_handles, marker_labels = [], [], [], []

    for handle, label in zip(handles, labels):
        if label in special_labels:
            marker_handles.append(handle)
            marker_labels.append(label)
        else:
            ordered_handles.append(handle)
            ordered_labels.append(label)
            
    ax.legend(ordered_handles + marker_handles, ordered_labels + marker_labels, fontsize=11)
    plt.grid(True)
    plt.show()

def plot_eps(dfs, vent_rates, epsilon, htank):
    print("\n[*] Generating Epsilon vs. Time plot...")
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    num_rates = len(vent_rates)
    colors = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, num_rates)]

    for i, (df, rate) in enumerate(zip(dfs, vent_rates)):
        if not df.empty:
            ax.plot(df['Time'], df['eps'], linestyle='-', linewidth=2, 
                    label=f'initial vent rate = {rate} lbm/s', color=colors[i])
    
    ax.set_title(f'Epsilon vs. Time (Epsilon Case: {epsilon})', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Epsilon (ε)', fontsize=12)
    ax.legend(fontsize=11)
    plt.grid(True)
    plt.show()
    
def plot_diagnostics(dfs, vent_rates, fill_fraction, epsilon, htank):
    print("\n[*] Generating Diagnostic Plots (Superheat, dP/dt, and BL Volume)...")
    plt.style.use('default')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), dpi=300, sharex=True)
    
    num_rates = len(vent_rates)
    colors = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, num_rates)]
    
    # --- Plot 1: Superheat (Primary) and dP/dt (Secondary) ---
    ax1.set_title(f'Driver: Liquid Superheat & Depressurization (Fill: {fill_fraction*100:.0f}%, ε: {epsilon})', fontsize=14)
    ax1.set_ylabel('Liquid Superheat (K)', fontsize=12, color='black')
    
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel('dP/dt (psi/s)', fontsize=12, color='black')
    
    lines = []
    labels = []
    red_x_plotted_1 = False

    for i, (df, rate) in enumerate(zip(dfs, vent_rates)):
        if df.empty: continue
        
        l1, = ax1.plot(df['Time'], df['Superheat'], linestyle='-', linewidth=2,
                 label=f'Superheat (Vent: {rate})', color=colors[i])
        
        l2, = ax1_twin.plot(df['Time'], df['dP/dtha'], linestyle=':', linewidth=2,
                      label=f'dP/dt (Vent: {rate})', color=colors[i], alpha=0.9)
        
        lines.append(l1)
        lines.append(l2)
        labels.append(f'Superheat (Vent: {rate})')
        labels.append(f'dP/dt (Vent: {rate})')

        exceed_mask = df['Height'] >= htank
        if exceed_mask.any():
            idx = exceed_mask[exceed_mask].index[0]
            time_ex = df.loc[idx, 'Time']
            y_ex = df.loc[idx, 'Superheat']
            
            lx, = ax1.plot(time_ex, y_ex, 'rx', markersize=10, markeredgewidth=2, zorder=5)
            
            if not red_x_plotted_1:
                lines.append(lx)
                labels.append('Tank Height Exceeded')
                red_x_plotted_1 = True

    ax1.grid(True, alpha=0.5)
    ax1.legend(lines, labels, loc='upper left', fontsize=9, ncol=2)
    ax1.axvline(x=5.0, color='gray', linestyle='--', alpha=0.3)

    # --- Plot 2: BL Volume (Primary) vs Gravity (Secondary) ---
    ax2.set_title('Response: Vapor Volume Swelling vs Gravity', fontsize=14)
    ax2.set_ylabel('Boundary Layer Vol (ft³) [Log]', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_yscale('log')
    
    ax3 = ax2.twinx()
    ax3.set_ylabel("Gravity (g's)", fontsize=12, color='gray')
    
    if dfs and not dfs[0].empty:
        ax3.plot(dfs[0]['Time'], dfs[0]['Gravity_g'], color='gray', linestyle='--', 
                 linewidth=2, alpha=0.5, label='Gravity Profile')

    red_x_plotted_2 = False
    for i, (df, rate) in enumerate(zip(dfs, vent_rates)):
        if df.empty: continue
        ax2.plot(df['Time'], df['VBL vol'], linestyle='-', linewidth=2,
                 label=f'VBL (Vent: {rate})', color=colors[i])
        
        exceed_mask = df['Height'] >= htank
        if exceed_mask.any():
            idx = exceed_mask[exceed_mask].index[0]
            time_ex = df.loc[idx, 'Time']
            y_ex = df.loc[idx, 'VBL vol']
            
            label_x = 'Tank Height Exceeded' if not red_x_plotted_2 else ""
            ax2.plot(time_ex, y_ex, 'rx', markersize=10, markeredgewidth=2, label=label_x, zorder=5)
            red_x_plotted_2 = True

    ax2.grid(True, which="both", alpha=0.5)
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax3.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_vapor_generation(dfs, vent_rates, fill_fraction, epsilon):
    print("\n[*] Generating Vapor Generation Plots...")
    plt.style.use('default')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), dpi=300, sharex=True)
    num_rates = len(vent_rates)
    colors = [plt.get_cmap('viridis')(i) for i in np.linspace(0, 1, num_rates)]

    # --- Plot 1: Generation Rate (kg/s) ---
    for i, (df, rate) in enumerate(zip(dfs, vent_rates)):
        if df.empty: continue
        ax1.plot(df['Time'], df['Vap Gen Rate (kg/s)'], linestyle='-', linewidth=2,
                 label=f'Vent: {rate} lbm/s', color=colors[i])
    
    ax1.set_title(f'Vapor Generation Rate vs Time (Fill: {fill_fraction*100:.0f}%, ε: {epsilon})', fontsize=14)
    ax1.set_ylabel('Vapor Gen Rate (kg/s)', fontsize=12)
    ax1.grid(True, alpha=0.5)
    ax1.legend(fontsize=10, loc='upper right')

    # --- Plot 2: Cumulative Mass (kg) ---
    for i, (df, rate) in enumerate(zip(dfs, vent_rates)):
        if df.empty: continue
        ax2.plot(df['Time'], df['Total Vap Gen (kg)'], linestyle='-', linewidth=2,
                 label=f'Vent: {rate} lbm/s', color=colors[i])

    ax2.set_title('Cumulative Vapor Mass Generated', fontsize=14)
    ax2.set_ylabel('Total Vapor Mass (kg)', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.show()