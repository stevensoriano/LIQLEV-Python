# LIQLEV-Python: Cryogenic Liquid Level Rise Simulation

A modernized Python framework of the legacy LIQLEV boundary-layer model. This tool predicts transient liquid level rise (LLR) and two-phase fluid behavior during zero-gravity venting of cryogenic propellants.

## Overview

The management of cryogenic propellants under microgravity conditions presents significant challenges, particularly the safe venting of vapor without entraining liquid. When a cryogenic tank is vented to relieve internal pressure, the rapid pressure reduction initiates thermodynamic flashing and bulk boiling. Vapor generated within the superheated liquid bulk and along the wetted tank walls forms bubbles that displace the surrounding liquid, causing the liquid-vapor interface to rise. If this mixture reaches the vent port, liquid entrainment can occur, resulting in propellant loss.

This project is a complete Python reimplementation of the original Fortran LIQLEV model developed following the Saturn S-IVB AS-203 flight experiment[^1]. It builds upon subsequent adaptations for the Human Landing System (HLS) program[^2], upgrading the sequential Excel VBA environment into a robust, parallelized Python framework.

## Technical Details & Mathematical Model

The LIQLEV model explicitly accounts for wall-driven boil-off and its contribution to interface rise by treating the sidewall as a surface on which a growing vapor film forms. Bubbles nucleate, grow, and detach within this film, displacing bulk liquid. 

### Piston-Like Displacement
The framework retains the original one-dimensional piston-like displacement assumption, where all vapor generated is treated as uniformly distributed across the tank cross-section. The instantaneous interface location is tracked via the discrete relation:

$$Z_{\rm int}(t) = Z_{\rm int}(t-\Delta t) + \frac{\Delta V_{\rm BL} + \Delta m_{\rm liq}/\rho_{\rm liq}}{A_c}\Delta t$$

where $\Delta V_{\rm BL}$ is the incremental vapor volume generated in the wall thermal boundary layer, $\Delta m_{\rm liq}$ is the mass of liquid flashed to vapor, $\rho_{\rm liq}$ is the saturated liquid density, and $A_c$ is the tank cross-sectional area.

### Boil-Off Partitioning Ratio ($\epsilon$)
The parameter $\epsilon$ represents the instantaneous fraction of total vapor mass generated within the wall thermal boundary layer relative to the vapor generated at the liquid-vapor free surface. In the height-dependent mode, it is computed from the geometric wetted-area ratio:

$$\epsilon(t) = \frac{\pi D h(t)}{\pi D h(t) + \frac{\pi D^{2}}{4}}$$

*Note for Large-Scale Tanks:* For large-volume tanks at high fill fractions, distributed bulk flashing dominates. The model includes a "bulk_fake" mode (setting $\epsilon \approx 50$) which scales the effective vapor-generating surface area to emulate distributed volumetric boiling.

### Thermodynamic Integration
Outdated localized property lookups and curve fits have been replaced with the open-source `CoolProp` library[^3]. This provides consistent, high-accuracy real-fluid properties for Nitrogen, Hydrogen, Oxygen, and Methane across any pressure regime. The critical saturation-curve slope $(dP/dT)_{\rm sat}$ is computed dynamically via a central finite-difference scheme.

## Project Structure

```text
LIQLEV-Python-Simulation/
├── data/                 # Directory for input gravity profiles (CSV)
├── results/              # Output directory for simulation results
├── config.py             # User interface: Simulation settings and inputs
├── core.py               # The main mathematical solver loop
├── plotting.py           # Matplotlib visualization functions
├── thermo_utils.py       # CoolProp and thermodynamic helper functions
├── main.py               # The execution script
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/yourusername/LIQLEV-Python-Simulation.git](https://github.com/yourusername/LIQLEV-Python-Simulation.git)
cd LIQLEV-Python-Simulation
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Configure the Simulation:** Open `config.py` to adjust the simulation parameters. This file acts as the central interface where you can set the `'FLUID'` (e.g., Nitrogen, Hydrogen), `'VENT_RATES'`, `'INITIAL_FILL_FRACTIONS'`, and tank dimensions.
2. **Manage Gravity Profiles:** The simulation can run using either a constant gravity level or a transient gravity profile (like a drop tower or parabolic flight) provided via a CSV file. Open `config.py` and locate the **GRAVITY SETTINGS** section to toggle between these modes:
    * **To use a constant gravity level:** Set `config['USE_CONSTANT_GRAVITY'] = True`. You can then define the exact g-level by changing `config['CONSTANT_GRAVITY_G']` (e.g., `0.001`).
    * **To use a transient gravity CSV:** Set `config['USE_CONSTANT_GRAVITY'] = False`. Place your custom CSV file inside the `data/` folder and ensure the `config['GRAVITY_FILE']` path points to it. The CSV must contain columns named `normalized_time` (seconds) and `ax_positive` (g's). If the simulation duration exceeds the length of the CSV data, the model will automatically hold the final gravity value constant for the remainder of the run.
3. **Run the Model:**
```bash
python main.py
```
4. **Outputs:** The script generates CSV files in the `results/` directory containing time-series data for pressure, height, superheat, boundary layer volume, and vapor generation rates. It will also automatically generate `matplotlib` diagnostic plots correlating liquid superheat, depressurization, and boundary layer swelling.

---

## References

[^1]: Bradshaw, R. D., "Evaluation and Application of Data from Low-Gravity Orbital Experiment, Phase I Final Report", General Dynamics, Convair Division, NASA CR-109847 (Report No. GDC-DDB-70-003), 1970.
[^2]: Moran, Matt, "Boundary Layer Model: Adaptation for HLS", Internal NASA Document, n.d.
[^3]: Bell, Ian H. et al., "Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and the Open-Source Thermophysical Property Library CoolProp", Industrial & Engineering Chemistry Research, vol. 53, no. 6, pp. 2498-2508, 2014.