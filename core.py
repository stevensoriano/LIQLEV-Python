# -*- coding: utf-8 -*-
"""
core.py
Core execution loop for the LIQLEV boundary layer simulation.

This module calculates the transient liquid level rise and two-phase fluid 
behavior during zero-gravity venting. It uses a one-dimensional piston-like 
displacement assumption where vapor generated along the wetted walls 
(boundary layer) and bulk liquid flashing causes volumetric swelling.
"""

import numpy as np
import pandas as pd
from thermo_utils import sli, Tsat, Psat, DensitySat, Cpsat, LHoV, dPdTsat

def liqlev_simulation(inputs):
    """Main simulation logic converted from the LIQLEV VBA subroutine."""
    
    # ---------------------------------------------------------
    # 1. UNPACK INPUTS & INITIALIZE GEOMETRY
    # ---------------------------------------------------------
    inputs = {k.lower(): v for k, v in inputs.items()}
    title, delta, thetin, units = inputs['title'], inputs['delta'], inputs['thetin'], inputs['units']
    nvmd, tvmdot, xvmdot = int(inputs['nvmd']), inputs['tvmdot'], inputs['xvmdot']
    neps, teps, xeps = int(inputs['neps']), inputs['teps'], inputs['xeps']
    nlattm, tspal, xspacl = int(inputs['nlattm']), inputs['tspal'], inputs['xspacl']
    nvertm, tspav, xspacv = int(inputs['nvertm']), inputs['tspav'], inputs['xspacv']
    nggo, tggo, xggo = int(inputs['nggo']), inputs['tggo'], inputs['xggo']
    gravity_function = inputs.get('gravity_function', None)
    fluid = inputs['liquid']
    
    # Convert SI inputs to British units (the core solver math relies on British units)
    if units == "SI":
        dtank, htzero, volt, xmlzro, pinit, pfinal, tinit = inputs['dtank']*3.28, inputs['htzero']*3.28, inputs['volt']*(3.28**3), inputs['xmlzro']*2.20462, inputs['pinit']*0.145, inputs['pfinal']*0.145, inputs['tinit']*1.8
    else:
        dtank, htzero, volt, xmlzro, pinit, pfinal, tinit = inputs['dtank'], inputs['htzero'], inputs['volt'], inputs['xmlzro'], inputs['pinit'], inputs['pfinal'], inputs['tinit']

    # Tank geometric properties
    perim = np.pi * dtank                  # Tank perimeter (ft)
    ac = 0.7854 * (dtank ** 2)             # Cross-sectional area (ft^2)
    htank, fill = volt/ac, htzero*ac/volt  # Total tank height and initial fill fraction
    print(f"Initial % fill volume: {fill * 100:.2f}%")
    
    # Initialize state variables
    dhdt, vbl1, hldak3 = 0.0, 0.0, 0.0     # dh/dt (ft/s), V_BL (ft^3), hold for solver constant
    p1, t1, h1, theta1, xml1, zht1 = pinit, tinit, htzero, thetin, xmlzro, htzero
    results, xmvap1, xmvap2 = [], 0.0, 0.0
    is_first_iteration = True
    cumulative_vap_kg = 0.0                # Total vapor generated tracker
    last_print_time = -1.0
    PSI_TO_KPA = 6.89475729
    
    # ---------------------------------------------------------
    # 2. MAIN TRANSIENT SOLVER LOOP
    # ---------------------------------------------------------
    # The simulation runs until the tank depressurizes to the final target pressure
    while p1 > pfinal:
        # Print progress every 0.5 seconds of simulation time
        if theta1 - last_print_time >= 0.5:
            print(f"   ...Simulating Time: {theta1:.2f}s | Pressure: {p1:.2f} | Height: {h1:.4f}")
            last_print_time = theta1

        # --- Evaluate Thermodynamic Saturation Properties ---
        if fluid == "Hydrogen":
            # Legacy empirical polynomials for Hydrogen (used to exact-match AS-203 flight data)
            rhol = 0.1709+0.7454*t1-0.04421*t1**2+0.001248*t1**3-1.738e-5*t1**4+9.424e-8*t1**5  # Liquid density (lbm/ft^3)
            rhov = -0.2511+0.04294*t1-0.00286*t1**2+9.159e-5*t1**3-1.422e-6*t1**4+1.001e-8*t1**5 # Vapor density (lbm/ft^3)
            cs, hfg = 0.078*(t1-34.0)+2.12, -2.0*(t1-34.0)+194.5                                # Specific heat (cs) and Latent Heat (hfg)
            dpdts = 2.49-0.22*t1+0.00407*t1**2+5.22e-5*t1**3                                    # Slope of saturation curve (dP/dT)
        else:
            # CoolProp high-accuracy property evaluation for Nitrogen, Oxygen, Methane, etc.
            t1_k, ps_kpa = t1/1.8, Psat(fluid, t1/1.8)
            rhol, rhov = DensitySat(fluid, "liquid", ps_kpa)*0.0624279606, DensitySat(fluid, "vapor", ps_kpa)*0.0624279606
            cs, hfg = Cpsat(fluid, "liquid", ps_kpa)*0.2388458966, LHoV(fluid, ps_kpa)*0.4299226
            dpdts = dPdTsat(fluid, ps_kpa)*0.08057652094
            
        # Calculate instantaneous volumes and masses
        volliq = xml1/rhol + vbl1    # Total liquid volume = pure liquid volume + boundary layer vapor volume
        volgas = volt - volliq       # Ullage volume = Total volume - total liquid volume
        xmvap3 = volgas * rhov       # Ullage vapor mass
        if is_first_iteration: xmvap1 = xmvap3
        
        # --- Time & Vent Rate Interpolation ---
        dtdps = 1.0/dpdts if dpdts!=0 else 0  # Inverse slope (dT/dP)
        theta2 = theta1 + delta               # Advance time step
        thetav = 0.5 * (theta1 + theta2)      # Mid-step time for interpolation
        vmdot = sli(thetav, tvmdot, xvmdot)   # Interpolate current vent mass flow rate
        
        # --- Thermodynamic Depressurization Physics ---
        # dpdtha: Rate of pressure decay (dP/dt). Derived from an energy balance 
        # combining the vent extraction rate and the liquid's flashing response.
        denom = (xml1*cs*dtdps/hfg + xmvap1*(1/p1-dtdps/t1)) if hfg*p1*t1!=0 else 1
        dpdtha = -vmdot/denom if denom!=0 else 0
        delp = dpdtha * delta                 # Pressure drop for this time step
        p2 = p1 + delp                        # New pressure
        t2 = t1 + dtdps * delp                # New saturation temperature based on dT/dP
        
        # delme: Mass of bulk liquid evaporated (flashed) due to the temperature drop
        delme = xml1*cs*(t2-t1)/hfg if hfg!=0 else 0
        xml2 = xml1 + delme                   # Update liquid mass (delme is negative)
        delmv = vmdot*delta                   # Total mass vented out of the tank
        
        # --- Boundary Layer & Bubble Physics ---
        # eps: Partitioning ratio. Determines what fraction of the total boil-off 
        # happens along the wetted wall (creating the boundary layer) vs. the free surface.
        eps = sli(thetav, teps, xeps) if neps>0 else (perim*h1)/(perim*h1+ac)
        
        # Bubble spacing and transient gravity
        spacv, spacl = sli(thetav, tspav, xspacv), sli(thetav, tspal, xspacl)
        ggo_ft_s2 = gravity_function(thetav) if gravity_function is not None else sli(thetav, tggo, xggo)
        
        # Boundary layer differential equation coefficients
        # AK1: Related to buoyancy forces pulling bubbles upward
        ak1_term = (10.8*(1+spacl)*(1+spacv)*ggo_ft_s2*(rhol-rhov)/rhol) if rhol!=0 else 0
        ak1 = 1.089*(ak1_term**0.5) if ak1_term>0 else 0
        # AK2: Related to thermal vapor generation at the wall
        ak2 = -eps*cs*rhol*dtdps*dpdtha/rhov/hfg if rhov*hfg!=0 else 0
        # AK3: Ratio of thermal vapor generation to buoyant removal (driving force of swelling)
        ak3 = ak2/ak1 if ak1!=0 else 0
        
        # ---------------------------------------------------------
        # 3. BOUNDARY LAYER VOLUME ITERATIVE SOLVER (Secant Method)
        # ---------------------------------------------------------
        # This nested loop iteratively solves for the Boundary Layer Thickness (delblz) 
        # and Boundary Layer Volume (vbl2) to ensure mass and volume conservation.
        nconv = 0
        ak4, fvbl4 = 0, 0
        solver_loop_active = True
        
        while solver_loop_active and nconv < 80:
            zht2 = zht1 + dhdt * delta
            if ak3 < 0: ak3 = hldak3
            
            # Initial guess for boundary layer thickness at the top of the interface
            delblz = (0.375*dtank*ak3*zht2)**(2/3) if (0.375*dtank*ak3*zht2)>0 else 0
            
            # Inner Newton-Raphson loop to refine boundary layer thickness (delblz)
            n_inner = 0
            while n_inner < 20:
                sum1 = sum((4**(l-1))*(delblz**(l+0.5))/(dtank**l)/(2**l+1) for l in range(1,11))
                fdelt = 8.0*sum1/ak3 - zht2 if ak3!=0 else float('inf')
                if abs(fdelt) <= 1e-5*zht2: break
                summ = sum((4**(k-1))*(delblz**(k-0.5))/(dtank**k) for k in range(1,11))
                fpdelt = 4.0*summ/ak3 if ak3!=0 else float('inf')
                delblz -= fdelt/(fpdelt if fpdelt!=0 else 1e-9)
                n_inner += 1
                
            # Calculate total vapor volume contained in the boundary layer (vbl2)
            sum1_vbl = sum((2*l+1)*(delblz**(l+1.5))/(l+1.5)/(dtank**(l-1)) for l in range(1,11))
            vbl2 = sum1_vbl*np.pi/ak3 if ak3!=0 else 0
            
            # Error function (fvbl) comparing calculated volume against continuity
            fvbl = vbl2-ak2*xml1*delta/rhol+2.1*ak1*dtank*(delblz**1.5)*delta-vbl1 if rhol!=0 else float('inf')
            
            # Secant convergence check
            if abs(fvbl) <= 0.001 * vbl2:
                solver_loop_active = False # Converged
            else:
                # Update ak3 guess using Secant method
                if nconv <= 1:
                    if nconv == 0: nconv = 1
                    savak3, svfvbl = ak3, fvbl
                    if fvbl > 0: ak3 *= 0.1
                    else: ak3 *= 2.0; nconv = 0
                else:
                    if (fvbl > 0 and svfvbl < 0) or (fvbl < 0 and svfvbl > 0):
                        pass
                    else:
                        savak3, svfvbl = ak4, fvbl4
                    ak4, fvbl4 = ak3, fvbl
                    fvbl_diff = fvbl - svfvbl
                    secant_ak3 = ak3 - fvbl * (ak3 - savak3) / fvbl_diff if fvbl_diff != 0 else ak3
                    ak3 = 0.5 * (secant_ak3 + savak3)
                    nconv += 1
        
        # ---------------------------------------------------------
        # 4. POST-SOLVER KINEMATICS & DATA TRACKING
        # ---------------------------------------------------------
        # dhdt: Piston-like displacement velocity of the liquid-vapor interface
        dhdt = ((vbl2-vbl1)+(xml2-xml1)/rhol)/ac/delta if rhol*ac*delta!=0 else 0
        
        # Update absolute height and dimensionless level increase (Hratio)
        zht2, h2, delh = zht1+dhdt*delta, zht1+dhdt*delta, (zht1+dhdt*delta)-h1
        hratio = (h2-htzero)/htzero if htzero!=0 else 0
        
        # beta: Fraction of evaporated mass remaining entrained in the boundary layer
        beta_denom = (1-np.exp(-cs*(tinit-t2)/hfg)) if hfg!=0 else 0
        beta = vbl2*rhov/xmlzro/beta_denom if xmlzro*beta_denom!=0 else 0
        
        xmvbl2 = vbl2*rhov  # Mass of vapor currently in the boundary layer
        xmdtbl = 2.1*ak1*dtank*(delblz**1.5)*delta*rhov # Vapor mass leaving the boundary layer into the ullage
        
        # Vapor Generation tracking (Boundary layer output + Bulk Flashing)
        # Note: delme is negative when liquid evaporates, so -(1-eps)*delme is a positive mass addition
        mass_gen_step_lbm = xmdtbl - (1 - eps) * delme
        LBM_TO_KG = 0.453592
        vap_gen_rate_kg_s = (mass_gen_step_lbm / delta) * LBM_TO_KG
        
        if not is_first_iteration:
            cumulative_vap_kg += mass_gen_step_lbm * LBM_TO_KG

        # Update ullage mass conservation
        xmvap2 = xmvap1-(1-eps)*delme-delmv+xmdtbl
        
        # Superheat tracking: Difference between current liquid temp and saturation temp at current pressure
        if fluid == "Hydrogen":
            t_sat_k_curr = Tsat(fluid, p2 * PSI_TO_KPA) 
            t_sat_r_curr = t_sat_k_curr * 1.8
        else:
            t_sat_k_curr = Tsat(fluid, p2 * PSI_TO_KPA)
            t_sat_r_curr = t_sat_k_curr * 1.8
        
        superheat_rankine = t2 - t_sat_r_curr
        superheat_kelvin = superheat_rankine / 1.8 
        
        # Append all time step data to the results list
        results.append({
            'Time':theta2, 'Press':p2, 'Temp':t2, 'Liq Mass':xml2, 'Ullage Mass':xmvap2, 
            'Height':h2, 'dh/dt':dhdt, 'delh':delh, 'Hratio':hratio, 'dP/dtha':dpdtha, 
            'delp':delp, 'eps':eps, 'beta':beta, 'VBL vol':vbl2, 'BL thick':delblz, 
            'AK3':ak3, 'Vent Rate':vmdot, 'AK1':ak1, 'AK2':ak2, 'AK2/AK1':ak2/ak1 if ak1!=0 else 0, 
            'Vapor in BL':xmvbl2, 'BL Vap Out':xmdtbl, 'Ullage from Calc':xmvap3, 
            'Conv Iterations':nconv,
            'Gravity_g': ggo_ft_s2 / 32.174,   
            'Superheat': superheat_kelvin,     
            'Vap Gen Rate (kg/s)': vap_gen_rate_kg_s,
            'Total Vap Gen (kg)': cumulative_vap_kg
        })
        
        # Setup variables for the next time step
        p1, t1, h1, theta1, xml1, xmvap1, vbl1, zht1 = p2, t2, h2, theta2, xml2, xmvap2, vbl2, zht2
        hldak3, is_first_iteration = ak3, False
        
        # Terminate early if the vent schedule is complete
        if theta2 >= tvmdot[-1]-delta: break
        
    return pd.DataFrame(results)