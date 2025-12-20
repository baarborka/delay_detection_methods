import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas as pd
from pathlib import Path
import time

# ==========================================
# PARAMETER LOADING LOGIC
# ==========================================

def load_parameters(filename="params3.txt"):
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
        
    file_path = script_dir / filename
    print(f"--- Attempting to load parameters from: '{file_path}' ---")
    
    params = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            print(f"--- Successfully found and loading '{file_path}' ---")
            for line in file:
                if not line.strip() or line.strip().startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                try:
                    params[key] = ast.literal_eval(value)
                except Exception:
                    params[key] = value
    except FileNotFoundError:
        print(f"--- WARNING: Parameter file '{file_path}' not found. Using hardcoded defaults. ---")
    return params

loaded_params = load_parameters()

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================

MC_ITERATIONS = loaded_params.get('mc_iterations', 500) 
MAX_LAG = loaded_params.get('max_lag', 4)

# Physics
AREA = loaded_params.get('area', 100.0)
T_START = loaded_params.get('t_start', 0.0)
T_END = loaded_params.get('t_end', 2.0)
N_TIME_POINTS = loaded_params.get('n_time_points', 1000)
ENERGY_BANDS = loaded_params.get('energy_bands', [(50, 100), (100, 300), (300, 1000), (10000, 100000)])

# Spectrum
ALPHA = loaded_params.get('alpha', -1.0)
BETA = loaded_params.get('beta', -2.5)
E_PEAK = loaded_params.get('E_peak', 300.0)
E0 = loaded_params.get('E0', 100.0)
F0 = loaded_params.get('F0', 1.0)
K_PL = loaded_params.get('K_pl', 1e-8)
GAMMA = loaded_params.get('gamma', 1.7)
E_PIVOT = loaded_params.get('E_pivot', 100000.0)
E_CUT = loaded_params.get('E_cut', 1000000.0)

# Time
RISE_INDEX = loaded_params.get('rise_index', 1.0)
DECAY_INDEX = loaded_params.get('decay_index', 1.0)
T_PEAK_MIN = loaded_params.get('t_peak_min', 0.2)
T_PEAK_MAX = loaded_params.get('t_peak_max', 0.3)

# Background
bg_cfg = loaded_params.get('background', {'type': 'powerlaw', 'norm': 100, 'index': -1.0})
BG_NORM = bg_cfg.get('norm', 100)
BG_INDEX = bg_cfg.get('index', -1.0)

# --- CRITICAL SWITCHES ---
INCLUDE_BACKGROUND = loaded_params.get('include_background', True)
GENERATE_NOISELESS = loaded_params.get('generate_noiseless', False)
INCLUDE_POWER_LAW = loaded_params.get('include_power_law', False)

# NEW: Simulate CSV rounding (Standard CSV is ~8-10 digits)
SIMULATE_CSV_PRECISION = True 

print(f"\n=== CONFIGURATION ===")
print(f"Iterations: {MC_ITERATIONS}")
print(f"Noiseless Mode: {GENERATE_NOISELESS}")
print(f"Background: {INCLUDE_BACKGROUND}")
print(f"Power Law: {INCLUDE_POWER_LAW}")
print(f"Simulate CSV Rounding: {SIMULATE_CSV_PRECISION}")
print(f"=====================\n")

# ==========================================
# OUTPUT SETUP
# ==========================================
# Create a folder for results
timestamp = int(time.time())
OUTPUT_DIR = Path(f"mc_results_{timestamp}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"--- Results will be saved to: {OUTPUT_DIR.resolve()} ---")

# ==========================================
# PHYSICS KERNEL
# ==========================================

def band_function(E, E_peak, alpha, beta, E0, F0):
    E = np.asarray(E)
    E_break = (alpha - beta) * E_peak
    if E_break <= 0: E_break = E_peak
    cond = E <= E_break
    spec = np.zeros_like(E)
    spec[cond] = F0 * (E[cond]/E0)**alpha * np.exp(-E[cond]/E_peak)
    spec[~cond] = F0 * (E_break/E0)**(alpha-beta) * np.exp(beta-alpha) * (E[~cond]/E0)**beta
    return spec

def power_law_cutoff(E, K_pl, E_pivot, gamma, E_cut):
    E = np.asarray(E)
    E_safe = np.where(E == 0, 1e-10, E)
    return K_pl * (E_safe/E_pivot)**(-gamma) * np.exp(-E_safe/E_cut)

def integrate_flux(E_min, E_max, E_grid, spec_total):
    mask = (E_grid >= E_min) & (E_grid <= E_max)
    if not np.any(mask): return 0.0
    return np.trapz(spec_total[mask], E_grid[mask])

def light_curve_profile(t, t_peak, rise, decay):
    t = np.asarray(t)
    lc = np.zeros_like(t)
    mask_rise = t < t_peak
    denom = t_peak if t_peak > 0 else 1.0
    lc[mask_rise] = (t[mask_rise]/denom)**rise
    mask_decay = ~mask_rise
    td = t[mask_decay]
    td_safe = np.where(td==0, 1e-10, td)
    t_peak_safe = t_peak if t_peak > 0 else 1.0
    lc[mask_decay] = (t_peak/td_safe)**decay * np.exp(-(td - t_peak)/t_peak_safe)
    return lc

# ==========================================
# CORE SIMULATION & ANALYSIS ENGINE
# ==========================================

def run_monte_carlo_step(rng, time_grid, E_grid, spec_total, avg_energies, t_peaks, dt):
    n_bands = len(ENERGY_BANDS)
    raw_counts = np.zeros((n_bands, len(time_grid)))
    
    # --- 1. SIMULATION PHASE ---
    for i, (E_min, E_max) in enumerate(ENERGY_BANDS):
        flux = integrate_flux(E_min, E_max, E_grid, spec_total) * AREA
        profile = light_curve_profile(time_grid, t_peaks[i], RISE_INDEX, DECAY_INDEX)
        
        if INCLUDE_BACKGROUND:
            bg = BG_NORM * (avg_energies[i] ** BG_INDEX)
        else:
            bg = 0.0
        
        rate = (flux * profile) + bg
        expected_counts = rate * dt
        
        if GENERATE_NOISELESS:
            raw_counts[i] = expected_counts
        else:
            raw_counts[i] = rng.poisson(expected_counts)

    # --- 2. ANALYSIS PHASE ---
    
    # Sort bands by energy (Logic from Code 2)
    sort_indices = np.argsort(avg_energies)
    sorted_counts = raw_counts[sort_indices]
    sorted_energies = avg_energies[sort_indices]

    # Simulate CSV Truncation (Crucial for Noiseless matching)
    if SIMULATE_CSV_PRECISION:
        sorted_counts = np.round(sorted_counts, 10)

    # Reference Channel (Lowest Energy)
    ref_counts = sorted_counts[0]
    mean_ref = np.mean(ref_counts)
    
    iteration_autocorr = np.zeros((n_bands - 1, MAX_LAG))
    
    # Loop over higher energy channels
    for i in range(1, n_bands):
        target_counts = sorted_counts[i]
        mean_target = np.mean(target_counts)
        
        if mean_ref > 0:
            predicted = ref_counts * (mean_target / mean_ref)
        else:
            predicted = np.zeros_like(ref_counts)
            
        diff = target_counts - predicted
        
        if mean_target > 0:
            norm_diff = diff / mean_target
        else:
            norm_diff = np.zeros_like(diff)
            
        deviations = np.sign(norm_diff)
        
        for lag in range(1, MAX_LAG + 1):
            c = deviations[:-lag] * deviations[lag:]
            iteration_autocorr[i-1, lag-1] = np.mean(c)
            
    return raw_counts, iteration_autocorr, sorted_energies, sorted_counts

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================

def main():
    print(f"--- Starting Monte Carlo Analysis ({MC_ITERATIONS} iterations) ---")
    
    t_grid = np.linspace(T_START, T_END, N_TIME_POINTS)
    dt = (T_END - T_START) / (N_TIME_POINTS - 1)
    
    # Grid setup (Matches Code 1)
    min_energy_sys = min(b[0] for b in ENERGY_BANDS)
    max_E = max(band[1] for band in ENERGY_BANDS)
    E_grid_full = np.logspace(np.log10(min(1.0, min_energy_sys)), np.log10(max(max_E, E_CUT*1.5)), 5000)
    
    spec_band = band_function(E_grid_full, E_PEAK, ALPHA, BETA, E0, F0)
    spec_pl = power_law_cutoff(E_grid_full, K_PL, E_PIVOT, GAMMA, E_CUT)
    
    if INCLUDE_POWER_LAW:
        spec_total = spec_band + spec_pl
    else:
        spec_total = spec_band
    
    avg_energies = np.array([(b[0]+b[1])/2 for b in ENERGY_BANDS])
    log_avg = np.log10(avg_energies)
    
    if len(avg_energies) > 1 and (log_avg.max() != log_avg.min()):
        norm = (log_avg - log_avg.min()) / (log_avg.max() - log_avg.min())
    else:
        norm = np.zeros_like(log_avg)
        
    t_peaks = T_PEAK_MIN + norm * (T_PEAK_MAX - T_PEAK_MIN)
    
    mc_results = np.zeros((MC_ITERATIONS, len(ENERGY_BANDS)-1, MAX_LAG))
    example_lightcurves = [] 
    rng = np.random.default_rng()
    
    final_sorted_counts = None
    final_sorted_energies = None

    for i in range(MC_ITERATIONS):
        if i % 50 == 0: print(f"Iteration {i}/{MC_ITERATIONS}...") 
        counts, autocorr_results, sorted_energies, sorted_counts = run_monte_carlo_step(
            rng, t_grid, E_grid_full, spec_total, avg_energies, t_peaks, dt
        )
        mc_results[i] = autocorr_results
        final_sorted_energies = sorted_energies
        final_sorted_counts = sorted_counts
        if i < 3: example_lightcurves.append(counts)

    mean_autocorr = np.mean(mc_results, axis=0) 
    std_autocorr = np.std(mc_results, axis=0)
    
    # --- SAVE STATISTICS TO CSV ---
    stats_data = []
    energies_for_plot = final_sorted_energies[1:] 
    
    for i, energy in enumerate(energies_for_plot):
        row = {'Energy_keV': energy}
        for lag in range(MAX_LAG):
            row[f'Lag_{lag+1}_Mean'] = mean_autocorr[i, lag]
            row[f'Lag_{lag+1}_Std'] = std_autocorr[i, lag]
        stats_data.append(row)
    
    df_stats = pd.DataFrame(stats_data)
    stats_file = OUTPUT_DIR / "mc_statistics.csv"
    df_stats.to_csv(stats_file, index=False)
    print(f"\nSaved statistics to: {stats_file}")

    print(f"\nAnalysis Complete. Reference Channel: {final_sorted_energies[0]} keV")

    # PLOTTING
    # PLOT 1: Light Curves
    fig, axes = plt.subplots(len(ENERGY_BANDS), 1, figsize=(10, 10), sharex=True)
    colors = ['red', 'green', 'blue']
    if len(ENERGY_BANDS) == 1: axes = [axes]
    
    for band_idx, (E_min, E_max) in enumerate(ENERGY_BANDS):
        ax = axes[band_idx]
        runs_to_show = 1 if GENERATE_NOISELESS else min(3, len(example_lightcurves))
        for run_idx in range(runs_to_show):
            ax.step(t_grid, example_lightcurves[run_idx][band_idx], 
                   where='mid', alpha=0.8 if GENERATE_NOISELESS else 0.5, 
                   color=colors[run_idx % 3], lw=1.5,
                   label=f'Run {run_idx+1}' if band_idx==0 else "")
        ax.set_ylabel(f'Counts\n{E_min}-{E_max} keV')
        ax.grid(True, alpha=0.3)
    if not GENERATE_NOISELESS: axes[0].legend(loc='upper right')
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f"Example Light Curves ({'Noiseless' if GENERATE_NOISELESS else 'Noisy'})", fontsize=14)
    plt.tight_layout()
    
    # Save Plot 1
    plot1_path = OUTPUT_DIR / "example_lightcurves.png"
    plt.savefig(plot1_path)
    print(f"Saved Light Curves plot to: {plot1_path}")
    plt.show()

    # PLOT 2: Error Correlation
    if len(ENERGY_BANDS) > 1:
        plt.figure(figsize=(12, 7))
        
        energies_for_plot = final_sorted_energies[1:] 
        offsets = np.linspace(-0.02, 0.02, MAX_LAG)
        
        for lag in range(MAX_LAG):
            y_means = mean_autocorr[:, lag]
            y_errs = std_autocorr[:, lag]
            x_vals = energies_for_plot * (1 + offsets[lag])
            plt.errorbar(x_vals, y_means, yerr=y_errs, fmt='-o', capsize=5, lw=2, label=f'Lag {lag+1}')
            
        plt.xscale('log')
        plt.axhline(0, color='black', linestyle='--', lw=1)
        plt.xlabel('Energy Band Center (keV)', fontsize=12)
        plt.ylabel('Autocorrelation of Deviations', fontsize=12)
        title_str = f'Monte Carlo Analysis ({MC_ITERATIONS} Runs)\nMode: {"NOISELESS" if GENERATE_NOISELESS else "NOISY"}'
        plt.title(title_str, fontsize=14)
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        
        # Save Plot 2
        plot2_path = OUTPUT_DIR / "mc_deviation_autocorr.png"
        plt.savefig(plot2_path)
        print(f"Saved Correlation plot to: {plot2_path}")
        plt.show()
    else:
        print("\nNot enough bands to plot correlation analysis.")

if __name__ == "__main__":
    main()