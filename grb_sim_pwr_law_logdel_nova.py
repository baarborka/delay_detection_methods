import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
from pathlib import Path
import time
import shutil

# ---------------------------
# Helper Functions
# ---------------------------

def format_energy_label(energy):
    """Formats energy values in keV, MeV, or GeV for plot labels."""
    if energy >= 1e6:
        return f'{energy / 1e6:.2f} GeV'
    elif energy >= 1e3:
        return f'{energy / 1e3:.2f} MeV'
    else:
        return f'{energy:.2f} keV'

def load_parameters(filename="params2.txt"):
    """Loads simulation parameters from a text file."""
    
    # --- NEW: Make path relative to the script's location ---
    # This makes the script find 'params2.txt' in its *own* folder
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        # Fallback if __file__ is not defined (e.g., in an interactive prompt)
        print("--- NOTE: __file__ is not defined. Falling back to Current Working Directory. ---")
        script_dir = Path.cwd()
        
    file_path = script_dir / filename
    print(f"--- Attempting to load parameters from: '{file_path}' ---")
    # --- END NEW ---
    
    params = {}
    try:
        # --- ZMENA: Pridané encoding='utf-8' pre kompatibilitu s Windows ---
        with open(file_path, 'r', encoding='utf-8') as file: # <-- Use file_path
            print(f"--- Successfully found and loading '{file_path}' ---")
            for line in file:
                if not line.strip() or line.strip().startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                try:
                    params[key] = ast.literal_eval(value)
                    print(f"         Loaded: {key} = {params[key]}") # <-- ADDED VERBOSE PRINT
                except Exception:
                    params[key] = value
                    print(f"         Loaded: {key} = {params[key]}") # <-- ADDED VERBOSE PRINT
    except FileNotFoundError:
        print(f"--- WARNING: Parameter file '{file_path}' not found. ---")
        print(f"--- Current working directory is: '{Path.cwd()}' ---")
        print(f"--- Using hard-coded default values. ---")
    return params

def calculate_background_rate(avg_energy, norm, index):
    """Calculates background rate using a power-law model: norm * E^index."""
    return norm * (avg_energy ** index)

def light_curve(t, t_peak, rise_index, decay_index):
    """Defines the temporal shape of the GRB pulse."""
    t = np.asarray(t)
    lc = np.zeros_like(t, dtype=float)
    rise = t < t_peak
    if np.any(rise):
        denom = t_peak if t_peak > 0 else 1.0
        lc[rise] = (t[rise] / denom)**rise_index
    decay = ~rise
    if np.any(decay):
        td = t[decay]
        # Avoid division by zero if t_peak or td is 0
        td_safe = np.where(td == 0, np.finfo(float).eps, td)
        t_peak_safe = t_peak if t_peak > 0 else 1.0
        lc[decay] = (t_peak / td_safe)**decay_index * np.exp(-(td - t_peak) / t_peak_safe)
    return lc

# --- Spectral Model Functions ---

def band_function(E, E_peak, alpha, beta, E0, F0):
    """Models the GRB energy spectrum (Band function)."""
    E = np.asarray(E, dtype=float)
    E_break = (alpha - beta) * E_peak
    
    # Ensure E_break is positive, otherwise the model is unphysical
    if E_break <= 0:
        E_break = E_peak # Fallback, though parameters are likely problematic
        
    spec = np.where(
        E <= E_break,
        F0 * (E / E0)**alpha * np.exp(-E / E_peak),
        F0 * (E_break / E0)**(alpha - beta) * np.exp(beta - alpha) * (E / E0)**beta
    )
    return spec

def power_law_cutoff(E, K_pl, E_pivot, gamma, E_cut):
    """
    Models the high-energy power-law component with an exponential cutoff.
    N(E) = K * (E / E_pivot)^(-gamma) * exp(-E / E_cut)
    """
    E = np.asarray(E, dtype=float)
    # Avoid division by zero if E is 0
    E_safe = np.where(E == 0, np.finfo(float).eps, E)
    return K_pl * (E_safe / E_pivot)**(-gamma) * np.exp(-E_safe / E_cut)

def total_spectrum_model(E, E_peak, alpha, beta, E0, F0, K_pl, E_pivot, gamma, E_cut, area):
    """
    Calculates the total spectral model by summing the Band function
    and the high-energy power-law cutoff component, scaled by area.
    """
    spec_band = band_function(E, E_peak, alpha, beta, E0, F0)
    spec_pl = power_law_cutoff(E, K_pl, E_pivot, gamma, E_cut)
    
    total_spec = spec_band + spec_pl
    return total_spec * area

def band_integrated_flux(E_min, E_max, E_grid, E_peak, alpha, beta, E0, F0, K_pl, E_pivot, gamma, E_cut, area):
    """Integrates the *total* (Band + PL) spectrum over a specific energy band."""
    mask = (E_grid >= E_min) & (E_grid <= E_max)
    if not np.any(mask):
        return 0.0
    
    # Get the total spectrum value at the grid points within the band
    spec_total = total_spectrum_model(E_grid[mask], E_peak, alpha, beta, E0, F0, K_pl, E_pivot, gamma, E_cut, area)
    
    # Integrate using trapezoidal rule
    return np.trapz(spec_total, E_grid[mask])

# --- Simulation Functions ---

def cumulative_intensity(lambda_t, t_grid):
    """Calculates the cumulative integral of a rate function."""
    t = np.asarray(t_grid)
    lam = np.asarray(lambda_t)
    Lambda = np.zeros_like(t, dtype=float)
    if t.size > 1:
        dt = np.diff(t)
        Lambda[1:] = np.cumsum(0.5 * (lam[1:] + lam[:-1]) * dt)
    return Lambda

def simulate_toas_from_rate(t_grid, lambda_t, rng=None):
    """Simulates photon arrival times from a rate function using inversion sampling."""
    if rng is None:
        rng = np.random.default_rng()
    
    Lambda = cumulative_intensity(lambda_t, t_grid)
    if not Lambda.size or Lambda[-1] <= 0:
        return np.array([])
    
    Lambda_end = Lambda[-1]
    
    events_L = []
    S = 0.0
    while True:
        S += rng.exponential(1.0) # Draw from Exp(1)
        if S > Lambda_end:
            break
        events_L.append(S)
        
    if not events_L:
        return np.array([])
    
    events_L = np.array(events_L)
    t_events = np.interp(events_L, Lambda, t_grid)
    return t_events

# ---------------------------
# Main Script Logic
# ---------------------------

# --- Step 0: Setup Folders ---
base_name = 'grb_simulation_with_background'
output_base = Path(base_name)
output_base.mkdir(parents=True, exist_ok=True)
existing_folders = [int(f.name.split('_')[-1]) for f in output_base.glob(f'{base_name}_*') if f.name.split('_')[-1].isdigit()]
next_number = max(existing_folders, default=0) + 1
output_folder = output_base / f'{base_name}_{next_number}'
output_folder.mkdir(parents=True, exist_ok=True)
common_graphs_folder = output_base / "graphs"
common_graphs_folder.mkdir(parents=True, exist_ok=True)
run_graphs_folder = output_folder

# --- Step 1: Load Parameters ---
params = load_parameters('params3.txt')
print("--- Loaded Parameters (from file) ---")
if params:
    for key, value in params.items():
        print(f"  {key}: {value}")
else:
    print("  (No parameters loaded from file)")
print("----------------------------------\n")

# --- NEW: Dictionary to store all parameters used in this run ---
simulation_params = {}
# --- END NEW ---

# Assign params to variables, using .get() for safety
area = params.get('area', 100) # cm^2
simulation_params['area'] = area

# Band function parameters
alpha = params.get('alpha', -1.0)
simulation_params['alpha'] = alpha
beta = params.get('beta', -2.5)
simulation_params['beta'] = beta
E0 = params.get('E0', 100.0)      # keV
simulation_params['E0'] = E0
F0 = params.get('F0', 1.0)       # photons / (cm^2 * s * keV) at E0
simulation_params['F0'] = F0
E_peak = params.get('E_peak', 300.0) # keV
simulation_params['E_peak'] = E_peak

# --- FIXED DEFAULTS ---
# NEW: Power-law cutoff parameters
# Default K_pl is now much smaller (1e-8)
default_K_pl = 1e-8
K_pl = params.get('K_pl', default_K_pl)       # photons / (cm^2 * s * keV) at E_pivot
simulation_params['K_pl'] = K_pl
E_pivot = params.get('E_pivot', 100000.0) # 100 MeV, in keV
simulation_params['E_pivot'] = E_pivot
# Default gamma is now positive 1.7
default_gamma = 1.7
gamma = params.get('gamma', default_gamma)      # Photon index (model uses -gamma)
simulation_params['gamma'] = gamma
E_cut = params.get('E_cut', 1000000.0)   # 1 GeV, in keV
simulation_params['E_cut'] = E_cut

# --- ADDED: Explicit check for parameter loading ---
if 'K_pl' not in params:
    print(f"\n--- WARNING: 'K_pl' not found in params file. Using default: {default_K_pl}")
if 'gamma' not in params:
    print(f"--- WARNING: 'gamma' not found in params file. Using default: {default_gamma}")
# --- END ADDED ---

# --- END FIXED DEFAULTS ---

# Time parameters
t_start = params.get('t_start', 0.0)
simulation_params['t_start'] = t_start
t_end = params.get('t_end', 2.0)
simulation_params['t_end'] = t_end
n_time_points = params.get('n_time_points', 1000)
simulation_params['n_time_points'] = n_time_points

# Energy bands
energy_bands = params.get('energy_bands', [(50, 100), (100, 300), (300, 1000), (10000, 100000)]) # keV
simulation_params['energy_bands'] = energy_bands

# Light curve shape parameters
t_peak_min = params.get('t_peak_min', 0.2)
simulation_params['t_peak_min'] = t_peak_min
t_peak_max = params.get('t_peak_max', 0.3)
simulation_params['t_peak_max'] = t_peak_max
rise_index = params.get('rise_index', 1.0)
simulation_params['rise_index'] = rise_index
decay_index = params.get('decay_index', 1.0)
simulation_params['decay_index'] = decay_index

# Background parameters
default_background = {'type': 'powerlaw', 'norm': 0, 'index': -1.0}
bg_cfg = params.get('background', default_background)
simulation_params['background'] = bg_cfg
# Also store the derived bg params for clarity
bg_type = bg_cfg.get('type', 'powerlaw')
bg_norm = bg_cfg.get('norm', 100)
bg_index = bg_cfg.get('index', -1.0)
simulation_params['bg_type'] = bg_type
simulation_params['bg_norm'] = bg_norm
simulation_params['bg_index'] = bg_index

# --- ZMENA: Načítanie prepínačov pre "noiseless" simuláciu ---
generate_noiseless = params.get('generate_noiseless', False)
include_background = params.get('include_background', True)
simulation_params['generate_noiseless'] = generate_noiseless
simulation_params['include_background'] = include_background
# --- KONIEC ZMENY ---


# --- Step 2: Prepare Grids and Models ---
min_energy = min(E_min for E_min, _ in energy_bands)
max_energy_in_bands = max(E_max for _, E_max in energy_bands)
# Extend full energy grid to cover the high-energy component up to its cutoff
max_energy_grid = max(max_energy_in_bands, E_cut * 1.5) # Extend 50% past cutoff

# Use more points for finer integration, especially with multiple components
E_full = np.logspace(np.log10(min(1.0, min_energy)), np.log10(max_energy_grid), 5000)
t = np.linspace(t_start, t_end, n_time_points)

# --- ZMENA: Presunuté z Kroku 4 ---
# Vypočítame šírku časového intervalu (bin)
time_bin_width = (t_end - t_start) / (n_time_points - 1)
# Vypočítame hrany intervalov pre histogram (potrebné pre 'noisy' mód)
edges = np.zeros(n_time_points + 1)
if n_time_points > 1:
    edges[1:-1] = 0.5 * (t[1:] + t[:-1]) # Vnútorné hrany
    edges[0] = t[0] - time_bin_width/2.0             # Prvá hrana
    edges[-1] = t[-1] + time_bin_width/2.0           # Posledná hrana
else:
    edges = np.array([t[0] - time_bin_width/2.0, t[0] + time_bin_width/2.0])
# --- KONIEC ZMENY ---

avg_energies = [(E_min + E_max) / 2 for (E_min, E_max) in energy_bands]
min_avg = min(avg_energies)
max_avg = max(avg_energies)

# Calculate t_peak for each band (spectral lag)
energy_peak_times = {}
peaks_df = pd.DataFrame(columns=["avg_energy", "t_peak"])
print("Calculating peak times (spectral lag)...") # <-- ADDED
for i, avg_energy in enumerate(avg_energies):
    # --- CHANGED: Logarithmic interpolation for t_peak ---
    # Use log of energies for normalization
    log_avg = np.log10(avg_energy)
    log_min = np.log10(min_avg)
    log_max = np.log10(max_avg)
    
    if log_max == log_min:
        norm = 0.0
    else:
        norm = (log_avg - log_min) / (log_max - log_min)
    # --- END CHANGED ---
    
    t_peak = t_peak_min + norm * (t_peak_max - t_peak_min)
    peaks_df.loc[i] = [avg_energy, t_peak]
    energy_peak_times[i] = t_peak
    print(f"  Band {i+1} (Avg Energy: {format_energy_label(avg_energy)}): t_peak = {t_peak:.4f} s") # <-- ADDED

# --- ZMENA: Krok 3 a 4 sú zlúčené ---

# --- Step 3: Simulate or Calculate Photon Counts ---
rng = np.random.default_rng()
counts_by_band = {} # Inicializujeme slovník pre počty
background_rates_per_band = [] # Toto ponecháme pre Plot C

print("Simulating or calculating photon counts for each energy band...")
for i, (E_min, E_max) in enumerate(energy_bands):
    
    # Vypočítame tok signálu
    signal_flux = band_integrated_flux(
        E_min, E_max, E_full, 
        E_peak, alpha, beta, E0, F0, 
        K_pl, E_pivot, gamma, E_cut, 
        area
    )
    
    # Modulujeme signál časovou krivkou
    lambda_signal = signal_flux * light_curve(t, energy_peak_times[i], rise_index, decay_index)
    
    # Vypočítame pozadie (iba ak je povolené)
    if include_background:
        bg_rate = calculate_background_rate(avg_energies[i], bg_norm, bg_index)
        background_rates_per_band.append(bg_rate) # Pre plot
        lambda_bg = np.full_like(t, bg_rate, dtype=float)
    else:
        bg_rate = 0.0 # Pre tlač
        background_rates_per_band.append(0.0) # Pre plot
        lambda_bg = np.zeros_like(t, dtype=float) # Žiadne pozadie
    
    # Celková rýchlosť (signal + pozadie)
    lambda_total = lambda_signal + lambda_bg
    
    # --- HLAVNÁ LOGIKA: NOISY vs NOISELESS ---
    if generate_noiseless:
        # NOISELESS CESTA: Priamy výpočet očakávaných hodnôt
        # Očakávané počty = rýchlosť (počet/s) * trvanie intervalu (s)
        expected_counts = lambda_total * time_bin_width
        counts_by_band[i] = expected_counts
        
        print(f"  Band {i+1} ({format_energy_label(E_min)}-{format_energy_label(E_max)}): "
              f"Calculated {np.sum(expected_counts):.2f} expected counts (Signal Flux: {signal_flux:.2f} c/s, BG Rate: {bg_rate:.2f} c/s).")
              
    else:
        # NOISY CESTA: Plná simulácia (pôvodná metóda)
        toas = simulate_toas_from_rate(t, lambda_total, rng=rng)
        
        # Okamžité binovanie (logika z pôvodného Kroku 4)
        counts_by_band[i] = np.histogram(toas, bins=edges)[0]
        
        print(f"  Band {i+1} ({format_energy_label(E_min)}-{format_energy_label(E_max)}): "
              f"{len(toas)} total photons simulated (Signal Flux: {signal_flux:.2f} c/s, BG Rate: {bg_rate:.2f} c/s).")
# --- KONIEC ZMENY ---


# --- Step 4: (Odstránený) Bin TOAs into Fixed-Width Bins ---
# Táto logika je teraz priamo v Kroku 3


# --- Step 5: Save Data to CSV ---

# --- ZMENA: Určíme výstupný dátový typ ---
output_dtype = float if generate_noiseless else int
# --- KONIEC ZMENY ---

data_counts = {'Time (s)': t}
for i, avg_energy in enumerate(avg_energies):
    # --- ZMENA: Použijeme output_dtype ---
    data_counts[avg_energy] = counts_by_band.get(i, np.zeros_like(t, dtype=output_dtype))
# --- KONIEC ZMENY ---

df_photon_counts = pd.DataFrame(data_counts)
csv_counts_filename = output_folder / 'photon_counts_over_time.csv'
df_photon_counts.to_csv(csv_counts_filename, index=False, float_format='%.10f')
print(f"\nSaved fixed-bin counts to: {csv_counts_filename}")

peaks_csv_filename = output_folder / 'posun_and_peaks_time.csv'
peaks_df['Difference'] = peaks_df['t_peak'].iloc[0] - peaks_df['t_peak']
peaks_df.to_csv(peaks_csv_filename, index=False)
print(f"Saved peak time data to: {peaks_csv_filename}")

# --- Step 6: Generate and Save Plots ---
print("Generating and saving plots...")
timestamp = int(time.time() * 1000)

# Plot A: Light Curves (Counts vs. Time)
n_cols = 2
n_rows = (len(energy_bands) + n_cols - 1) // n_cols
fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, max(6, 3 * n_rows)), sharex=True, squeeze=False)
axs_flat = axs.flatten()
for i, (E_min, E_max) in enumerate(energy_bands):
    ax = axs_flat[i]
    # .get(i, []) funguje pre noisy (int) aj noiseless (float)
    ax.plot(t, counts_by_band.get(i, []), color='hotpink', lw=1.2)
    ax.set_title(f'Counts: {format_energy_label(E_min)} – {format_energy_label(E_max)}')
    ax.grid(True)
    if i // n_cols == n_rows - 1 or i + n_cols >= len(energy_bands):
         ax.set_xlabel('Time (s)')
    if i % n_cols == 0:
         ax.set_ylabel('Counts per bin')

# Delete unused subplots
for j in range(len(energy_bands), len(axs_flat)):
    fig.delaxes(axs_flat[j])

plt.tight_layout()
path = common_graphs_folder / "photon_counts"
path.mkdir(parents=True, exist_ok=True)
common_path = path / f'photon_counts_run_{next_number}_{timestamp}.png'
run_path = run_graphs_folder / 'photon_counts.png'
plt.savefig(common_path, dpi=150, bbox_inches='tight')
shutil.copy(common_path, run_path)
plt.close()
print(f"  Saved Light Curve plot.")

# Plot B: Total Spectrum Model
plt.figure(figsize=(10, 6))
E_plot = np.logspace(np.log10(min(1.0, min_energy)), np.log10(max_energy_grid), 1000) # Plot from 1 keV or lower

# Calculate components
spec_band = band_function(E_plot, E_peak, alpha, beta, E0, F0) * area
spec_pl = power_law_cutoff(E_plot, K_pl, E_pivot, gamma, E_cut) * area
spec_total = spec_band + spec_pl

plt.loglog(E_plot, spec_total, label='Total Spectrum (Band + PL)', color='black', lw=2)
plt.loglog(E_plot, spec_band, label=f'Band Component (E_peak={format_energy_label(E_peak)})', linestyle='--', color='blue')
plt.loglog(E_plot, spec_pl, label=f'PL Component (gamma={gamma:.2f}, E_cut={format_energy_label(E_cut)})', linestyle='--', color='green')

# Add vertical lines for energy bands
for E_min, E_max in energy_bands:
    plt.axvspan(E_min, E_max, alpha=0.1, color='gray')

plt.xlabel("Energy (keV)"); plt.ylabel("Counts / s / keV")
plt.title("Total Spectral Model (scaled by area)")
plt.ylim(bottom=max(np.min(spec_total[spec_total > 0]), 1e-12)) # Avoid plotting zero on log scale
plt.legend()
plt.grid(True, which="both", ls="--")
path = common_graphs_folder / "total_spectrum"
path.mkdir(parents=True, exist_ok=True)
common_path = path / f'total_spectrum_run_{next_number}_{timestamp}.png'
run_path = run_graphs_folder / 'total_spectrum.png'
plt.savefig(common_path, dpi=150, bbox_inches='tight')
shutil.copy(common_path, run_path)
plt.close()
print(f"  Saved Total Spectrum plot.")

# Plot C: Background Model Visualization
plt.figure(figsize=(10, 6))
plt.plot(avg_energies, background_rates_per_band, marker='o', linestyle='--')
plt.xscale('log'); plt.yscale('log')
plt.xlabel("Energy Band Center (keV)"); plt.ylabel("Background Rate (counts/s)")
title_suffix = f"Rate = {bg_norm} * E^({bg_index})" if include_background else "Background Disabled"
plt.title(f"Background Model: {title_suffix}")
plt.grid(True, which="both", ls="--")
path = common_graphs_folder / "background_model"
path.mkdir(parents=True, exist_ok=True)
common_path = path / f'background_model_run_{next_number}_{timestamp}.png'
run_path = run_graphs_folder / 'background_model.png'
plt.savefig(common_path, dpi=150, bbox_inches='tight')
shutil.copy(common_path, run_path)
plt.close()
print(f"  Saved Background Model plot.")

# Plot D: Peak Time vs. Energy
plt.figure(figsize=(10, 6))
plt.plot(peaks_df['avg_energy'], peaks_df['t_peak'], marker='D', linestyle='-')
plt.xscale('log')
plt.xlabel("Energy Band Center (keV)"); plt.ylabel("Peak Time (s)")
plt.title("Embedded Peak Time vs. Energy")
plt.grid(True)
path = common_graphs_folder / "peak_time_vs_energy"
path.mkdir(parents=True, exist_ok=True)
common_path = path / f'peak_time_vs_energy_run_{next_number}_{timestamp}.png'
run_path = run_graphs_folder / 'peak_time_vs_energy.png'
plt.savefig(common_path, dpi=150, bbox_inches='tight')
shutil.copy(common_path, run_path)
plt.close()
print(f"  Saved Peak Time vs. Energy plot.")

# Plot E: Peak Difference Plot
plt.figure(figsize=(10, 6))
plt.scatter(peaks_df['avg_energy'], peaks_df['Difference'], color='hotpink', marker='D', label='Data points')
if len(peaks_df) >= 2:
    m, b = np.polyfit(peaks_df['avg_energy'], peaks_df['Difference'], 1)
    x_line = np.linspace(peaks_df['avg_energy'].min(), peaks_df['avg_energy'].max(), 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, label=f'Fit: y={m:.2e}x+{b:.2f}', color='deeppink', alpha=0.7)
plt.xscale('log')
plt.xlabel('Energy Band Center (keV)'); plt.ylabel('Time Shift (s)')
lowest_band_label = format_energy_label(energy_bands[0][0])
plt.title(f'Embedded Time Shifts Relative to Lowest Band ({lowest_band_label})')
plt.legend(); plt.grid(True)
path = common_graphs_folder / "peak_difference"
path.mkdir(parents=True, exist_ok=True)
common_path = path / f'peak_difference_run_{next_number}_{timestamp}.png'
run_path = run_graphs_folder / 'peak_difference.png'
plt.savefig(common_path, dpi=150, bbox_inches='tight')
shutil.copy(common_path, run_path)
plt.close()
print(f"  Saved Peak Difference plot.")

# Plot F: Relative Noise Level
plt.figure(figsize=(13, 6))
for i, (E_min, E_max) in enumerate(energy_bands):
    counts = counts_by_band.get(i, np.array([]))
    # Poisson noise is sqrt(N), relative noise is sqrt(N)/N = 1/sqrt(N)
    # np.where chráni pred delením nulou alebo log(0)
    relative_noise_percent = np.where(counts > 0, (1 / np.sqrt(counts)) * 100, np.nan)
    
    # Ak je noiseless, toto bude NaN, čo je v poriadku, plot sa nezobrazí
    if not np.all(np.isnan(relative_noise_percent)):
        plt.plot(t, relative_noise_percent, label=f'{format_energy_label(E_min)}-{format_energy_label(E_max)}')

title_suffix = "(1/√N) Over Time" if not generate_noiseless else "(Plot N/A for Noiseless Data)"
plt.xlabel('Time (s)'); plt.ylabel('Relative Noise Level (%)')
plt.title(f'Relative Noise Level {title_suffix}')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)); plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
path = common_graphs_folder / "relative_noise"
path.mkdir(parents=True, exist_ok=True)
common_path = path / f'relative_noise_run_{next_number}_{timestamp}.png'
run_path = run_graphs_folder / 'relative_noise.png'
plt.savefig(common_path, dpi=150, bbox_inches='tight')
shutil.copy(common_path, run_path)
plt.close()
print(f"  Saved Relative Noise plot.")

# --- Step 7: Finalize ---
params_filename = output_folder / 'current_parameters.txt'
print(f"\nSaving all *used* parameters to: {params_filename}") # <-- ADDED PRINT
with open(params_filename, 'w', encoding='utf-8') as f: # Pridané encoding
    f.write("Parameters used for this run (including defaults):\n")
    # --- CHANGED: Iterate over simulation_params, not params ---
    for key, value in simulation_params.items():
        f.write(f"{key} = {value}\n") # <-- Changed colon to = for consistency
print(f"\nSimulation complete. All outputs saved in: {output_folder}")