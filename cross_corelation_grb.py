import pandas as pd
from scipy.signal import correlate
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path
import time
import shutil

# ---------------------------------------------------------------------------
# Step 0: Locate and load the output folder from the previous simulation.
# ---------------------------------------------------------------------------
with open('latest_folder.txt', 'r') as f:
    output_folder = Path(f.read().strip())

# Definujeme spoločnú zložku pre grafy, ktorá bola vytvorená v predchádzajúcom kóde.
# V predchádzajúcom kóde bola zložka "graphs" vytvorená v base folder (t.j. output_folder.parent).
common_graphs_folder = output_folder.parent / "graphs"
common_graphs_folder.mkdir(parents=True, exist_ok=True)

# Load the CSV file created by the simulation script.
csv_counts = output_folder / 'photon_counts_over_time.csv'
df_photon_counts = pd.read_csv(csv_counts)

# Extract time column and photon counts for each energy band.
time_vals = df_photon_counts['Time (s)']
photon_counts_columns = df_photon_counts.columns[1:]  # Exclude the time column

# ---------------------------------------------------------------------------
# Step 1: Calculate cross-correlation between each pair of energy bands.
# ---------------------------------------------------------------------------
cross_correlations = {}
for i in range(len(photon_counts_columns)):
    for j in range(i + 1, len(photon_counts_columns)):
        # Perform cross-correlation between two energy bands.
        correlation = correlate(df_photon_counts[photon_counts_columns[i]],
                                df_photon_counts[photon_counts_columns[j]],
                                mode='full')
        max_correlation = correlation.max()
        # Calculate the shift (lag) index; 0 znamená žiadny posun.
        shift_index = correlation.argmax() - (len(correlation) // 2)
        cross_correlations[(photon_counts_columns[i], photon_counts_columns[j])] = (max_correlation, shift_index)

# Convert cross-correlation results to a DataFrame for lepšiu prehľadnosť.
cross_corr_df = pd.DataFrame(
    [(pair[0], pair[1], values[0], values[1]) for pair, values in cross_correlations.items()],
    columns=['Energy Band 1', 'Energy Band 2', 'Max Correlation', 'Time Shift']
)

# Prevod stĺpcov "Energy Band 1" a "Energy Band 2" na numerické hodnoty.
cross_corr_df['Energy Band 1'] = pd.to_numeric(cross_corr_df['Energy Band 1'], errors='coerce')
cross_corr_df['Energy Band 2'] = pd.to_numeric(cross_corr_df['Energy Band 2'], errors='coerce')

# Odstránenie riadkov s NaN, ktoré vznikli napr. pri konverzii.
cross_corr_df = cross_corr_df.dropna(subset=['Energy Band 1', 'Energy Band 2'])

# Prevod časového posunu zo "steps" na sekundy.
time_step_duration = time_vals.iloc[1] - time_vals.iloc[0]
cross_corr_df['Time Shift (s)'] = cross_corr_df['Time Shift'] * time_step_duration

# Uloženie výsledkov cross-korelačnej analýzy (CSV súbor ukladáme do aktuálneho behu).
cross_corr_df.to_csv(output_folder / 'cross_corr.csv', index=False)

# ---------------------------------------------------------------------------
# Step 2: Filter to only keep correlations with the lowest energy band.
# ---------------------------------------------------------------------------
lowest_band = photon_counts_columns[0]
relative_shifts = cross_corr_df[cross_corr_df['Energy Band 1'] == float(lowest_band)].copy()

# Zabezpečíme numerickú konzistenciu pre lineárnu regresiu.
relative_shifts.loc[:, 'Energy Band 2'] = pd.to_numeric(relative_shifts['Energy Band 2'], errors='coerce')
relative_shifts.loc[:, 'Time Shift (s)'] = pd.to_numeric(relative_shifts['Time Shift (s)'], errors='coerce')
relative_shifts = relative_shifts.dropna(subset=['Energy Band 2', 'Time Shift (s)'])

# ---------------------------------------------------------------------------
# Step 3: Perform linear regression and compute fitted line.
# ---------------------------------------------------------------------------
slope, intercept, r_value, p_value, std_err = stats.linregress(
    relative_shifts['Energy Band 2'], relative_shifts['Time Shift (s)'])
fit = slope * relative_shifts['Energy Band 2'] + intercept

# ---------------------------------------------------------------------------
# Step 4: Plot time shifts of higher energy bands relative to the lowest band (smooth fit)
# ---------------------------------------------------------------------------

# Extrahujeme x a y
x = relative_shifts['Energy Band 2'].values
y = relative_shifts['Time Shift (s)'].values

# Vygenerujeme hladké body pre priamku na log-škále
x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
y_fit = slope * x_fit + intercept  # slope a intercept už máte z Step 3

# Vykreslíme scatter dáta a hladkú fit priamku
plt.figure(figsize=(10, 6))
plt.scatter(x, y,
            color='hotpink', marker='D', label='Data points')
plt.plot(x_fit, y_fit, color='hotpink', alpha=0.5 ,label=f'Fit: y={slope:.2e}·x+{intercept:.2f}')
plt.xscale('log')
plt.xlabel('Higher Energy Bands (keV)')
plt.ylabel('Time Shift (s)')
plt.title(f'Time Shifts Relative to Lowest Energy Band: {lowest_band} keV')
plt.legend()
plt.tight_layout()

# Na základe názvu aktuálneho run folderu získame run_id (napr. "skuska mastru 4_2").
run_id = output_folder.name
timestamp = int(time.time() * 1000)  # časová pečiatka pre jedinečné názvy súborov

# Uloženie grafu rovnako ako predtým...
timestamp = int(time.time() * 1000)
delay_common_filename = f'delay_from_lowest_cross_{run_id}_{timestamp}.png'
delay_run_filename    = f'delay_from_lowest_cross_{run_id}.png'
delay_specific_folder = common_graphs_folder / "delay_from_lowest"
delay_specific_folder.mkdir(parents=True, exist_ok=True)
delay_common_path = delay_specific_folder / delay_common_filename
delay_run_path    = output_folder / delay_run_filename

plt.savefig(delay_common_path)
shutil.copy(delay_common_path, delay_run_path)
plt.close()
print(f"Saved delay-from-lowest graph in common graphs folder: {delay_common_path}")
print(f"Saved delay-from-lowest graph in run folder: {delay_run_path}")

# ---------------------------------------------------------------------------
# Step 5: Residual analysis to check linearity.
# ---------------------------------------------------------------------------
residuals = relative_shifts['Time Shift (s)'] - fit
mean_residual = np.mean(np.abs(residuals))

plt.figure(figsize=(10, 6))
plt.scatter(relative_shifts['Energy Band 2'], residuals, color='hotpink',
            label='Residuals', marker='D')
plt.axhline(0, color='darkmagenta', linestyle='--', alpha=0.5)
plt.xlabel('Higher Energy Bands (keV)')
plt.ylabel('Residuals (s)')
plt.title('Residuals of Linear Fit')
#plt.grid(True)
plt.tight_layout()

# Definovanie názvov súborov pre residuals graf.
residual_common_filename = f'residuals_plot_{run_id}_{timestamp}.png'
residual_run_filename = f'residuals_plot_{run_id}.png'

# Vytvorenie podadresára pre residuals grafy.
residual_specific_folder = common_graphs_folder / "residuals"
residual_specific_folder.mkdir(parents=True, exist_ok=True)
residual_common_path = residual_specific_folder / residual_common_filename
residual_run_path = output_folder / residual_run_filename

plt.savefig(residual_common_path)
shutil.copy(residual_common_path, residual_run_path)
print(f"Saved residuals plot in common graphs folder: {residual_common_path}")
print(f"Saved residuals plot in run folder: {residual_run_path}")
plt.close()

# ---------------------------------------------------------------------------
# Step 6: Save residuals table.
# ---------------------------------------------------------------------------
residuals_df = pd.DataFrame({
    'Energy Band 2': relative_shifts['Energy Band 2'],
    'Residuals (s)': residuals
})
# Vytvoríme priečinok pre tabuľky a uložíme súbor so zvyškovou analýzou.
table_folder = output_folder / "tables_from_cross"
table_folder.mkdir(parents=True, exist_ok=True)
residuals_table_filename = table_folder / 'residuals_table.csv'
residuals_df.to_csv(residuals_table_filename, index=False)

# ---------------------------------------------------------------------------
# Step 7: Create table from cross-correlation for additional comparison.
# ---------------------------------------------------------------------------
in_peaks = pd.read_csv(output_folder / 'posun_and_peaks_time.csv')
in_peaks4 = in_peaks.tail(4).copy()
first_4_rows = cross_corr_df.head(4).copy()
first_4_rows.reset_index(drop=True, inplace=True)
in_peaks4.reset_index(drop=True, inplace=True)
in_peaks4['Posun od prvého binu z cross_corr'] = first_4_rows['Time Shift (s)']
in_peaks4['rozdiel cross a in'] = in_peaks4['Difference'] - first_4_rows['Time Shift (s)']

table_from_cross_path = table_folder / 'table_from_cross.csv'
in_peaks4.to_csv(table_from_cross_path, index=False)

# ---------------------------------------------------------------------------
# Step 8: Print additional results.
# ---------------------------------------------------------------------------
print(in_peaks4)

# ---------------------------------------------------------------------------
# Step 9: Save fitted parameters to a text file.
# ---------------------------------------------------------------------------
params_folder = output_folder / "fitted_parameters_files"
params_folder.mkdir(parents=True, exist_ok=True)
fitted_params_filename = params_folder / 'fitted_parameters.txt'
with open(fitted_params_filename, 'w') as f:
    f.write(f"Slope: {slope}\n")
    f.write(f"Intercept: {intercept}\n")
    f.write(f"R-value: {r_value}\n")
    f.write(f"P-value: {p_value}\n")
    f.write(f"Standard Error: {std_err}\n")
    f.write(f"Mean Residual: {mean_residual}\n")

print(f"Saved fitted parameters in: {fitted_params_filename}")