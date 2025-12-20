import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import time
import shutil

# ---------------------------
# Step 0: Find the latest simulation run
# ---------------------------
base_name = 'grb_simulation_with_background'
output_base = Path(base_name)

if not output_base.exists():
    raise FileNotFoundError(f"Base folder '{base_name}' was not found. Please run the simulation first.")

# Find all subdirectories and select the one with the highest number
existing_folders = [d for d in output_base.iterdir() if d.is_dir() and d.name.split('_')[-1].isdigit()]
if not existing_folders:
    raise FileNotFoundError(f"No results folders found in '{base_name}'.")

latest_folder = max(existing_folders, key=lambda d: int(d.name.split('_')[-1]))
csv_filename = latest_folder / 'photon_counts_over_time.csv'

if not csv_filename.exists():
    raise FileNotFoundError(f"CSV file not found at: {csv_filename}")

print("Loading CSV file from latest run:", csv_filename)
output_folder = latest_folder 

# ---------------------------
# Step 1: Load the simulated data
# ---------------------------
df = pd.read_csv(csv_filename)
print("\nData Preview:")
print(df.head())

# ---------------------------
# Step 2: Extract Time and Channel Data
# ---------------------------
time_vals = df['Time (s)'].values

# Get channel names (all columns except time) and sort them by energy
channel_names = [col for col in df.columns if col != 'Time (s)']
channel_names_sorted = sorted(channel_names, key=lambda x: float(x))
print("\nChannels (sorted by energy):", channel_names_sorted)

# The first (lowest energy) channel is our reference template 'A'
ref_channel_name = channel_names_sorted[0]
ref_channel_A = df[ref_channel_name].values

# ---------------------------
# Step 3: Calculate Deviations (UPDATED)
# ---------------------------
# We now store TWO things:
# 1. The Sign (+1, 0, -1) for autocorrelation and the sign histogram.
# 2. The Numeric Value (float) for the new numeric histogram.

deviations_by_channel = {}        # Stores signs
numeric_errors_by_channel = {}    # Stores actual values
deviations_by_channel[ref_channel_name] = np.zeros_like(ref_channel_A)
mean_A = np.mean(ref_channel_A)

print("\nCalculating deviations for each channel...")
for ch_name in channel_names_sorted[1:]:
    current_channel_B = df[ch_name].values
    mean_B = np.mean(current_channel_B)

    if mean_A > 0:
        predicted_B = ref_channel_A * (mean_B / mean_A)
    else:
        predicted_B = np.zeros_like(ref_channel_A)

    difference = current_channel_B - predicted_B
    
    if mean_B > 0:
        normalized_difference = difference / mean_B
    else:
        normalized_difference = np.zeros_like(difference)
    
    # --- SAVE RAW NUMERIC ERROR ---
    numeric_errors_by_channel[ch_name] = normalized_difference

    # --- SAVE SIGNS ---
    deviations = np.sign(normalized_difference)
    deviations_by_channel[ch_name] = deviations
    
    print(f"  Processed channel: {ch_name}")

# ---------------------------
# Step 4: Quantify and Save Deviation Bias
# ---------------------------
print("\nQuantifying overall deviation bias...")
deviation_summary = []
for ch_name in channel_names_sorted[1:]:
    deviations = deviations_by_channel[ch_name]
    mean_deviation = np.mean(deviations)
    deviation_summary.append({
        'Channel (keV)': ch_name,
        'Mean Deviation': mean_deviation
    })

summary_df = pd.DataFrame(deviation_summary)
summary_table_filename = output_folder / 'deviation_summary_table.csv'
summary_df.to_csv(summary_table_filename, index=False)
print(f"Saved deviation summary table: {summary_table_filename}")

# ---------------------------
# Step 5: Calculate Autocorrelation
# ---------------------------
print("\nCalculating autocorrelation...")
max_lag = 4
autocorrelations_by_lag = {lag: [] for lag in range(1, max_lag + 1)}
channel_energies_for_corr = []

for ch_name in channel_names_sorted[1:]:
    deviations = deviations_by_channel[ch_name]
    for lag in range(1, max_lag + 1):
        corr_vector = deviations[:-lag] * deviations[lag:]
        mean_corr = np.mean(corr_vector)
        autocorrelations_by_lag[lag].append(mean_corr)
    channel_energies_for_corr.append(float(ch_name))

# Save Autocorrelation Data
autocorr_data = {"Energy (keV)": channel_energies_for_corr}
for lag in range(1, max_lag + 1):
    autocorr_data[f"Lag {lag} Autocorr"] = autocorrelations_by_lag[lag]

autocorr_df = pd.DataFrame(autocorr_data)
autocorr_table_filename = output_folder / 'deviation_autocorrelation_table.csv'
autocorr_df.to_csv(autocorr_table_filename, index=False)

# ---------------------------
# Step 6: Save Full Deviations Table
# ---------------------------
deviations_df = pd.DataFrame({"Time (s)": time_vals})
for ch_name in channel_names_sorted:
    deviations_df[f"Dev_{ch_name}"] = deviations_by_channel[ch_name]
deviations_table_filename = output_folder / 'deviations_table.csv'
deviations_df.to_csv(deviations_table_filename, index=False)

# ---------------------------
# GRAPH SETUP
# ---------------------------
common_graphs_folder = output_folder.parent / "graphs"
common_graphs_folder.mkdir(parents=True, exist_ok=True)
run_id = output_folder.name
timestamp = int(time.time() * 1000)

# ---------------------------
# Step 7a: Plot Deviations Over Time (PNG Export)
# ---------------------------
print("\nGenerating deviations plot (PNG)...")
deviations_plot_folder = common_graphs_folder / "deviations_plot"
deviations_plot_folder.mkdir(parents=True, exist_ok=True)
plot_common_path = deviations_plot_folder / f'deviations_plot_{run_id}_{timestamp}.png'
plot_run_path = output_folder / 'deviations_plot.png'

n_channels_to_plot = len(channel_names_sorted) - 1

if n_channels_to_plot > 0:
    n_cols = 2 if n_channels_to_plot > 1 else 1
    n_rows = (n_channels_to_plot + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Channel {ch} keV" for ch in channel_names_sorted[1:]],
        shared_xaxes=True,
        vertical_spacing=0.08
    )

    for i, ch_name in enumerate(channel_names_sorted[1:]):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        deviations = deviations_by_channel[ch_name]
        
        # Add Step Plot trace
        fig.add_trace(
            go.Scatter(
                x=time_vals, 
                y=deviations, 
                mode='lines',
                line_shape='hv',  # Horizontal-Vertical step
                line=dict(color='darkviolet', width=1.5),
                name=f"{ch_name} keV"
            ),
            row=row, col=col
        )
        
        # Add Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="grey", row=row, col=col)
        
        # Update Y-axis ticks to show meaning
        fig.update_yaxes(
            tickvals=[-1, 0, 1],
            ticktext=["Dimmer", "Same", "Brighter"],
            range=[-1.5, 1.5],
            row=row, col=col
        )

    fig.update_layout(
        height=300 * n_rows, 
        width=1200, 
        title_text=f"Deviation Pattern vs. Time (Ref: {ref_channel_name} keV)",
        showlegend=False,
        plot_bgcolor='white'
    )
    
    # Save as PNG
    try:
        fig.write_image(plot_common_path, scale=3) # Scale 3 for high resolution
        shutil.copy(plot_common_path, plot_run_path)
        print(f"Saved deviations plot: {plot_run_path}")
    except ValueError as e:
        print("\nERROR SAVING IMAGE: You might need to install kaleido.")
        print("Run: pip install -U kaleido")
        print(f"Details: {e}")

else:
    print("Not enough channels to plot.")

# ---------------------------
# Step 7b: Plot Autocorrelation (PNG Export) - PRESERVED
# ---------------------------
print("Generating autocorrelation plot (PNG)...")
autocorr_plot_folder = common_graphs_folder / "deviation_autocorrelation"
autocorr_plot_folder.mkdir(parents=True, exist_ok=True)
plot_common_path = autocorr_plot_folder / f'deviation_autocorr_{run_id}_{timestamp}.png'
plot_run_path = output_folder / 'deviation_autocorrelation.png'

fig_corr = go.Figure()

colors = ['indigo', 'darkviolet', 'hotpink', 'pink']
for lag, color in zip(range(1, max_lag + 1), colors):
    fig_corr.add_trace(go.Scatter(
        x=channel_energies_for_corr,
        y=autocorrelations_by_lag[lag],
        mode='lines+markers',
        name=f'Lag {lag}',
        line=dict(color=color),
        marker=dict(symbol='diamond')
    ))

fig_corr.update_layout(
    title='Temporal Autocorrelation of Deviation Series vs. Energy',
    xaxis_title='Channel Energy (keV)',
    yaxis_title='Autocorrelation',
    xaxis_type='log', # Semilog X
    yaxis_range=[-1.05, 1.05],
    plot_bgcolor='white'
)
fig_corr.add_hline(y=0, line_dash="dash", line_color="grey")
fig_corr.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig_corr.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

try:
    fig_corr.write_image(plot_common_path, scale=3)
    shutil.copy(plot_common_path, plot_run_path)
    print(f"Saved autocorrelation plot: {plot_run_path}")
except ValueError:
    print("Error saving autocorrelation plot. Check kaleido installation.")

# ---------------------------
# Step 8: Histogram of Deviation Signs (PNG Export) - PRESERVED
# ---------------------------
print("Generating sign histogram (PNG)...")

# 1. Prepare Data
sign_counts = {'Channel': [], 'Minus': [], 'Zero': [], 'Plus': []}

for ch_name in channel_names_sorted[1:]:
    devs = deviations_by_channel[ch_name]
    sign_counts['Channel'].append(ch_name)
    sign_counts['Minus'].append(np.sum(devs == -1))
    sign_counts['Zero'].append(np.sum(devs == 0))
    sign_counts['Plus'].append(np.sum(devs == 1))

sign_df = pd.DataFrame(sign_counts)
sign_df.to_csv(output_folder / 'deviation_sign_counts.csv', index=False)

# 2. Plotting
hist_plot_folder = common_graphs_folder / "sign_histogram"
hist_plot_folder.mkdir(parents=True, exist_ok=True)
plot_common_path = hist_plot_folder / f'sign_histogram_{run_id}_{timestamp}.png'
plot_run_path = output_folder / 'deviation_sign_histogram.png'

fig_hist = go.Figure()

fig_hist.add_trace(go.Bar(
    name='Dimmer (-1)',
    x=sign_df['Channel'], y=sign_df['Minus'],
    marker_color='cornflowerblue'
))

fig_hist.add_trace(go.Bar(
    name='Same (0)',
    x=sign_df['Channel'], y=sign_df['Zero'],
    marker_color='lightgrey'
))

fig_hist.add_trace(go.Bar(
    name='Brighter (+1)',
    x=sign_df['Channel'], y=sign_df['Plus'],
    marker_color='salmon'
))

fig_hist.update_layout(
    barmode='stack',
    title='Distribution of Deviation Signs per Channel',
    xaxis_title='Channel Energy (keV)',
    yaxis_title='Count of Time Steps',
    plot_bgcolor='white'
)
fig_hist.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

try:
    fig_hist.write_image(plot_common_path, scale=3)
    shutil.copy(plot_common_path, plot_run_path)
    print(f"Saved sign histogram plot: {plot_run_path}")
except ValueError:
    print("Error saving histogram plot. Check kaleido installation.")


# ---------------------------
# Step 9: Histogram of NUMERIC Errors (NEW)
# ---------------------------
print("Generating numeric error histograms (PNG)...")
num_hist_plot_folder = common_graphs_folder / "numeric_error_histograms"
num_hist_plot_folder.mkdir(parents=True, exist_ok=True)
plot_common_path = num_hist_plot_folder / f'numeric_hist_{run_id}_{timestamp}.png'
plot_run_path = output_folder / 'numeric_error_histogram.png'

n_channels_to_plot = len(channel_names_sorted) - 1

if n_channels_to_plot > 0:
    n_cols = 2 if n_channels_to_plot > 1 else 1
    n_rows = (n_channels_to_plot + n_cols - 1) // n_cols

    fig_num = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Channel {ch} keV" for ch in channel_names_sorted[1:]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    for i, ch_name in enumerate(channel_names_sorted[1:]):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        # Access the raw numeric data we stored in Step 3
        raw_values = numeric_errors_by_channel[ch_name]
        
        fig_num.add_trace(
            go.Histogram(
                x=raw_values,
                nbinsx=40, # Adjust bin size as needed
                marker_color='teal',
                opacity=0.75,
                name=f"{ch_name} keV"
            ),
            row=row, col=col
        )
        
        # Add vertical line at 0 (No error)
        fig_num.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, row=row, col=col)
        
        # Update axes
        fig_num.update_xaxes(title_text="Norm. Error", row=row, col=col)

    fig_num.update_layout(
        height=300 * n_rows, 
        width=1200, 
        title_text="Distribution of Numeric Errors (Normalized Difference)",
        showlegend=False,
        plot_bgcolor='white',
        bargap=0.05
    )

    try:
        fig_num.write_image(plot_common_path, scale=3)
        shutil.copy(plot_common_path, plot_run_path)
        print(f"Saved numeric histogram plot: {plot_run_path}")
    except ValueError:
        print("Error saving numeric histogram. Check kaleido installation.")