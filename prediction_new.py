import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


def estimate_poly_background(counts, tc, bg_intervals, degree=2):
    """
    Estimate a time-dependent background using a polynomial fit.
    
    Parameters:
    - tc: time bin centers
    - counts: counts per bin
    - bg_intervals: list of (t_min, t_max) intervals used as background (off-burst)
    - degree: polynomial degree
    
    Returns:
    - bg_model: background model evaluated at all tc
    - poly: np.poly1d polynomial object
    - bg_mask: boolean mask of bins used for the background fit
    """
    bg_mask = np.zeros_like(tc, dtype=bool)
    for tmin_bg, tmax_bg in bg_intervals:
        bg_mask |= (tc >= tmin_bg) & (tc <= tmax_bg)

    if not np.any(bg_mask):
        print("No background bins, setting background to 0")
        return np.zeros_like(counts, dtype=float), None, bg_mask

    tc_bg = tc[bg_mask]
    counts_bg = counts[bg_mask]

    # Poisson noise: sigma ~ sqrt(N), weights ~ 1/sigma
    sigma_bg = np.sqrt(np.clip(counts_bg.astype(float), 1.0, None))
    w = 1.0 / sigma_bg

    coeffs = np.polyfit(tc_bg, counts_bg, deg=degree, w=w)
    poly = np.poly1d(coeffs)

    bg_model = poly(tc)

    return bg_model, poly, bg_mask


def plot_raw_bg_net(tc, counts, bg_model, counts_net, title_band):
    plt.step(tc, counts,     where="mid", label="raw")
    plt.step(tc, bg_model,   where="mid", linestyle="--", label="background")
    plt.step(tc, counts_net, where="mid", label="net")
    plt.ylabel("Counts/bin")
    plt.title(title_band)
    plt.legend()


def lag1_autocorr(sign_residuals):
    mask = (sign_residuals != 0) # ignore 0
    c_eff = sign_residuals[mask]

    if len(c_eff) > 1:
        rho = np.corrcoef(c_eff[:-1], c_eff[1:])[0, 1] # correlation for sign residuals
        return rho
    else:
        return np.nan


def plot_low_vs_high_net_lc(tc, counts_low_net, low_band_keV, counts_high_net, high_band_keV, tmin, tmax):
    Emin, Emax = high_band_keV

    plt.figure(figsize=(8, 3))
    plt.step(tc, counts_low_net,  where="mid", label=f"Low {low_band_keV[0]}–{low_band_keV[1]} keV")
    plt.step(tc, counts_high_net, where="mid", label=f"High {Emin}–{Emax} keV")

    plt.xlim(tmin, tmax)
    plt.xlabel("Time since trigger [s]")
    plt.ylabel("Counts/bin (net)")
    plt.title("Netto light curves: low vs high band")
    plt.legend()
    plt.tight_layout()
    plt.show()


# settings

filename = "glg_tte_n0_bn160509374_v00.fit"

# binning for LC
dt = 0.05
tmin = -1.0
tmax = 100

# low-energy band
low_band_keV = (8, 50)

# high-energy bands
high_bands = [(100, 150), (200, 300), (500, 700)]

# background intervals (off-burst)
bg_intervals = [
    (-1.0, -0.2),   # before burst
    (31, 100),    # after burst
]

poly_degree = 2

# data loading and preparation

with fits.open(filename) as hdul:
    t = hdul["EVENTS"].data["TIME"].astype(float)
    pha = hdul["EVENTS"].data["PHA"]
    eb = hdul["EBOUNDS"].data
    t0 = hdul[0].header.get("TRIGTIME", t.min())

# time since trigger
t = t - t0

# geometric mean
E_mid = np.sqrt(eb["E_MIN"][pha] * eb["E_MAX"][pha])  # keV

# time mask
time_mask = (t >= tmin) & (t <= tmax)

# bins
edges = np.arange(tmin, tmax + dt, dt)
tc = 0.5 * (edges[:-1] + edges[1:])


# reference low band (histogram + polynomial background)


# mask for low band
low_mask = (E_mid >= low_band_keV[0]) & (E_mid < low_band_keV[1])

# histogram (raw counts/bin) for low band
counts_low, _ = np.histogram(t[time_mask & low_mask], bins=edges)

# poly background for low band
bg_model_low, poly_low, bg_mask_low = estimate_poly_background(counts_low, tc, bg_intervals, degree=poly_degree)

print(f"Number of background bins (low band): {bg_mask_low.sum()}")

if poly_low is not None:
    mean_bg_low = bg_model_low[bg_mask_low].mean()
else:
    mean_bg_low = 0.0
    
print(f"Mean background estimation (low band, poly deg={poly_degree}): ~ {mean_bg_low:.3f} counts/bin")

# netto counts
counts_low_net = np.clip(counts_low - bg_model_low, 0, None)

# graph for reference (low) band
plt.figure(figsize=(9, 4))
plot_raw_bg_net(tc, counts_low, bg_model_low, counts_low_net, f"Low band ({low_band_keV[0]}–{low_band_keV[1]} keV)")
plt.xlim(tmin, tmax)
plt.xlabel("Time since trigger [s]")
plt.tight_layout()
plt.show()

# mean netto signal in low band
m_low = counts_low_net.mean()
print(f"Mean netto counts (low band): {m_low:.3f} counts/bin")


# predictor method

rho_values = []

print("\n=== predictor method for multiple high bands ===")

for Emin, Emax in high_bands:
    print(f"\nHigh band: {Emin}–{Emax} keV")

    # mask and histogram for high bands (raw counts)
    high_mask = (E_mid >= Emin) & (E_mid < Emax)
    counts_high, _ = np.histogram(t[time_mask & high_mask], bins=edges)

    # poly background for high band
    bg_model_high, poly_high, bg_mask_high = estimate_poly_background(counts_high, tc, bg_intervals, degree=poly_degree)

    if poly_high is not None:
        mean_bg_high = bg_model_high[bg_mask_high].mean()
    else:
        mean_bg_high = 0.0

    counts_high_net = np.clip(counts_high - bg_model_high, 0, None)

    print(f"Number of background bins (high band): {bg_mask_high.sum()}")
    print(f"Mean background estimation (high band, poly deg={poly_degree}): ~ {mean_bg_high:.3f} counts/bin")
    print(f"Sum net counts low:  {counts_low_net.sum():.1f}")
    print(f"Sum net counts high: {counts_high_net.sum():.1f}")

    # netto light curves low + high band
    plot_low_vs_high_net_lc(tc, counts_low_net, low_band_keV, counts_high_net, (Emin, Emax), tmin, tmax)

    # high band prediction from low band
    m_high = counts_high_net.mean()
    alpha = (m_high / m_low) if m_low > 0 else 0.0
    counts_high_pred = counts_low_net * alpha

    # residuals a sign
    residuals_high = counts_high_pred - counts_high_net
    c_high = np.sign(residuals_high)

    # lag-1 correlation sign(residuals)
    rho_high = lag1_autocorr(c_high)
    print(f"rho (lag = 1 bin = {dt} s) = {rho_high:.4f}")

    # mean energy of the band
    E_mean = np.sqrt(Emin * Emax)
    rho_values.append((E_mean, rho_high))

# results

print("\n=== Results: E_mean vs rho ===")
for E_mean, rho in rho_values:
    print(f"E_mean ~ {E_mean:7.1f} keV  -->  rho = {rho: .4f}")

E_arr   = np.array([x[0] for x in rho_values])
rho_arr = np.array([x[1] for x in rho_values])

plt.figure(figsize=(6, 4))
plt.plot(E_arr, rho_arr, marker="o")
plt.xlabel("Mean energy of high band [keV]")
plt.ylabel("rho (lag = 1 bin)")
plt.title("Energy dependence of predictor-method correlation")
plt.grid(True)
plt.tight_layout()
plt.show()



