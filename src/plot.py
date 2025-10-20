# plot.py
import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import compute_lat_grid, compute_dipole_moment

def generate_all_plots(config, model, train_state):
    os.makedirs(config.output_dir, exist_ok=True)

    # Generate lat-lon grid
    lat_grid = compute_lat_grid(config)
    time_array = np.linspace(0, config.Tmax, config.num_time_points + 1)

    # Predict magnetic field
    X, T, B = [], [], []
    for j in lat_grid['lat_cen'] / np.pi:
        lat = np.ones_like(time_array) * j
        X.append(lat)
        T.append(time_array)
        coords = np.stack((lat, time_array), axis=1)
        B.append(model.predict(coords))

    B = np.array(B).squeeze().T
    np.save(os.path.join(config.output_dir, "field.npy"), B)

    # Magnetic field plot
    plt.figure(figsize=(6, 6))
    plt.imshow(B.T * config.B_unit, aspect="auto", origin="lower",
               extent=[0, config.simul_time, config.lat_min, config.lat_max])
    plt.colorbar(label="Magnetic field (Gauss)")
    plt.xlabel("Time (Years)")
    plt.ylabel("Latitude (deg)")
    plt.title("1D SFT Magnetic Field")
    plt.savefig(os.path.join(config.output_dir, "magnetic_field.png"))
    plt.close()

    # Dipole moment plot
    dip = compute_dipole_moment(B, lat_grid['lat_cen'], config)
    plt.figure(figsize=(7, 4))
    plt.plot(time_array * config.simul_time, dip)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("Time (Years)")
    plt.ylabel("Dipole Moment")
    plt.title("Dipole Moment Evolution")
    plt.savefig(os.path.join(config.output_dir, "dipole_moment.png"))
    plt.close()

    vmin, vmax = np.percentile(B, [2, 98])  # robust scaling for PINN output
    plt.contourf(T, X, B.T, levels=100, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    # Also add a polar-cap time series for |lat| ≥ 60°
    polar_mask = np.abs(lat_grid['lat_cen']*180/np.pi) >= 60
    polar_mean = B[:, polar_mask].mean(axis=1)
    plt.figure(figsize=(7, 3.2))
    plt.plot(time_array * config.simul_time, polar_mean)
    plt.axhline(0, ls="--", c="k"); plt.ylabel("⟨B⟩_|lat|≥60° [G]"); plt.xlabel("Time [yr]")
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "polar_cap_mean.png"))
