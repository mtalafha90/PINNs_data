# run_experiments.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src import sft_pde
from src.config import Config
from src.train import train_model
from src.plot import compute_dipole_moment, compute_lat_grid, generate_all_plots
from src.utils import compute_amplitudes_from_gaussians
from src.extract import build_synoptic_source, plot_synoptic_map

# Experiment parameters
quenching_types = ["TQ+LQ"]  # you can add ["TQ", "LQ", "none"]
colors = {"none": "k", "TQ": "b", "LQ": "g", "TQ+LQ": "r"}
markers = {"none": "o", "TQ": "s", "LQ": "^", "TQ+LQ": "D"}

results = {}

for qtype in quenching_types:
    print(f"Running: {qtype}")

    # Setup config
    config = Config(mode="", output_dir=f"results/q_{qtype}", quenching_type=qtype)

    # Use the same WSO dataset everywhere
    cycle_data_dir = config.wso_path

    # Random fluctuation array
    np.random.seed(42)
    base_gauss = np.random.normal(loc=0.0, scale=0.13, size=config.num_cycles)
    sft_pde.random_array_tf = tf.convert_to_tensor(base_gauss, dtype=tf.float32)

    # Reset amplitude tracking
    sft_pde.ampl_used_by_time.clear()

    # Build synoptic source from WSO data
    build_synoptic_source(data_dir=cycle_data_dir)

    # Train model
    model, train_state = train_model(config)
    generate_all_plots(config, model, train_state)

    # Compute amplitudes
    amplitudes_real = compute_amplitudes_from_gaussians(base_gauss, config)

    # Compute dipole moment
    lat_grid = compute_lat_grid(config)
    B = np.load(os.path.join(config.output_dir, "field.npy"))
    dipole = compute_dipole_moment(B, lat_grid["lat_cen"], config)
    NpolarB = np.mean(B[:, lat_grid["polar_mask"]], axis=1)

    # Diagnostics
    dipcycmin = dipole[np.argmin(amplitudes_real)]
    enddip = dipole[-1]
    final_dip = abs(enddip - dipcycmin * np.exp(-11 / sft_pde.tau * 365.25))

    results[qtype] = (amplitudes_real, final_dip)

    # Plot final synoptic map for the entire solar cycle
    plot_synoptic_map(data_dir=cycle_data_dir,
                      save_path=os.path.join(config.output_dir, f"{cycle_data_dir.replace('data/', '')}_map.png"))
