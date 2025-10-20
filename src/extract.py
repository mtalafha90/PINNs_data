import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interp1d
import datetime
from collections import defaultdict

synoptic_source_interp = None
initial_lats = None
initial_profile = None

def load_full_wso_data(filepath):
    data = []
    ct_labels = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].startswith("CT"):
            ct_label = lines[i].split()[0]  # e.g., CT2293:360
            ct_labels.append(ct_label)

            row_data = []

            for j in range(4):  # Read 4 lines per synoptic row
                line = lines[i + j].strip().split()

                # Skip the CT label on the first line
                if j == 0 and ':' in line[0]:
                    line = line[1:]

                # Convert remaining strings to float
                row_data.extend(map(float, line))

            if len(row_data) != 30:
                raise ValueError(f"Expected 30 values per CT row, got {len(row_data)} at {ct_label}")

            data.append(row_data)
            i += 4  # Move to next CT block
        else:
            i += 1

    return np.array(data), ct_labels


def build_synoptic_map(data_dir="data/24", lat_points=360):
    file_paths = sorted(glob.glob(os.path.join(data_dir, "WSO.*.F.txt")))
    if not file_paths:
        raise FileNotFoundError(f"No WSO files found in {data_dir}")
    full_map = []
    rotation_nums = []
    data_rows=[]

    # WSO latitudes
    sinlats = np.linspace(14.5 / 15, -14.5 / 15, 30)
    lats_deg = np.arcsin(sinlats) * 180 / np.pi
    model_lats = np.linspace(-90, 90, lat_points)

    # Reference Carrington rotation number and date
    if data_dir =="data/25":
        ref_rot = 2225  
        ref_date = datetime.datetime(2019, 12, 10)  
    if data_dir =="data/24":
        ref_rot = 2078  
        ref_date = datetime.datetime(2008, 12, 17) 
    if data_dir =="data/23":
        ref_rot = 1913  
        ref_date = datetime.datetime(1996, 8, 22)  
    if data_dir =="data/22":
        ref_rot = 1780  
        ref_date = datetime.datetime(1986, 9, 16)  
    if data_dir =="data/21":
        ref_rot = 1614 
        ref_date = datetime.datetime(1974, 4, 24)  
    print (ref_date)
    carrington_period_days = 27.2753

    for filepath in file_paths:
        with open(filepath, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            if lines[i].startswith("CT"):
                ct_label = lines[i].split()[0]  # e.g., CT2293:360
                rot_num = int(ct_label[2:6])
                longitude = int(ct_label.split(":")[1])
                #print(rot_num)
                row_data = []
                for j in range(4):
                    parts = lines[i + j].strip().split()
                    if j == 0 and ':' in parts[0]:
                        parts = parts[1:]
                    row_data.extend(map(float, parts))
                i += 4

                if len(row_data) == 30:
                    data_rows.append({
                        'rotation':rot_num,
                        'longitude':longitude,
                        'flux':np.array(row_data)
                        })
                    # Interpolate to model lat grid
                    #interp_func = interp1d(lats_deg, row_data, kind='cubic', fill_value='extrapolate')
                    #full_map.append(interp_func(model_lats))
                    rotation_nums.append(rot_num)
            else:
                i += 1
    # Group flux arrays by rotation number
    rotation_flux_map = defaultdict(list)
    for entry in data_rows:
        rot = entry['rotation']
        rotation_flux_map[rot].append(entry['flux'])
    rotation_nums = sorted(rotation_flux_map.keys())
    mean_profiles = []
    for rot in rotation_nums:
        fluxes = np.array(rotation_flux_map[rot])  # shape: (N_longitudes, 30)
        mean_profile = np.mean(fluxes, axis=0)     # average over longitude
        mean_profiles.append(mean_profile)
    #print (len(mean_profiles))

    # Convert rotation numbers to datetime
    dates = [ref_date + datetime.timedelta(days=(rot - ref_rot) * 27.2753) for rot in rotation_nums]
    # Convert datetime to float years (for plotting)
    years_float = np.array([d.year + d.timetuple().tm_yday / 365.25 for d in dates])
    days_since_start = np.array([(d - ref_date).days for d in dates])

    synoptic_map = np.array(mean_profiles)  # shape: (num_rotations, lat_points)
    return days_since_start, lats_deg, synoptic_map, dates

synoptic_source_interp = None  # global variable

def _remove_monopole_per_time(B_2D):
    """Subtract the latitude-mean at each time so net monopole ≈ 0."""
    return B_2D - np.mean(B_2D, axis=1, keepdims=True)

def _build_interp_from_arrays(t_vec, lat_deg, B_2D):
    T, L = np.meshgrid(t_vec, lat_deg, indexing="ij")
    pts  = np.column_stack([T.ravel(), L.ravel()])
    vals = B_2D.ravel()
    return LinearNDInterpolator(pts, vals, fill_value=0.0)


def build_synoptic_source(data_dir, lat_points=360, config=None, return_arrays=False):
    """
    Build a balanced + scaled synoptic source interpolator:
      synoptic_source_interp(lat_deg, t_years) -> Gauss

    data_dir     : folder with WSO files
    lat_points   : target uniform latitude grid size (default 360)
    config       : expects .source_scale (float), .start_year (optional)
    return_arrays: if True, also returns (t_years, model_lats, B_bal_scaled)

    Side effects:
      sets globals: synoptic_source_interp, initial_lats, initial_profile
    """

    global synoptic_source_interp, initial_lats, initial_profile

    # 1) Load your synoptic map (your existing helper)
    # Returns:
    #   days_since_start : (Nt,) monotonically increasing (can be unsorted initially)
    #   lats_deg         : (Nlat,) in degrees (can be unsorted)
    #   syn_map          : (Nt, Nlat) Gauss (longitudinally averaged)
    #   dates            : (Nt,) datetime or strings (optional)
    days_since_start, lats_deg, syn_map, dates = build_synoptic_map(data_dir, lat_points)

    # ---- sort by latitude, then time (ensures monotonic axes) ----
    lat_sort_idx      = np.argsort(lats_deg)
    lats_deg_sorted   = np.asarray(lats_deg)[lat_sort_idx]
    syn_map_sorted_lat = syn_map[:, lat_sort_idx]                      # (Nt, Nlat_sorted)

    time_sort_idx = np.argsort(days_since_start)
    days_sorted   = np.asarray(days_since_start)[time_sort_idx]
    syn_map_sorted = syn_map_sorted_lat[time_sort_idx, :]              # (Nt_sorted, Nlat_sorted)

    # 2) Resample to a uniform latitude grid you’ll use everywhere
    model_lats = np.linspace(-90.0, 90.0, int(lat_points))
    syn_map_uniform = np.empty((len(days_sorted), len(model_lats)), dtype=float)
    for k, row in enumerate(syn_map_sorted):
        f = interp1d(
            lats_deg_sorted, row,
            kind="cubic", bounds_error=False, fill_value="extrapolate"
        )
        syn_map_uniform[k, :] = f(model_lats)

    # 3) REMOVE MONOPOLE (area-weighted), per time step, on the uniform grid
    # For a surface integral, dμ is uniform where μ = sin(λ), so weight ∝ dμ/dλ = cos(λ).
    lat_rad = np.deg2rad(model_lats)
    w = np.cos(lat_rad)                         # shape (Nlat,)
    w = np.clip(w, 0.0, None)                  # guard rare numerical negatives near poles
    w /= w.sum()                                # normalize weights

    # Weighted mean over latitude for each time slice
    # mean_k = sum_j B[k,j] * w[j]
    means = syn_map_uniform @ w[:, None]        # (Nt,1)
    B_bal = syn_map_uniform - means             # remove monopole bias per time

    # 4) SCALE the source with the single knob
    scale = 1.0 if (config is None or not hasattr(config, "source_scale")) else float(config.source_scale)
    B_bal_scaled = scale * B_bal

    # 5) Choose time unit for the spline (years is convenient & stable)
    t_years = np.asarray(days_sorted, dtype=float) / 365.25

    # 6) Build a lat × time spline (we’ll query as f(lat_deg, t_years))
    # RectBivariateSpline expects x increasing; we pass (lat, time) then transpose data.
    synoptic_source_interp = RectBivariateSpline(
        model_lats, t_years, B_bal_scaled.T, kx=3, ky=3
    )

    # 7) Provide IC profile from the first time slice (already balanced+scaled)
    initial_lats = model_lats
    initial_profile = B_bal_scaled[0].copy()

    print(f"Synoptic source built: Nt={len(t_years)}, Nlat={len(model_lats)}, "
          f"scale={scale:.3f}, mean|B|={np.mean(np.abs(B_bal_scaled)):.3f} G")

    if return_arrays:
        return t_years, model_lats, B_bal_scaled
    return synoptic_source_interp, initial_lats, initial_profile

def get_wso_constraints(Tmax, lat_points, time_steps, B_unit, data_dir="data/22"):
    """
    Interpolates all WSO maps in a directory and returns (obs_X, obs_Y) for supervised PINN training.

    Args:
        Tmax (float): total normalized simulation time (usually 1.0)
        lat_points (int): number of latitude points in the PINN
        time_steps (int): number of time steps in the PINN
        B_unit (float): normalization for magnetic field
        data_dir (str): directory containing WSO*.F.txt files

    Returns:
        obs_X (Nx2), obs_Y (Nx1)
    """
    # Discover all WSO map files
    file_paths = sorted(glob.glob(os.path.join(data_dir, "WSO.*.F.txt")))
    if not file_paths:
        raise FileNotFoundError(f"No WSO files found in {data_dir}")

    sinlats = np.linspace(14.5 / 15, -14.5 / 15, 30)
    lats_deg = np.arcsin(sinlats) * 180 / np.pi  # Convert to degrees
    model_lats_deg = np.linspace(-90, 90, lat_points)

    obs_X = []
    obs_Y = []

    total_maps = 0

    for path in file_paths:
        data_rows, ct_rows = load_full_wso_data(path)
        num_times = len(data_rows)
        total_maps += num_times

        # Interpolate to model latitude resolution
        interp_maps = []
        for row in data_rows:
            f_interp = interp1d(lats_deg, row, kind='cubic', bounds_error=False, fill_value='extrapolate')
            interp_maps.append(f_interp(model_lats_deg))
        interp_maps = np.array(interp_maps)
        #global initial_profile
        #initial_profile = interp_maps[0]

        # Normalized time for this file (within its own block)
        normalized_t = np.linspace(0, Tmax, num_times)

        # Append data
        for i, t_norm in enumerate(normalized_t):
            for j, lat_deg in enumerate(model_lats_deg):
                if abs(lat_deg) <= 60:
                    lam_norm = (lat_deg /180)  # normalize latitude to [0, 1]
                    obs_X.append([lam_norm, t_norm])
                    obs_Y.append(interp_maps[i, j] / B_unit)

    obs_X = np.array(obs_X)
    obs_Y = np.array(obs_Y).reshape(-1, 1)

    print(f"Loaded {len(file_paths)} WSO files, total time slices: {total_maps}, total points: {len(obs_X)}")
    return obs_X, obs_Y


# === INITIAL PROFILE SETUP ===

# Load one representative WSO file (first one in sorted list)
import glob
import os

def build_initial_profile(B_unit, data_dir="data/24"):
    file_paths = sorted(glob.glob(os.path.join(data_dir, "WSO.*.F.txt")))
    if not file_paths:
        raise FileNotFoundError(f"No WSO files found in {data_dir}")
    #print(file_paths[0])
    from src.extract import load_full_wso_data  # Safe even within same file
    data_rows, ct_rows = load_full_wso_data(file_paths[0])  # Just the first map

    sinlats = np.linspace(14.5 / 15, -14.5 / 15, 30)
    lats_deg = np.arcsin(sinlats) * 180 / np.pi
    model_lats_deg = np.linspace(-90, 90, 360)  # match model grid

    f_interp = interp1d(lats_deg, data_rows[0], kind='cubic', bounds_error=False, fill_value='extrapolate')
    profile = f_interp(model_lats_deg) / B_unit  # normalize

    return profile

# Set the variable here
initial_profile = build_initial_profile(B_unit=10.0, data_dir="data/24")



def plot_synoptic_map(data_dir="data/24", lat_points=360, save_path=None):
    # Get time, latitude, synoptic map, and dates
    days_since_start, lats_deg, syn_map, dates = build_synoptic_map(data_dir, lat_points)

    # Sort latitudes to be ascending for plotting
    lat_sort_idx = np.argsort(lats_deg)
    lats_deg_sorted = lats_deg[lat_sort_idx]
    syn_map_sorted = syn_map[:, lat_sort_idx]

    # Convert days to fractional years for smoother plotting
    years_float = np.array([d.year + d.timetuple().tm_yday / 365.25 for d in dates])

    plt.figure(figsize=(12, 6))
    plt.contourf(years_float, lats_deg_sorted, syn_map_sorted.T, levels=100, cmap="RdBu_r")
    cycle_label = os.path.basename(data_dir)  # e.g., "23" from "data/23"
    plt.title(f"WSO Synoptic Map - Solar Cycle {cycle_label}")
    plt.xlabel("Year")
    plt.ylabel("Latitude [°]")
    plt.colorbar(label="Magnetic Field [G]")

    # Set yearly ticks on x-axis
    plt.xticks(
        np.arange(int(years_float[0]), int(years_float[-1]) + 1, 1),
        rotation=45
    )

    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Synoptic map saved to {save_path}")
