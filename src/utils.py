# utils.py
import numpy as np
import tensorflow as tf
import random
import src.sft_pde as sft_pde  # Make sure sft_pde is importable
def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def compute_lat_grid(config):
    res_lat = 181
    dlat = (config.lat_max - config.lat_min) / res_lat
    lat_cen = np.array([(config.lat_min + (i + 0.5) * dlat) * np.pi / 180.0 for i in range(res_lat)])
    
    # Add polar mask for latitudes ≥ 60° in magnitude
    lat_deg = lat_cen * 180.0 / np.pi
    polar_mask = np.abs(lat_deg) >= 60.0

    return {
        "lat_cen": lat_cen,
        "polar_mask": polar_mask
    }

def compute_dipole_moment(B, lat_deg, config):
    """
    Parameters
    ----------
    B : (Nt, Nlat) array
        Longitudinally-averaged surface field in *Gauss*.
    lat_deg : (Nlat,) array
        Latitude centers in degrees (-90..+90).
    config : object with attributes
        - L_unit : solar radius in cm (e.g., 6.96e10)
        - (optional) B_unit : if your B is NOT yet in Gauss, multiply by B_unit first.

    Returns
    -------
    M_22 : (Nt,) array
        Axial dipole moment in units of 1e22 Mx·cm.
    """
    import numpy as np

    # Convert latitude to mu = cos(theta) = sin(latitude)
    lat_rad = np.deg2rad(lat_deg)
    mu = np.sin(lat_rad)  # shape (Nlat,)

    # Ensure B is in Gauss; if your B is normalized, scale it here:
    # B_phys = B * config.B_unit
    B_phys = B

    # a1(t) = (3/2) ∫ B(mu,t) * mu dmu    (trapz over mu axis)
    a1 = 1.5 * np.trapz(B_phys * mu[None, :], mu, axis=1)  # Gauss

    # M(t) = (R^3/2) * a1  => combine: (3/4) * R^3 * ∫ B mu dmu
    M = 0.5 * (config.L_unit ** 3) * a1  # G*cm^3 = Mx*cm

    # Return in 1e22 Mx·cm
    M_22 = M / 1e22
    return M_22


def compute_amplitudes_from_gaussians(gaussian_array, config):
    """
    Computes the maximum amplitude used in the source function for each cycle
    based on Gaussian fluctuations.

    Parameters:
        gaussian_array: 1D array of gaussian fluctuations per cycle (log10 space).
        config: Config object with time unit info.

    Returns:
        List of max amplitudes (one per cycle).
    """
    amps = []
    for g in gaussian_array:
        tc = np.linspace(0.01, 12.0, 500)  # in years (12-month cycles)
        sourcescale1 = sft_pde.sourcescales * np.exp(
            7.0 * 365.25 * 24.0 * 3600.0 / (sft_pde.tau * sft_pde.T_unit)
        )
        sourcescale = sourcescale1 * 10 ** g
        ampli = sourcescale * sft_pde.ahat * tc**3 / (
            np.exp(tc**2 / sft_pde.bhat**2) - sft_pde.chat
        )
        amps.append(np.max(ampli))  # or np.mean(ampli), if desired
    return amps
