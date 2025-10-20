# sft_pde.py
import numpy as np
import deepxde as dde
from deepxde.backend import tf
#from src.train import train_model
from src.extract import initial_profile
from src.extract import synoptic_source_interp


simul_time = 11.0  # 25 solar cycles
cycleper = 11.0 * 365.25
eta = 350e10
Rsun = 6.95e10
L_unit = Rsun
T_unit = simul_time * 365.25 * 24.0 * 3600.0
V_unit = L_unit / T_unit
eta = eta / (L_unit * V_unit)
tau = 8.0 * 365.25 * 24.0 * 3600.0 / T_unit
u_0 = -11.0e2 / V_unit
lam00 = tf.constant(90.0 * np.pi / 180.0)
B0 = 10.0 / 10.0

ahat = 0.00185
bhat = 48.7
chat = 0.71
joynorm0 = 1.5
sourcescales = 0.0015 * 1000 / B0

# Global random Gaussian fluctuations per cycle (to be set externally)
random_array_tf = tf.constant(np.zeros(32), dtype=tf.float32)
bjoy = 0.15
blat = 2.4

# Amplitude tracking for plotting
ampl_used_by_time = []


def adv(lam):
    u = u_0 * tf.sin(lam / lam00)
    return tf.where(tf.abs(lam) < lam00, u, tf.zeros_like(u))

'''
def source_from_wso(x):
    """
    TensorFlow wrapper to interpolate synoptic map.
    x: shape (N, 2) where x[:, 0] is latitude (normalized) and x[:, 1] is time (normalized)
    """
    def numpy_interp(x_numpy):
        lam = x_numpy[:, 0] * 180.0 - 90.0  # convert normalized to degrees
        t_days = x_numpy[:, 1] * simul_time * 365.25
        result = np.array([synoptic_source_interp(lat, t, grid=False) for lat, t in zip(lam, t_days)])
        return result.reshape(-1, 1)

    return tf.py_function(func=numpy_interp, inp=[x], Tout=tf.float32)
'''
def source_from_wso(x):
    """
    x: (N,2) with x[:,0]=λ_norm in [-0.5..0.5] (your normalized latitude), x[:,1]=t_norm in [0..1]
    Returns: (N,1) Gauss, masked to active latitudes (|lat| <= 45°).
    """
    def numpy_interp(x_numpy):
        lam_deg = x_numpy[:, 0] * 180.0                       # [-90..+90] deg
        t_years = x_numpy[:, 1] * simul_time                   # [0..simul_time] years

        # synoptic_source_interp expects (lat_deg, t_years)
        vals = np.array([
            synoptic_source_interp(ld, ty, grid=False)
            for ld, ty in zip(lam_deg, t_years)
        ]).reshape(-1, 1)

        belt = (np.abs(lam_deg) <= 45.0).reshape(-1, 1)        # inject only in AR belt
        return np.where(belt, vals, 0.0).astype(np.float32)

    return tf.py_function(func=numpy_interp, inp=[x], Tout=tf.float32)


def make_pde(config):
    def pde_SFT(x, y):
        dB_t = dde.grad.jacobian(y, x, j=1)
        dB1_lam = dde.grad.jacobian(y * (adv(np.pi * x[:, 0:1]) - eta * tf.tan(np.pi * x[:, 0:1])), x, j=0) / np.pi
        dB2_lam = dde.grad.hessian(y, x, i=0, j=0) / (np.pi ** 2)

        # Interpolated WSO field as 'source'
        lam = x[:, 0:1] * 180 - 89  # rescale back to degrees
        t_norm = x[:, 1:2]
        B_obs_vals = tf.numpy_function(interp_B_obs, [x], tf.float64)  # shape: (N, 1)
        B_obs_vals = tf.cast(B_obs_vals, tf.float32)
        sou = source_from_wso(x)*sourcescales
        S = dB_t - dB1_lam - eta * dB2_lam + y * (1 / tau + adv(np.pi * x[:, 0:1]) * tf.tan(x[:, 0:1] * np.pi) - eta * (1 / tf.cos(x[:, 0:1] * np.pi)) ** 2) - B / tau - sou  # replace synthetic source with observed data
        return S



# Build latitude axis matching initial_profile
initial_lats = np.linspace(-90, 90, len(initial_profile))

def init(x):
    # x[:, 0] is the normalized latitude in [lam_min, lam_max] ~ [-0.495, 0.495]
    lam = x[:, 0]
    lat_deg = lam * 180.0  # Convert normalized λ to degrees

    # If lat_deg is a Tensor, convert to numpy for interpolation
    if tf.is_tensor(lat_deg):
        lat_deg = lat_deg.numpy()

    values = np.interp(lat_deg, initial_lats, initial_profile)

    return tf.convert_to_tensor(values.reshape(-1, 1), dtype=tf.float32)

def boundary(x, on_boundary):
    return on_boundary
