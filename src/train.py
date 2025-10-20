# train.py
import numpy as np
import deepxde as dde
import tensorflow as tf
from . import sft_pde
from .sft_pde import adv, init, boundary, make_pde
from .utils import set_random_seed
from .extract import get_wso_constraints
from .extract import build_synoptic_source
from deepxde.icbc import PointSetBC
from scipy.interpolate import LinearNDInterpolator

def train_model(config):
    set_random_seed(42)

    # Build balanced + scaled WSO source on the model grid  <<< ADDED
    build_synoptic_source(
        config.wso_path,
        lat_points=config.num_lats,
        config=config,
    )

    # Geometry and domain
    geom = dde.geometry.Interval(config.lam_min, config.lam_max)   # [-0.495, 0.495]
    timedomain = dde.geometry.TimeDomain(0.0, config.Tmax)         # [0, 1]
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Network
    net = dde.nn.FNN(config.layer_sizes, config.activation, config.initializer)

    # Initial condition
    ic = dde.icbc.IC(geomtime, init, lambda _, on_initial: on_initial)

    # Polar Neumann(0): dB/dθ = 0 at θ = ±90° (latitude edges only)
    def on_theta_boundary(x, on_boundary):
        if not on_boundary:
            return False
        return np.isclose(x[0], config.lam_min) or np.isclose(x[0], config.lam_max)

    bc_pole_neumann = dde.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0),  # ∂B/∂(latitude_input) = 0
        on_theta_boundary,
    )

    # PDE
    pde_fn = make_pde(config)

    # Conditions
    conditions = [ic, bc_pole_neumann]

    # WSO point constraints  <<< UPDATED
    if config.use_wso:
        obs_X, obs_Y = get_wso_constraints(
            Tmax=config.Tmax,
            lat_points=config.num_lats,           # was num_time_points
            time_steps=config.num_time_points,
            B_unit=config.B_unit,
            data_dir=config.wso_path              # was "data/24"
        )
        interp_B_obs = LinearNDInterpolator(obs_X, obs_Y.ravel())
        obs_bc = PointSetBC(obs_X, obs_Y, component=0)
        conditions.append(obs_bc)

    # Data
    data = dde.data.TimePDE(
        geomtime,
        pde_fn,
        conditions,
        num_test=config.num_test,
        num_domain=config.num_domain,
        num_boundary=config.num_boundary,
        num_initial=config.num_initial,
    )

    # Training
    model = dde.Model(data, net)
    model.compile("adam", lr=config.lr, loss_weights=config.loss_weights)
    model.train(iterations=config.iter_adam, display_every=1000)

    model.compile("L-BFGS", loss_weights=config.loss_weights_lbfgs)
    losshistory, train_state = model.train(display_every=1000)

    return model, train_state
