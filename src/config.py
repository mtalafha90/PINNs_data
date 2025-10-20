# config.py
import numpy as np
import tensorflow as tf
from src import sft_pde  # Needed to set random_array_tf

class Config:
    def __init__(self, mode="fast", output_dir="results/Linear_1", quenching_type="none"):
        self.output_dir = output_dir
        self.quenching_type = quenching_type  # "none", "TQ", "LQ", "TQ+LQ"

        # Physical units
        self.simul_time = 11.0  # years
        self.B_unit = 10.0
        self.L_unit = 6.95e10  # Solar radius in cm
        self.T_unit = self.simul_time * 365.25 * 24 * 3600

        # Latitude/time domain
        self.lam_min = -0.495
        self.lam_max = 0.495
        self.lat_min = -90.0
        self.lat_max = 90.0
        self.Tmax = 1.0

        # Network
        self.layer_sizes = [2] + [41] * 10 + [1]
        self.activation = "tanh"
        self.initializer = "Glorot uniform"

        # Training config
        if mode == "fast":
            self.iter_adam = 1000
            self.num_domain = 2000
            self.num_boundary = 200
            self.num_initial = 200
            self.num_test = 500
            self.use_wso = True  # Enable or disable WSO constraint
        else:  # full
            self.iter_adam = 95000 # changed from 95000
            self.num_domain = 87460
            self.num_boundary = 2356
            self.num_initial = 2787
            self.num_test = 1000
            self.use_wso = True  # Enable or disable WSO constraint

        self.lr = 0.0022
        self.loss_weights = [3233, 48, 4975]#, 2000]
        self.loss_weights_lbfgs = [20, 1, 10]#, 5]

        self.source_scale = 0.95  # try 0.7–1.2 to tune reversals

        # Resolution
        self.num_time_points = 400
        self.num_lats = 181  # Number of latitudinal grid points for model and WSO interpolation
        self.wso_path = "data/24"  # or full path if needed

        # Random per-cycle amplitude variations
        self.num_cycles = int(self.simul_time / 11) + 2
        np.random.seed(42)
        random_array = np.random.normal(loc=0.0, scale=0.13, size=self.num_cycles)
        sft_pde.random_array_tf = tf.convert_to_tensor(random_array, dtype=tf.float32)
