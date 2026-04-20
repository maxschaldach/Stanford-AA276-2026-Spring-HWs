from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite

torch.multiprocessing.set_sharing_strategy('file_system')

controller_period = 0.05
simulation_dt = 0.01

parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
args.gpus = 1

# ===== SCENARIOS =====
nominal_params = {'g': 10.0}
scenarios = [nominal_params]

# ===== IMPORT PENDULUM =====
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pendulum_system import (
    state_limits,
    control_limits,
    safe_mask,
    failure_mask,
    f,
    g,
)

from neural_clbf.systems import ControlAffineSystem

# ===== FIXED SYSTEM CLASS =====
class PendulumSystem(ControlAffineSystem):
    def __init__(self, nominal_params):
        super().__init__(nominal_params)

    @property
    def n_dims(self):
        return 2

    @property
    def n_controls(self):
        return 1

    @property
    def angle_dims(self):
        return []

    def validate_params(self, params):
        return True

    # CRITICAL FIX
    def compute_linearized_controller(self, scenarios):
        return

    def _f(self, x, params=None):
        return f(x)

    def _g(self, x, params=None):
        return g(x)

    @property
    def state_limits(self):
        return state_limits()

    @property
    def control_limits(self):
        return control_limits()

    def safe_mask(self, x):
        return safe_mask(x)

    def unsafe_mask(self, x):
        return failure_mask(x)

# ===== FIXED INSTANTIATION =====
dynamics_model = PendulumSystem(nominal_params)

# ===== DATA =====
initial_conditions = [
    (-0.4, 0.4),
    (-2.0, 2.0),
]

data_module = EpisodicDataModule(
    dynamics_model,
    initial_conditions,
    trajectories_per_episode=0,
    trajectory_length=1,
    fixed_samples=100000,
    max_points=300000000,
    val_split=0.01,
    batch_size=1024,
)

experiment_suite = ExperimentSuite([])

# ===== CONTROLLER =====
cbf_controller = NeuralCBFController(
    dynamics_model,
    scenarios,
    data_module,
    experiment_suite=experiment_suite,
    cbf_hidden_layers=2,
    cbf_hidden_size=128,
    cbf_lambda=0.3,
    cbf_relaxation_penalty=1e3,
    controller_period=controller_period,
    primal_learning_rate=1e-4,
    scale_parameter=1.0,
    learn_shape_epochs=1,
    use_relu=True,
    disable_gurobi=True,
)

# ===== TRAINER =====
_OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')

tb_logger = pl_loggers.TensorBoardLogger(
    _OUTPUTS_DIR,
    name='',
)

trainer = pl.Trainer.from_argparse_args(
    args,
    logger=tb_logger,
    reload_dataloaders_every_epoch=True,
    max_epochs=80,
)

# ===== TRAIN =====
torch.autograd.set_detect_anomaly(True)
trainer.fit(cbf_controller)