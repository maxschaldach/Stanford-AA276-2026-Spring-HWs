import torch
from neural_clbf.systems import ControlAffineSystem
from bonus_part1 import f, g, state_limits, control_limits, safe_mask, failure_mask

m = 2.0
l = 1.0
g_const = 10.0


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
        return []   # no periodic handling

    def validate_params(self, params):
        return True

    # REQUIRED (even if unused)
    def compute_linearized_controller(self, scenarios):
        return

    # dynamics
    def _f(self, x, params=None):
        return f(x)

    def _g(self, x, params=None):
        return g(x)

    # limits
    @property
    def state_limits(self):
        return state_limits()

    @property
    def control_limits(self):
        return control_limits()

    # sets
    def safe_mask(self, x):
        return safe_mask(x)

    def unsafe_mask(self, x):
        return failure_mask(x)

    # nominal controller (used during training)
    def u_nominal(self, x):
        theta = x[:, 0]
        theta_dot = x[:, 1]

        u = m * l**2 * (
            -(g_const / l) * torch.sin(theta)
            - 1.5 * theta
            - 1.5 * theta_dot
        )

        return torch.clamp(u, -3.0, 3.0).unsqueeze(1)