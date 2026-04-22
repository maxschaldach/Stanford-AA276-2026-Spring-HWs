import torch
import matplotlib.pyplot as plt
from neural_clbf.controllers import NeuralCBFController

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bonus_part3 import plot_h, plot_and_eval_xts
from bonus_part1 import state_limits

state_max, state_min = state_limits()

ckptpath = os.path.join(os.path.dirname(__file__), 'bonus_cbf.ckpt')
neural_controller = NeuralCBFController.load_from_checkpoint(ckptpath)

fig, ax = plt.subplots()
ax.set_title(r'$h(\theta, \dot{\theta})$')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\dot{\theta}$')

# grid for visualization
theta = torch.linspace(-0.4, 0.4, 100)
theta_dot = torch.linspace(-2, 2, 100)

h_fn = lambda x: -neural_controller.V_with_jacobian(x)[0]
dhdx_fn = lambda x: -neural_controller.V_with_jacobian(x)[1].squeeze(1)

# sample initial states
x0 = torch.rand(100, 2) * (state_max - state_min) + state_min

# nominal control (zero is fine)
def u_ref_fn(x):
    return torch.zeros((len(x), 1))

gamma = 1
lmbda = 1e9
nt = 50
dt = 0.01

print('running plot_h...')
plot_h(fig, ax, theta, theta_dot, None, h_fn)

print('running plot_and_eval_xts...')
false_safety_rate = plot_and_eval_xts(
    fig, ax, x0, u_ref_fn, h_fn, dhdx_fn, gamma, lmbda, nt, dt
)

_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bonus_plot.png')
plt.savefig(_plot_path)
plt.close()

print(f'plot saved to {_plot_path}')
print(f'false safety rate: {false_safety_rate}')