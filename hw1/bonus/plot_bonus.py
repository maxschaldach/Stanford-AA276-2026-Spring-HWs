import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from neural_clbf.controllers import NeuralCBFController

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pendulum_system import h_old

_OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')

# ===== PATCH: recreate class for checkpoint loading =====
from neural_clbf.systems import ControlAffineSystem
import torch

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
        return []

    def validate_params(self, params):
        return True

    def compute_linearized_controller(self, scenarios):
        return

    def _f(self, x, params=None):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        theta = x[:, 0]
        theta_dot = x[:, 1]

        out = torch.zeros((x.shape[0], 2, 1), device=x.device)
        out[:, 0, 0] = theta_dot
        out[:, 1, 0] = (g_const / l) * torch.sin(theta)
        return out

    def _g(self, x, params=None):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        B = x.shape[0]
        G = torch.zeros((B, 2, 1), device=x.device)
        G[:, 1, 0] = 1 / (m * l**2)
        return G

    @property
    def state_limits(self):
        return torch.tensor([0.4, 2.0]), torch.tensor([-0.4, -2.0])

    @property
    def control_limits(self):
        return torch.tensor([3.0]), torch.tensor([-3.0])

    def safe_mask(self, x):
        return torch.abs(x[:, 0]) <= 0.3

    def unsafe_mask(self, x):
        return torch.abs(x[:, 0]) > 0.3

    def u_nominal(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        theta = x[:, 0]
        theta_dot = x[:, 1]

        u = m * l**2 * (
            -(g_const / l) * torch.sin(theta)
            - 1.5 * theta
            - 1.5 * theta_dot
        )

        return torch.clamp(u, -3.0, 3.0).unsqueeze(1)

# ===== LOAD MODEL =====
ckptpath = os.path.join(_OUTPUTS_DIR, 'cbf_bonus.ckpt')
neural_controller = NeuralCBFController.load_from_checkpoint(ckptpath)

h_new = lambda x: -neural_controller.V_with_jacobian(x)[0]
dhdx_new = lambda x: -neural_controller.V_with_jacobian(x)[1].squeeze(1)

# ===== ANALYTICAL GRADIENT =====
a = 0.14
b = np.sqrt(a * (3 - 20*np.sin(a)) / 2)

def grad_h_old(x):
    theta, theta_dot = x
    return np.array([
        -2*theta/(a**2),
        -2*theta_dot/(b**2)
    ])

# ===== GRID =====
theta = torch.linspace(-0.4, 0.4, 200)
thetadot = torch.linspace(-2, 2, 200)

TH, TD = torch.meshgrid(theta, thetadot, indexing='ij')

X = torch.stack([
    TH.reshape(-1),
    TD.reshape(-1)
], dim=1)

h_old_vals = h_old(X).reshape(TH.shape)
h_new_vals = h_new(X).reshape(TH.shape).detach()

# ===== CBF COMPARISON PLOT =====
fig, ax = plt.subplots()
ax.set_title('Analytical vs Refined CBF')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\dot{\theta}$')

# contours
c1 = ax.contour(TH, TD, h_old_vals, levels=[0], colors='blue')
c2 = ax.contour(TH, TD, h_new_vals, levels=[0], colors='red')

# failure set
ax.fill_betweenx([-2, 2], 0.3, 0.4, color='red', alpha=0.15)
ax.fill_betweenx([-2, 2], -0.4, -0.3, color='red', alpha=0.15)

ax.legend(['Old CBF', 'Refined CBF'])

# ===== VOLUME IMPROVEMENT =====
old_area = (h_old_vals >= 0).float().mean()
new_area = (h_new_vals >= 0).float().mean()
improvement = (new_area - old_area) / old_area * 100

print(f'Volume improvement (%): {improvement.item():.2f}')

plt.savefig(os.path.join(_OUTPUTS_DIR, 'plot_bonus.png'))
plt.close()

# ===== DYNAMICS =====
def f_np(x):
    return np.array([x[1], 10*np.sin(x[0])])

def g_np():
    return np.array([0, 1/(2*1**2)])

def step(x, u, dt=0.01):
    return x + (f_np(x) + g_np()*u)*dt

# ===== TRUE NOMINAL CONTROLLER =====
def u_nominal(x, t):
    if t < 1:
        return 3.0
    elif t < 2:
        return -3.0
    elif t < 3:
        return 3.0
    else:
        return 2.0 * (
            -10 * np.sin(x[0]) - np.array([1.5,1.5]) @ x
        )

# ===== CBF-QP (NEURAL) =====
def cbf_qp_neural(x, t):
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    h_val = h_new(xt)[0].item()
    dh = dhdx_new(xt)[0].detach().numpy()

    Lf = dh @ f_np(x)
    Lg = dh @ g_np()

    u_ref = u_nominal(x, t)

    def objective(u):
        return (u[0] - u_ref)**2

    gamma = 1.0
    def constraint(u):
        return Lf + Lg*u[0] + gamma * h_val

    res = minimize(objective, [u_ref],
                   bounds=[(-3, 3)],
                   constraints={'type': 'ineq', 'fun': constraint})

    return res.x[0] if res.success else u_ref

# ===== CBF-QP (OLD) =====
def cbf_qp_old(x, t):
    h_val = h_old(torch.tensor(x).unsqueeze(0))[0].item()
    dh = grad_h_old(x)

    Lf = dh @ f_np(x)
    Lg = dh @ g_np()

    u_ref = u_nominal(x, t)

    def objective(u):
        return (u[0] - u_ref)**2

    gamma = 1.0
    def constraint(u):
        return Lf + Lg*u[0] + gamma * h_val

    res = minimize(objective, [u_ref],
                   bounds=[(-3, 3)],
                   constraints={'type': 'ineq', 'fun': constraint})

    return res.x[0] if res.success else u_ref

# ===== SIMULATION =====
def simulate(x0, controller):
    x = np.array(x0)
    traj = [x.copy()]
    u_hist = []
    t_hist = []

    t = 0
    while t < 5:
        u = controller(x, t)
        x = step(x, u)

        traj.append(x.copy())
        u_hist.append(u)
        t_hist.append(t)

        t += 0.01

    return np.array(traj), np.array(u_hist), np.array(t_hist)

# ===== RUN =====
x0s = [(0,0), (0.05,0.05)]

results_new = [simulate(x0, cbf_qp_neural) for x0 in x0s]
results_old = [simulate(x0, cbf_qp_old) for x0 in x0s]

# ===== TRAJECTORY PLOT =====
plt.figure()

for i, (traj, _, _) in enumerate(results_old):
    plt.plot(traj[:,0], traj[:,1], '--', label=f"old x0={x0s[i]}")

for i, (traj, _, _) in enumerate(results_new):
    plt.plot(traj[:,0], traj[:,1], label=f"new x0={x0s[i]}")

# failure set
plt.fill_betweenx([-2, 2], 0.3, 0.4, color='red', alpha=0.15)
plt.fill_betweenx([-2, 2], -0.4, -0.3, color='red', alpha=0.15)

plt.xlabel("theta")
plt.ylabel("theta_dot")
plt.title("CBF-QP trajectories (old vs refined)")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(_OUTPUTS_DIR, 'traj_bonus.png'))
plt.close()

# ===== CONTROL PLOT =====
plt.figure()

for i, (_, u_hist, t_hist) in enumerate(results_old):
    plt.plot(t_hist, u_hist, '--', label=f"old x0={x0s[i]}")

for i, (_, u_hist, t_hist) in enumerate(results_new):
    plt.plot(t_hist, u_hist, label=f"new x0={x0s[i]}")

plt.xlabel("Time")
plt.ylabel("u(t)")
plt.title("Control inputs (old vs refined)")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(_OUTPUTS_DIR, 'control_bonus.png'))
plt.close()