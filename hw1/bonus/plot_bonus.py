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

# ===== LOAD MODEL =====
ckptpath = os.path.join(_OUTPUTS_DIR, 'cbf_bonus.ckpt')
neural_controller = NeuralCBFController.load_from_checkpoint(ckptpath)

h_fn = lambda x: -neural_controller.V_with_jacobian(x)[0]
dhdx_fn = lambda x: -neural_controller.V_with_jacobian(x)[1].squeeze(1)

# ===== GRID =====
theta = torch.linspace(-0.4, 0.4, 200)
thetadot = torch.linspace(-2, 2, 200)

TH, TD = torch.meshgrid(theta, thetadot, indexing='ij')

X = torch.stack([
    TH.reshape(-1),
    TD.reshape(-1)
], dim=1)

h_old_vals = h_old(X).reshape(TH.shape)
h_new_vals = h_fn(X).reshape(TH.shape).detach()

# ===== PLOT CBF =====
fig, ax = plt.subplots()
ax.set_title('Analytical vs Refined CBF')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\dot{\theta}$')

ax.contour(TH, TD, h_old_vals, levels=[0], colors='blue')
ax.contour(TH, TD, h_new_vals, levels=[0], colors='red')
ax.legend(['Old CBF', 'Refined CBF'])

# ===== VOLUME =====
old_area = (h_old_vals >= 0).float().mean()
new_area = (h_new_vals >= 0).float().mean()
improvement = (new_area - old_area) / old_area * 100

print(f'Volume improvement (%): {improvement.item():.2f}')

# ===== SAVE =====
plt.savefig(os.path.join(_OUTPUTS_DIR, 'plot_bonus.png'))
plt.close()

# ===== DYNAMICS (numpy for simulation) =====
def f_np(x):
    return np.array([x[1], 10*np.sin(x[0])])

def g_np():
    return np.array([0, 1/(2*1**2)])

def step(x, u, dt=0.01):
    return x + (f_np(x) + g_np()*u)*dt

# ===== NEURAL CBF-QP =====
def cbf_qp_neural(x):
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    h_val = h_fn(xt)[0].item()
    dh = dhdx_fn(xt)[0].detach().numpy()

    Lf = dh @ f_np(x)
    Lg = dh @ g_np()

    def objective(u):
        return (u[0])**2

    gamma = 1.0
    def constraint(u):
        return Lf + Lg*u[0] + gamma * h_val

    res = minimize(objective, [0.0],
                   bounds=[(-3, 3)],
                   constraints={'type': 'ineq', 'fun': constraint})

    return res.x[0] if res.success else 0.0

# ===== SIMULATION =====
def simulate(x0):
    x = np.array(x0)
    traj = [x.copy()]
    u_hist = []
    t_hist = []

    t = 0
    while t < 5:
        u = cbf_qp_neural(x)
        x = step(x, u)

        traj.append(x.copy())
        u_hist.append(u)
        t_hist.append(t)

        t += 0.01

    return np.array(traj), np.array(u_hist), np.array(t_hist)

# ===== RUN =====
x0s = [(0,0), (0.05,0.05)]
results = [simulate(x0) for x0 in x0s]

# ===== TRAJECTORY PLOT =====
plt.figure()
for i, (traj, _, _) in enumerate(results):
    plt.plot(traj[:,0], traj[:,1], label=f"x0={x0s[i]}")

plt.xlabel("theta")
plt.ylabel("theta_dot")
plt.title("Neural CBF-QP trajectories")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(_OUTPUTS_DIR, 'traj_bonus.png'))
plt.close()

# ===== CONTROL PLOT =====
plt.figure()
for i, (_, u_hist, t_hist) in enumerate(results):
    plt.plot(t_hist, u_hist, label=f"x0={x0s[i]}")

plt.xlabel("Time")
plt.ylabel("u(t)")
plt.title("Control inputs")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(_OUTPUTS_DIR, 'control_bonus.png'))
plt.close()