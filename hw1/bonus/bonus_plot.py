import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_clbf.controllers import NeuralCBFController
import os

from bonus_part1 import state_limits, safe_mask, f, g
from bonus_part3 import plot_and_eval_xts
from bonus_part2 import u_qp

# =========================
# LOAD MODEL
# =========================
ckptpath = os.path.join(os.path.dirname(__file__), 'bonus_cbf.ckpt')
neural_controller = NeuralCBFController.load_from_checkpoint(ckptpath)

h_new = lambda x: -neural_controller.V_with_jacobian(x)[0]
dhdx_new = lambda x: -neural_controller.V_with_jacobian(x)[1].squeeze(1)

# =========================
# OLD CBF (analytical)
# =========================
a = 0.14
b = np.sqrt(a * (3 - 20*np.sin(a)) / 2)

def h_old(x):
    theta = x[:, 0]
    theta_dot = x[:, 1]
    return 1 - (theta**2)/(a**2) - (theta_dot**2)/(b**2)

# =========================
# GRID
# =========================
theta = torch.linspace(-0.4, 0.4, 200)
theta_dot = torch.linspace(-2, 2, 200)

TH, TD = torch.meshgrid(theta, theta_dot)

X = torch.stack([TH.reshape(-1), TD.reshape(-1)], dim=1)

h_old_vals = h_old(X).reshape(TH.shape)
h_new_vals = h_new(X).reshape(TH.shape).detach()

# =========================
# PLOT 1: CBF COMPARISON
# =========================
fig1, ax1 = plt.subplots()

ax1.set_title("Old vs Learned CBF")
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"$\dot{\theta}$")

ax1.contour(TH, TD, h_old_vals, levels=[0], colors='blue')
ax1.contour(TH, TD, h_new_vals, levels=[0], colors='red')

# failure set
ax1.fill_betweenx([-2, 2], 0.3, 0.4, color='red', alpha=0.1)
ax1.fill_betweenx([-2, 2], -0.4, -0.3, color='red', alpha=0.1)

# volume estimate
old_area = (h_old_vals >= 0).float().mean()
new_area = (h_new_vals >= 0).float().mean()
improvement = (new_area - old_area) / old_area * 100

print(f"Volume improvement (%): {improvement.item():.2f}")

fig1.savefig("cbf_comparison.png")
plt.close(fig1)

# =========================
# INITIAL CONDITIONS (SAFE ONLY)
# =========================
state_max, state_min = state_limits()

x0 = torch.rand(1000, 2) * (state_max - state_min) + state_min
x0 = x0[safe_mask(x0)]  # only safe states

# =========================
# NOMINAL CONTROL (Problem 3)
# =========================
def u_nominal(x, t):
    if t < 1:
        return torch.full((len(x), 1), 3.0)
    elif t < 2:
        return torch.full((len(x), 1), -3.0)
    elif t < 3:
        return torch.full((len(x), 1), 3.0)
    else:
        theta = x[:, 0]
        theta_dot = x[:, 1]
        u = 2.0 * (-10*torch.sin(theta) - (1.5*theta + 1.5*theta_dot))
        return torch.clamp(u.unsqueeze(1), -3.0, 3.0)

# wrapper for rollout
def u_ref_fn(x, t):
    return u_nominal(x, t)

# =========================
# SIMULATION (WITH CONTROL TRACKING)
# =========================
def simulate(x0):
    x = x0.clone()
    traj = [x.clone()]
    u_hist = []
    t_hist = []

    dt = 0.01
    t = 0

    while t < 5:
        u_ref = u_nominal(x, t)
        u = u_qp(x, h_new(x), dhdx_new(x), u_ref, gamma=0.0, lmbda=1e6)

        fx = f(x).squeeze(-1)   # [B, 2]
        gx = g(x)               # [B, 2, 1]

        gu = torch.bmm(gx, u.unsqueeze(-1)).squeeze(-1)  # [B, 2]

        x = x + dt * (fx + gu)

        traj.append(x.clone())
        u_hist.append(u.clone())
        t_hist.append(t)

        t += dt

    return torch.stack(traj), torch.stack(u_hist), torch.tensor(t_hist)

# run sim for two ICs
x0s = torch.tensor([[0,0], [0.05,0.05]], dtype=torch.float32)

results = [simulate(x0.unsqueeze(0)) for x0 in x0s]

# =========================
# PLOT 2: STATE TRAJECTORIES
# =========================
fig2, ax2 = plt.subplots()

for i, (traj, _, _) in enumerate(results):
    traj = traj.squeeze(1)
    ax2.plot(traj[:,0], traj[:,1], label=f"x0={x0s[i].tolist()}")

ax2.fill_betweenx([-2,2], 0.3, 0.4, color='red', alpha=0.1)
ax2.fill_betweenx([-2,2], -0.4, -0.3, color='red', alpha=0.1)

ax2.set_xlabel("theta")
ax2.set_ylabel("theta_dot")
ax2.set_title("State Trajectories (CBF-QP)")
ax2.legend()
ax2.grid()

fig2.savefig("trajectories.png")
plt.close(fig2)

# =========================
# PLOT 3: CONTROL TRAJECTORIES
# =========================
fig3, ax3 = plt.subplots()

for i, (_, u_hist, t_hist) in enumerate(results):
    ax3.plot(t_hist.numpy(), u_hist.squeeze().numpy(), label=f"x0={x0s[i].tolist()}")

ax3.set_xlabel("time")
ax3.set_ylabel("u(t)")
ax3.set_title("Control Inputs (CBF-QP)")
ax3.legend()
ax3.grid()

fig3.savefig("controls.png")
plt.close(fig3)