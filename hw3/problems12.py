import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from problem1_helper import plot_value_and_safe_set_boundary


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
HW2_OUTPUT_DIR = REPO_ROOT / "hw2" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Problem setup
# -----------------------------------------------------------------------------

M = 2.0
L = 1.0
G = 10.0
U_MAX = 3.0
THETA_MIN = -np.pi
THETA_MAX = np.pi
THETA_DOT_MIN = -10.0
THETA_DOT_MAX = 10.0
SAFE_ANGLE = 0.3
DISTURBANCE_MAX = 0.4

T_FINAL = 5.0
DT = 0.01


@dataclass(frozen=True)
class Grid2D:
    coordinate_vectors: tuple[np.ndarray, np.ndarray]


# -----------------------------------------------------------------------------
# Load HW2 value function
# -----------------------------------------------------------------------------

values_converged = np.load(HW2_OUTPUT_DIR / "problem3_values_converged.npy")


def build_grid():
    # 1. Theta was trained as periodic: 101 intervals, excluding +pi
    n_theta = values_converged.shape[0]
    dtheta = (THETA_MAX - THETA_MIN) / n_theta
    theta_grid = THETA_MIN + np.arange(n_theta) * dtheta
    
    # 2. Theta_dot was trained as non-periodic: 100 intervals, including endpoint
    n_theta_dot = values_converged.shape[1]
    theta_dot_grid = np.linspace(THETA_DOT_MIN, THETA_DOT_MAX, n_theta_dot)
    
    return Grid2D((theta_grid, theta_dot_grid))


grid = build_grid()
theta_grid = np.asarray(grid.coordinate_vectors[0])
theta_dot_grid = np.asarray(grid.coordinate_vectors[1])

value_interp = RegularGridInterpolator(
    (theta_grid, theta_dot_grid),
    np.asarray(values_converged),
    bounds_error=False,
    fill_value=None,
)

dV_dtheta_grid, dV_dtheta_dot_grid = np.gradient(
    np.asarray(values_converged),
    theta_grid,
    theta_dot_grid,
    edge_order=2,
)

dV_dtheta_interp = RegularGridInterpolator(
    (theta_grid, theta_dot_grid),
    dV_dtheta_grid,
    bounds_error=False,
    fill_value=None,
)

dV_dtheta_dot_interp = RegularGridInterpolator(
    (theta_grid, theta_dot_grid),
    dV_dtheta_dot_grid,
    bounds_error=False,
    fill_value=None,
)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def wrap_theta(theta):
    return ((theta + np.pi) % (2.0 * np.pi)) - np.pi


def clip_u(u):
    return float(np.clip(u, -U_MAX, U_MAX))


def query_value(x):
    xq = np.array([
        wrap_theta(float(x[0])),
        float(np.clip(x[1], THETA_DOT_MIN, THETA_DOT_MAX)),
    ])
    return float(value_interp(xq))

def query_gradients(x):
    xq = np.array([
        wrap_theta(float(x[0])),
        float(np.clip(x[1], THETA_DOT_MIN, THETA_DOT_MAX)),
    ])
    dtheta = float(dV_dtheta_interp(xq))
    dtheta_dot = float(dV_dtheta_dot_interp(xq))
    return dtheta, dtheta_dot


# -----------------------------------------------------------------------------
# Nominal controller from the handout
# -----------------------------------------------------------------------------


def u_nom(x, t):
    theta = float(x[0])
    theta_dot = float(x[1])

    if 0.0 <= t < 1.0:
        u = U_MAX
    elif 1.0 <= t < 2.0:
        u = -U_MAX
    elif 2.0 <= t < 3.0:
        u = U_MAX
    else:
        u = M * L**2 * (-(G / L) * np.sin(theta) - 1.5 * theta - 1.5 * theta_dot)

    return clip_u(u)


# -----------------------------------------------------------------------------
# HW2 optimal safety controller recovered from the converged value function
# -----------------------------------------------------------------------------


def u_safe(x):
    _, dV_dtheta_dot = query_gradients(x)
    coeff = dV_dtheta_dot / (M * L**2)
    return U_MAX if coeff >= 0.0 else -U_MAX


# -----------------------------------------------------------------------------
# Scalar QP helper
# -----------------------------------------------------------------------------


def solve_scalar_qp(u_ref, a, c, lower=-U_MAX, upper=U_MAX):

    eps = 1e-12

    lo, hi = lower, upper

    # Constraint:
    #
    # a*u + c >= 0

    if a > eps:
        lo = max(lo, -c / a)

    elif a < -eps:
        hi = min(hi, -c / a)

    else:
        if c < 0.0:
            return float(np.clip(u_ref, lower, upper))

    if lo > hi:

        if a > 0.0:
            return float(upper)

        elif a < 0.0:
            return float(lower)

        return float(np.clip(u_ref, lower, upper))

    return float(np.clip(u_ref, lo, hi))


# -----------------------------------------------------------------------------
# Problem 1 filters
# -----------------------------------------------------------------------------


def least_restrictive_safety_filter(x, t):
    if query_value(x) > 0.0:
        return u_nom(x, t)
    return u_safe(x)


def smooth_least_restrictive_safety_filter(x, t):
    xq = np.array([
        wrap_theta(float(x[0])),
        float(np.clip(x[1], THETA_DOT_MIN, THETA_DOT_MAX)),
    ])

    Vx = float(value_interp(xq))
    u_ref = u_nom(x, t)
    if Vx > 0.0:
        return u_ref

    dV_dtheta, dV_dtheta_dot = query_gradients(x)
    theta = float(x[0])
    theta_dot = float(x[1])

    a = dV_dtheta_dot / (M * L**2)
    c = dV_dtheta * theta_dot + dV_dtheta_dot * ((G / L) * np.sin(theta))

    return solve_scalar_qp(u_ref, a, c)


def smooth_blending_safety_filter(x, t, gamma):
    xq = np.array([
        wrap_theta(float(x[0])),
        float(np.clip(x[1], THETA_DOT_MIN, THETA_DOT_MAX)),
    ])

    Vx = float(value_interp(xq))
    u_ref = u_nom(x, t)
    dV_dtheta, dV_dtheta_dot = query_gradients(x)
    theta = float(x[0])
    theta_dot = float(x[1])

    a = dV_dtheta_dot / (M * L**2)
    c = dV_dtheta * theta_dot + dV_dtheta_dot * ((G / L) * np.sin(theta)) + gamma * Vx

    return solve_scalar_qp(u_ref, a, c)


# -----------------------------------------------------------------------------
# Problem 2 robustification against d in [0, 0.4]
# -----------------------------------------------------------------------------


def robust_least_restrictive_safety_filter(x, t):
    # Propose the smooth blending approach to act before hitting the boundary!
    gamma = 5.0 
    
    u_ref = u_nom(x, t)
    
    dV_dtheta, dV_dtheta_dot = query_gradients(x)
    theta, theta_dot = float(x[0]), float(x[1])
    Vx = query_value(x)

    # 1. Pessimistic disturbance (worst-case)
    d_worst = DISTURBANCE_MAX if dV_dtheta < 0.0 else 0.0

    a = dV_dtheta_dot / (M * L**2)
    
    # 2. Add gamma * Vx to smoothly enforce safety before we hit the edge
    c = (
        dV_dtheta * (theta_dot + d_worst) 
        + dV_dtheta_dot * ((G / L) * np.sin(theta))
        + gamma * Vx
    )

    return solve_scalar_qp(u_ref, a, c)

# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------


def dynamics_step(x, u, dt, disturbance=0.0):
    theta, theta_dot = float(x[0]), float(x[1])

    # Disturbance enters here per the HW prompt
    theta_next = theta + dt * (theta_dot + disturbance) 
    
    theta_dot_next = theta_dot + dt * (
        (G / L) * np.sin(theta)
        + u / (M * L**2)
    )

    theta_next = wrap_theta(theta_next)

    return np.array([theta_next, theta_dot_next], dtype=float)


def simulate(x0, controller_fn, T=T_FINAL, dt=DT, disturbed=False, rng=None):
    n_steps = int(round(T / dt))
    ts = np.linspace(0.0, T, n_steps + 1)
    xs = np.zeros((n_steps + 1, 2), dtype=float)
    us = np.zeros(n_steps, dtype=float)
    ds = np.zeros(n_steps, dtype=float)

    xs[0] = np.asarray(x0, dtype=float)
    if rng is None:
        rng = np.random.default_rng()

    for k in range(n_steps):
        t = ts[k]
        u = float(controller_fn(xs[k], t))
        d = float(rng.uniform(0.0, DISTURBANCE_MAX)) if disturbed else 0.0
        xs[k + 1] = dynamics_step(xs[k], u, dt, disturbance=d)
        us[k] = u
        ds[k] = d

    return ts, xs, us, ds


def is_safe_trajectory(xs):
    return bool(np.all(np.abs(xs[:, 0]) < SAFE_ANGLE))


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
val = query_value([0.05, 0.05])
print(f"V(0.05, 0.05) = {val}")

def plot_value_function():
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    plot_value_and_safe_set_boundary(values_converged, grid, ax)
    ax.set_title("Converged HJ value function")
    ax.set_xlabel("$\\theta$ (rad)")
    ax.set_ylabel("$\\dot{\\theta}$ (rad/s)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "problem1_value_function.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories_and_controls(tag, simulations):
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    plot_value_and_safe_set_boundary(values_converged, grid, ax)

    for ts, xs, us, label, nominal_us in simulations:
        ax.plot(xs[:, 0], xs[:, 1], linewidth=2.0, label=label)
        ax.plot(xs[0, 0], xs[0, 1], "o", markersize=5)

    ax.axvline(SAFE_ANGLE, linestyle="--", linewidth=1.2, color="r")
    ax.axvline(-SAFE_ANGLE, linestyle="--", linewidth=1.2, color="r")
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlabel("$\\theta$ (rad)")
    ax.set_ylabel("$\\dot{\\theta}$ (rad/s)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for ts, xs, us, label, nominal_us in simulations:
        ax.step(ts[:-1], us, where="post", linewidth=2.0, label=label)
        ax.step(ts[:-1], nominal_us, where="post", linestyle="--", linewidth=1.1, alpha=0.8,
                label=f"Nominal ({label})")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("control $u$ (N·m)")
    ax.set_ylim([-U_MAX - 0.25, U_MAX + 0.25])
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_controls.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_problem2_results(simulations, tag="problem2"):
    # ------------------------------------------------------------------
    # Trajectories
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    plot_value_and_safe_set_boundary(values_converged, grid, ax)

    for ts, xs, us, ds, label in simulations:
        ax.plot(xs[:, 0], xs[:, 1], linewidth=2.0, label=label)
        ax.plot(xs[0, 0], xs[0, 1], "o", markersize=5)

    ax.axvline(SAFE_ANGLE, linestyle="--", linewidth=1.2, color="r")
    ax.axvline(-SAFE_ANGLE, linestyle="--", linewidth=1.2, color="r")

    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-1.2, 1.2])

    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{tag}_trajectories.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    for ts, xs, us, ds, label in simulations:
        ax.step(ts[:-1], us, where="post", linewidth=2.0, label=label)

    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"control $u$ (N·m)")
    ax.set_ylim([-U_MAX - 0.25, U_MAX + 0.25])

    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{tag}_controls.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Disturbances
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.0))

    for ts, xs, us, ds, label in simulations:
        ax.step(
            ts[:-1],
            ds,
            where="post",
            linewidth=1.8,
            label=f"d ({label})",
        )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("disturbance $d$")

    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{tag}_disturbances.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run_problem_1():
    plot_value_function()

    initial_conditions = [
        (np.array([0.0, 0.0], dtype=float), "x0 = (0, 0)"),
        (np.array([0.05, 0.05], dtype=float), "x0 = (0.05, 0.05)"),
    ]

    for tag, controller in [
        ("problem1_lr", least_restrictive_safety_filter),
        ("problem1_slr", smooth_least_restrictive_safety_filter),
    ]:
        simulations = []
        for x0, label in initial_conditions:
            ts, xs, us, ds = simulate(x0, controller, disturbed=False, rng=np.random.default_rng(0))
            nominal_us = np.array([u_nom(xs[k], ts[k]) for k in range(len(ts) - 1)], dtype=float)
            simulations.append((ts, xs, us, label, nominal_us))
        plot_trajectories_and_controls(tag, simulations)

    for gamma in [0.0, 0.5, 5.0]:
        simulations = []
        for x0, label in initial_conditions:
            controller = lambda x, t, gamma=gamma: smooth_blending_safety_filter(x, t, gamma)
            ts, xs, us, ds = simulate(x0, controller, disturbed=False, rng=np.random.default_rng(0))
            nominal_us = np.array([u_nom(xs[k], ts[k]) for k in range(len(ts) - 1)], dtype=float)
            simulations.append((ts, xs, us, label, nominal_us))
        plot_trajectories_and_controls(f"problem1_sb_gamma_{gamma:g}", simulations)


def run_problem_2():
    x0 = np.array([0.05, 0.05], dtype=float)

    safe_lr = 0
    safe_robust = 0

    # representative trajectories for plotting
    rng_lr_plot = np.random.default_rng(123)
    rng_rb_plot = np.random.default_rng(123)

    ts_lr, xs_lr_plot, us_lr_plot, ds_lr_plot = simulate(
        x0,
        least_restrictive_safety_filter,
        disturbed=True,
        rng=rng_lr_plot,
    )

    ts_rb, xs_rb_plot, us_rb_plot, ds_rb_plot = simulate(
        x0,
        robust_least_restrictive_safety_filter,
        disturbed=True,
        rng=rng_rb_plot,
    )

    plot_problem2_results(
        [
            (ts_lr, xs_lr_plot, us_lr_plot, ds_lr_plot, "LR filter"),
            (ts_rb, xs_rb_plot, us_rb_plot, ds_rb_plot, "Robust filter"),
        ]
    )

    rng = np.random.default_rng(0)

    for _ in range(100):
        _, xs_lr, _, _ = simulate(
            x0,
            least_restrictive_safety_filter,
            disturbed=True,
            rng=rng,
        )
        safe_lr += int(is_safe_trajectory(xs_lr))

    rng = np.random.default_rng(1)

    for _ in range(100):
        _, xs_rb, _, _ = simulate(
            x0,
            robust_least_restrictive_safety_filter,
            disturbed=True,
            rng=rng,
        )
        safe_robust += int(is_safe_trajectory(xs_rb))

    rate_lr = safe_lr / 100.0
    rate_robust = safe_robust / 100.0

    print("Problem 2 safety rates over 100 simulations:")
    print(f"  Least-restrictive filter: {rate_lr:.3f}")
    print(f"  Robustified filter:       {rate_robust:.3f}")


if __name__ == "__main__":
    run_problem_1()
    run_problem_2()
