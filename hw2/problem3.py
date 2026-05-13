import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import hj_reachability as hj

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hw2.problem3_helper import save_values_gif, plot_value_and_safe_set_boundary


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

OUTPUT_DIR = REPO_ROOT / "hw2" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class InvertedPendulum(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self):
        control_space = hj.sets.Box(
            jnp.array([-U_MAX]),
            jnp.array([U_MAX]),
        )
        disturbance_space = hj.sets.Box(
            jnp.array([0.0]),
            jnp.array([0.0]),
        )
        super().__init__(
            control_mode="max",
            disturbance_mode="max",
            control_space=control_space,
            disturbance_space=disturbance_space,
        )

    def open_loop_dynamics(self, state, time):
        theta, theta_dot = state
        return jnp.array([
            theta_dot,
            (G / L) * jnp.sin(theta),
        ])
    

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.0],
            [1.0 / (M * L**2)],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.zeros((2, 1))


def wrap_theta(theta):
    return ((theta + np.pi) % (2.0 * np.pi)) - np.pi


def build_grid():
    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(
            np.array([THETA_MIN, THETA_DOT_MIN]),
            np.array([THETA_MAX, THETA_DOT_MAX]),
        ),
        (101, 101),
        periodic_dims=0,
    )


def initial_values_on_grid(grid):
    # Failure set: |theta| >= 0.3
    # Implicit surface negative inside failure set, positive in the safe region.
    return jnp.array(SAFE_ANGLE - jnp.abs(grid.states[..., 0]))


def solve_brt(grid, dynamics):
    times = np.linspace(0.0, -5.0, 101)
    initial_values = initial_values_on_grid(grid)

    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high",
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
    )

    all_values = np.asarray(
        hj.solve(solver_settings, dynamics, grid, times, initial_values)
    )

    np.save(OUTPUT_DIR / "problem3_all_values.npy", all_values)
    np.save(OUTPUT_DIR / "problem3_times.npy", times)

    return times, all_values


def estimate_convergence_time(times, all_values, tol=1e-3):
    diffs = np.max(np.abs(np.diff(all_values, axis=0).reshape(len(times) - 1, -1)), axis=1)
    idx = np.where(diffs < tol)[0]
    if len(idx) == 0:
        return times[-1], diffs
    return times[idx[0] + 1], diffs


def save_representative_frames(times, all_values, grid):
    frame_idxs = np.linspace(0, len(times) - 1, 5, dtype=int)
    for k, i in enumerate(frame_idxs, start=1):
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        plot_value_and_safe_set_boundary(all_values[i], grid, ax)
        ax.set_title(f"$V(x, {times[i]:3.2f})$")
        ax.set_xlabel("$\\theta$ (rad)")
        ax.set_ylabel("$\\dot{\\theta}$ (rad/s)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"problem3_value_frame_{k}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def estimate_safe_area(values_converged, grid, n_samples=200_000, seed=0):
    theta_grid = np.asarray(grid.coordinate_vectors[0])
    theta_dot_grid = np.asarray(grid.coordinate_vectors[1])

    interpolator = RegularGridInterpolator(
        (theta_grid, theta_dot_grid),
        np.asarray(values_converged),
        bounds_error=False,
        fill_value=None,
    )

    rng = np.random.default_rng(seed)
    samples = np.column_stack([
        rng.uniform(theta_grid[0], theta_grid[-1], size=n_samples),
        rng.uniform(theta_dot_grid[0], theta_dot_grid[-1], size=n_samples),
    ])
    vals = interpolator(samples)
    safe_fraction = np.mean(vals > 0.0)

    area_total = (theta_grid[-1] - theta_grid[0]) * (theta_dot_grid[-1] - theta_dot_grid[0])
    safe_area = safe_fraction * area_total
    return safe_area, safe_fraction


def cbf_safe_area():
    a = 0.14
    b = np.sqrt(a * (3.0 - 20.0 * np.sin(a)) / 2.0)
    return np.pi * a * b, a, b


def compute_gradients(values_converged, grid):
    theta_grid = np.asarray(grid.coordinate_vectors[0])
    theta_dot_grid = np.asarray(grid.coordinate_vectors[1])

    dV_dtheta, dV_dtheta_dot = np.gradient(
        np.asarray(values_converged),
        theta_grid,
        theta_dot_grid,
        edge_order=2,
    )

    interp_dV_dtheta = RegularGridInterpolator(
        (theta_grid, theta_dot_grid),
        dV_dtheta,
        bounds_error=False,
        fill_value=None,
    )
    interp_dV_dtheta_dot = RegularGridInterpolator(
        (theta_grid, theta_dot_grid),
        dV_dtheta_dot,
        bounds_error=False,
        fill_value=None,
    )

    return interp_dV_dtheta, interp_dV_dtheta_dot


def optimal_safety_control(x, interp_dV_dtheta, interp_dV_dtheta_dot, grid):
    theta_grid = np.asarray(grid.coordinate_vectors[0])
    theta_dot_grid = np.asarray(grid.coordinate_vectors[1])

    xq = np.array([
        wrap_theta(x[0]),
        np.clip(x[1], theta_dot_grid[0], theta_dot_grid[-1]),
    ])

    grad_theta = interp_dV_dtheta(xq).item()
    grad_theta_dot = interp_dV_dtheta_dot(xq).item()

    # Control enters only in theta_dot dynamics with coefficient 1/(M*L^2)
    coefficient = grad_theta_dot / (M * L**2)

    return U_MAX if coefficient >= 0.0 else -U_MAX


def dynamics_step(x, u, dt):
    theta, theta_dot = x
    theta_next = theta + dt * theta_dot
    theta_dot_next = theta_dot + dt * ((G / L) * np.sin(theta) + u / (M * L**2))
    theta_next = wrap_theta(theta_next)
    return np.array([theta_next, theta_dot_next], dtype=float)


def simulate(x0, interp_dV_dtheta, interp_dV_dtheta_dot, grid, T=1.0, dt=0.01):
    n_steps = int(round(T / dt))
    ts = np.linspace(0.0, T, n_steps + 1)
    xs = np.zeros((n_steps + 1, 2), dtype=float)
    us = np.zeros(n_steps, dtype=float)

    xs[0] = np.asarray(x0, dtype=float)

    for k in range(n_steps):
        us[k] = optimal_safety_control(xs[k], interp_dV_dtheta, interp_dV_dtheta_dot, grid)
        xs[k + 1] = dynamics_step(xs[k], us[k], dt)

    return ts, xs, us


def plot_trajectories_and_controls(values_converged, grid, simulations):
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    plot_value_and_safe_set_boundary(values_converged, grid, ax)

    for ts, xs, us, label in simulations:
        ax.plot(xs[:, 0], xs[:, 1], linewidth=2.0, label=label)
        ax.plot(xs[0, 0], xs[0, 1], "o", markersize=5)

    ax.axvline(0.3, linestyle="--", linewidth=1.5, color="r")
    ax.axvline(-0.3, linestyle="--", linewidth=1.5, color="r")
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-1.0, 1.0])
    ax.set_xlabel("$\\theta$ (rad)")
    ax.set_ylabel("$\\dot{\\theta}$ (rad/s)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "problem3_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for ts, xs, us, label in simulations:
        ax.step(ts[:-1], us, where="post", linewidth=2.0, label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("control $u$ (N·m)")
    ax.set_ylim([-U_MAX - 0.25, U_MAX + 0.25])
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "problem3_controls.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    dynamics = InvertedPendulum()
    grid = build_grid()

    times, all_values = solve_brt(grid, dynamics)
    values_converged = np.asarray(all_values[-1])

    np.save(OUTPUT_DIR / "problem3_values_converged.npy", values_converged)

    save_values_gif(all_values, grid, times, save_path=str(OUTPUT_DIR / "problem3_values.gif"))
    save_representative_frames(times, all_values, grid)

    conv_time, diffs = estimate_convergence_time(times, all_values, tol=1e-3)
    safe_area, safe_fraction = estimate_safe_area(values_converged, grid, n_samples=200_000, seed=0)
    cbf_area, a, b = cbf_safe_area()

    interp_dV_dtheta, interp_dV_dtheta_dot = compute_gradients(values_converged, grid)

    initial_conditions = [
        (np.array([-0.1, 0.4]), "x0 = (-0.1, +0.4)"),
        (np.array([-0.1, -0.3]), "x0 = (-0.1, -0.3)"),
    ]

    simulations = []
    for x0, label in initial_conditions:
        ts, xs, us = simulate(x0, interp_dV_dtheta, interp_dV_dtheta_dot, grid, T=1.0, dt=0.01)
        simulations.append((ts, xs, us, label))

    plot_trajectories_and_controls(values_converged, grid, simulations)

    print("\n=== Problem 3 summary ===")
    print(f"Estimated convergence time: t ≈ {conv_time:.2f} s")
    print(f"Safe-set fraction in sampled state space: {safe_fraction:.6f}")
    print(f"Estimated maximal safe-set area: {safe_area:.6f}")
    print(f"CBF ellipse parameters: a = {a:.3f}, b = {b:.6f}")
    print(f"CBF safe-set area: {cbf_area:.6f}")
    print(f"Ratio HJ / CBF area: {safe_area / cbf_area:.3f}x")


if __name__ == "__main__":
    main()