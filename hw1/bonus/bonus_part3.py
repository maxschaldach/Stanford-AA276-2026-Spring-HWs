import torch


def plot_h(fig, ax, px, py, slice, h_fn):
    """
    Pendulum version: state dim = 2 (theta, theta_dot)
    """

    # grid
    PX, PY = torch.meshgrid(px, py)
    X = torch.zeros((len(px), len(py), 2))

    X[..., 0] = PX      # theta
    X[..., 1] = PY      # theta_dot

    # flatten
    X_flat = X.reshape(-1, 2)

    h_vals = h_fn(X_flat).reshape(len(px), len(py)).detach().cpu()

    # colormap
    c = ax.pcolormesh(px, py, h_vals.T, shading='auto')
    fig.colorbar(c, ax=ax)

    # zero level set
    ax.contour(px, py, h_vals.T, levels=[0], colors='black')


from bonus_part1 import safe_mask, failure_mask
from bonus_part2 import roll_out, u_qp


def plot_and_eval_xts(fig, ax, x0, u_ref_fn, h_fn, dhdx_fn, gamma, lmbda, nt, dt):
    """
    Pendulum version: state dim = 2
    """

    def u_fn(x):
        return u_qp(x, h_fn(x), dhdx_fn(x), u_ref_fn(x), gamma, lmbda)

    # rollout
    xts = roll_out(x0, u_fn, nt, dt)   # [B, nt, 2]

    B = x0.shape[0]

    # ---- plot trajectories ----
    xts_np = xts.detach().cpu()

    for i in range(B):
        theta = xts_np[i, :, 0]
        theta_dot = xts_np[i, :, 1]
        ax.plot(theta, theta_dot)

    # ---- compute false safety rate ----

    is_safe0 = safe_mask(x0)   # [B]

    xts_flat = xts.reshape(-1, 2)
    failure = failure_mask(xts_flat).reshape(B, nt)

    violated = failure.any(dim=1)

    false_safe = is_safe0 & violated

    if is_safe0.sum() == 0:
        false_safety_rate = 0.0
    else:
        false_safety_rate = false_safe.sum().item() / is_safe0.sum().item()

    return false_safety_rate