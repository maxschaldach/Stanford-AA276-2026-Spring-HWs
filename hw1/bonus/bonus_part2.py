import torch
from bonus_part1 import f, g


"""Note: the following functions operate on batched inputs."""


def euler_step(x, u, dt):
    """
    args:
        x: [B, 2]
        u: [B, 1]
    returns:
        xn: [B, 2]
    """
    fx = f(x)                      # [B, 2]
    gx = g(x)                      # [B, 2, 1]

    gu = torch.bmm(gx, u.unsqueeze(-1))  # [B, 2, 1]

    fx = fx.squeeze(-1)            # [B, 2]
    gu = gu.squeeze(-1)            # [B, 2]

    xn = x + dt * (fx + gu)
    return xn
    
def roll_out(x0, u_fn, nt, dt):
    """
    args:
        x0: [B, 2]
        u_fn: x -> [B, 1]
    returns:
        xts: [B, nt, 2]
    """
    xs = []
    x = x0

    for _ in range(nt):
        u = u_fn(x)                # [B, 1]
        x = euler_step(x, u, dt)   # [B, 2]
        xs.append(x)

    xts = torch.stack(xs, dim=1)   # [B, nt, 2]
    return xts


import cvxpy as cp
from bonus_part1 import control_limits


def u_qp(x, h, dhdx, u_ref, gamma, lmbda):
    """
    args:
        x: [B, 2]
        h: [B]
        dhdx: [B, 2]
        u_ref: [B, 1]
    returns:
        u_qp: [B, 1]
    """
    B = x.shape[0]

    u_out = []

    # control limits
    u_upper, u_lower = control_limits()
    u_upper = u_upper.numpy()
    u_lower = u_lower.numpy()

    for i in range(B):
        xi = x[i].unsqueeze(0)

        # dynamics
        fi = f(xi)[0].detach().cpu().numpy()        # [2]
        gi = g(xi)[0].detach().cpu().numpy()        # [2,1]

        grad_h = dhdx[i].detach().cpu().numpy()     # [2]
        hi = h[i].item()
        uref = u_ref[i].detach().cpu().numpy()      # [1]

        # Lie derivatives
        Lf_h = grad_h @ fi                          # scalar
        Lg_h = (grad_h @ gi).item()                 # scalar (IMPORTANT)

        # decision variables
        u = cp.Variable(1)
        delta = cp.Variable()

        # objective
        objective = cp.Minimize(cp.sum_squares(u - uref) + lmbda * cp.square(delta))

        # constraints
        constraints = [
            Lg_h * u + delta >= -Lf_h - gamma * hi,
            u >= u_lower,
            u <= u_upper,
            delta >= 0
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)

        # fallback
        if u.value is None:
            u_out.append(uref)
        else:
            u_out.append(u.value)

    return torch.tensor(u_out, dtype=torch.float32)