import torch

# parameters
m = 2.0
l = 1.0
g_const = 10.0

# ===== STATE LIMITS =====
def state_limits():
    upper = torch.tensor([0.4, 2.0])
    lower = torch.tensor([-0.4, -2.0])
    return upper, lower

# ===== CONTROL LIMITS =====
def control_limits():
    upper = torch.tensor([3.0])
    lower = torch.tensor([-3.0])
    return upper, lower

# ===== SAFE SET =====
def safe_mask(x):
    theta = x[:, 0]
    return torch.abs(theta) <= 0.3

# ===== FAILURE SET =====
def failure_mask(x):
    theta = x[:, 0]
    return torch.abs(theta) > 0.3

# ===== DYNAMICS =====
def f(x):
    # ensure batch dimension
    if x.ndim == 1:
        x = x.unsqueeze(0)

    theta = x[:, 0]
    theta_dot = x[:, 1]

    out = torch.zeros_like(x)
    out[:, 0] = theta_dot
    out[:, 1] = (g_const / l) * torch.sin(theta)

    return out

def g(x):
    # ensure batch dimension
    if x.ndim == 1:
        x = x.unsqueeze(0)

    B = x.shape[0]
    G = torch.zeros((B, 2, 1), device=x.device)

    G[:, 1, 0] = 1 / (m * l**2)

    return G

# ===== ANALYTICAL CBF =====
a = 0.14
b = (a * (3 - 20 * torch.sin(torch.tensor(a))) / 2)**0.5

def h_old(x):
    theta = x[:, 0]
    theta_dot = x[:, 1]
    return 1 - (theta**2)/(a**2) - (theta_dot**2)/(b**2)