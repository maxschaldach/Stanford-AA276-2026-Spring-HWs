import torch


# ===== PARAMETERS =====
m = 2.0
l = 1.0
g_const = 10.0


def state_limits():
    """
    Return a tuple (upper, lower) describing the state bounds for the system.
    
    returns:
        (upper, lower)
            where upper: torch float32 tensor with shape [2]
                  lower: torch float32 tensor with shape [2]
    """
    upper = torch.tensor([
        0.4,   # theta
        2.0    # theta_dot
    ], dtype=torch.float32)

    lower = torch.tensor([
        -0.4,
        -2.0
    ], dtype=torch.float32)

    return upper, lower


def control_limits():
    """
    Return a tuple (upper, lower) describing the control bounds for the system.
    
    returns:
        (upper, lower)
            where upper: torch float32 tensor with shape [1]
                  lower: torch float32 tensor with shape [1]
    """
    upper = torch.tensor([3.0], dtype=torch.float32)
    lower = torch.tensor([-3.0], dtype=torch.float32)
    return upper, lower


"""Note: the following functions operate on batched inputs.""" 

def safe_mask(x):
    """
    Safe set: ellipse learned in problem 2
    """

    # parameters from Problem 2
    a = 0.14
    b = (a * (3 - 20 * torch.sin(torch.tensor(a, device=x.device))) / 2)**0.5

    theta = x[:, 0]
    theta_dot = x[:, 1]

    h = 1 - (theta**2)/(a**2) - (theta_dot**2)/(b**2)
    return h > 0.02


def failure_mask(x):
    """
    Failure set: |theta| > 0.3
    """
    theta = x[:, 0]
    return torch.abs(theta) > 0.3


def f(x):
    """
    Control-independent dynamics f(x)

    returns:
        f: [batch_size, 2]
    """
    theta = x[:, 0]
    theta_dot = x[:, 1]

    f = torch.zeros((x.shape[0], 2, 1), dtype=x.dtype, device=x.device)
    f[:, 0] = theta_dot
    f[:, 1] = (g_const / l) * torch.sin(theta)

    return f


def g(x):
    """
    Control-dependent dynamics g(x)

    returns:
        g: [batch_size, 2, 1]
    """
    batch_size = x.shape[0]

    g = torch.zeros((batch_size, 2, 1), dtype=torch.float32)

    g[:, 1, 0] = 1.0 / (m * l**2)

    return g