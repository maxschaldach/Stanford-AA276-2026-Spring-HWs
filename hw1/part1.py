"""
AA 276 Homework 1 | Coding Portion | Part 1 of 3


OVERVIEW

In this file, you will implement several functions required by the 
neural CBF library developed by the REALM Lab at MIT to
automatically learn your own CBFs for a 13D quadrotor system!

From this exercise, you will hopefully better understand the course
materials through a concrete example, appreciate the advantages
(and disadvantages) of learning a CBF versus manually constructing
one, and get some hands-on coding experience with using state-of-the-art
tools for synthesizing safety certificates, which you might find
useful for your own work.

If you are interested in learning more, you can find the library
here: https://github.com/MIT-REALM/neural_clbf


INSTRUCTIONS

Please refer to the Homework 1 handout for instructions and implementation details.

Function headers are provided below.
Your code should go into the sections marked by "YOUR CODE HERE"

When you are done, you can sanity check your code (locally) by running `python scripts/check1.py`.
After the tests pass, train a neural CBF (in your VM) by running `python scripts/train.py`.


IMPORTANT NOTES ON TRAINING
The training can take a substantial amount of time to complete [~9 hours ~= $10].
However, you should be able to implement all code for Parts 1, 2, and 3 in the meantime.
After each training epoch [50 total], the CBF model will save to 'outputs/cbf.ckpt'.
As long as you have at least one checkpoint saved [~10 minutes], Part 3 will load this checkpoint.
Try your best to not exceed $10 in credits -  you can stop training early if you reach this budget limit.

REMEMBER TO SHUTDOWN YOUR VIRTUAL MACHINES AFTER TRAINING, TO AVOID ACCUMULATING FEES.
"""


import torch


def state_limits():
    """
    Return a tuple (upper, lower) describing the state bounds for the system.
    
    returns:
        (upper, lower)
            where upper: torch float32 tensor with shape [13]
                  lower: torch float32 tensor with shape [13]
    """
    upper = torch.tensor([
         3,  3,  3,      # px, py, pz
         1,  1,  1,  1,  # qw, qx, qy, qz
         5,  5,  5,      # vx, vy, vz
         5,  5,  5       # wx, wy, wz
    ], dtype=torch.float32)

    lower = torch.tensor([
        -3, -3, -3,
        -1, -1, -1, -1,
        -5, -5, -5,
        -5, -5, -5
    ], dtype=torch.float32)

    return upper, lower


def control_limits():
    """
    Return a tuple (upper, lower) describing the control bounds for the system.
    
    returns:
        (upper, lower)
            where upper: torch float32 tensor with shape [4]
                  lower: torch float32 tensor with shape [4]
    """
    upper = torch.tensor([20, 8, 8, 4], dtype=torch.float32)
    lower = torch.tensor([-20, -8, -8, -4], dtype=torch.float32)
    return upper, lower


"""Note: the following functions operate on batched inputs.""" 


def safe_mask(x):
    """
    Return a boolean tensor indicating whether the states x are in the prescribed safe set.

    args:
        x: torch float32 tensor with shape [batch_size, 13]

    returns:
        is_safe: torch bool tensor with shape [batch_size]
    """
    px = x[:, 0]
    py = x[:, 1]

    r = torch.sqrt(px**2 + py**2)
    return r > 2.8


def failure_mask(x):
    """
    Return a boolean tensor indicating whether the states x are in the failure set.

    args:
        x: torch float32 tensor with shape [batch_size, 13]

    returns:
        is_failure: torch bool tensor with shape [batch_size]
    """
    px = x[:, 0]
    py = x[:, 1]

    r = torch.sqrt(px**2 + py**2)
    return r < 0.5


def f(x):
    """
    Return the control-independent part of the control-affine dynamics.
    Note: we have already implemented this for you!

    args:
        x: torch float32 tensor with shape [batch_size, 13]
        
    returns:
        f: torch float32 tensor with shape [batch_size, 13]
    """
    PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
    PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [x[:, i] for i in range(13)]

    f = torch.zeros_like(x)
    f[:, PXi] = VX
    f[:, PYi] = VY
    f[:, PZi] = VZ
    f[:, QWi] = -0.5*(WX*QX + WY*QY + WZ*QZ)
    f[:, QXi] =  0.5*(WX*QW + WZ*QY - WY*QZ)
    f[:, QYi] =  0.5*(WY*QW - WZ*QX + WX*QZ)
    f[:, QZi] =  0.5*(WZ*QW + WY*QX - WX*QY)
    f[:, VZi] = -9.8
    f[:, WXi] = -5 * WY * WZ / 9.0
    f[:, WYi] =  5 * WX * WZ / 9.0
    return f


def g(x):
    """
    Return the control-dependent part of the control-affine dynamics.

    args:
        x: torch float32 tensor with shape [batch_size, 13]
       
    returns:
        g: torch float32 tensor with shape [batch_size, 13, 4]
    """
    batch_size = x.shape[0]

    # indices
    QW, QX, QY, QZ = 3, 4, 5, 6
    VX, VY, VZ = 7, 8, 9
    WX, WY, WZ = 10, 11, 12

    qw = x[:, QW]
    qx = x[:, QX]
    qy = x[:, QY]
    qz = x[:, QZ]

    g = torch.zeros((batch_size, 13, 4), dtype=torch.float32)

    # ---- F (thrust) ----
    g[:, VX, 0] = 2 * (qw * qy + qx * qz)
    g[:, VY, 0] = 2 * (qy * qz - qw * qx)
    g[:, VZ, 0] = 2 * (0.5 - qx**2 - qy**2)

    # ---- alpha_x ----
    g[:, WX, 1] = 1.0

    # ---- alpha_y ----
    g[:, WY, 2] = 1.0

    # ---- alpha_z ----
    g[:, WZ, 3] = 1.0

    return g