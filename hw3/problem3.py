import torch

# So that I can run code on my Mac CPU
torch.Tensor.cuda = lambda self, *args, **kwargs: self.to('cpu')
torch.nn.Module.cuda = lambda self, *args, **kwargs: self.to('cpu')

orig_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['map_location'] = torch.device('cpu')
    return orig_load(*args, **kwargs)
torch.load = patched_load

import cvxpy as cp
from problem3_helper import control_limits, f, g

from problem3_helper import NeuralVF
vf = NeuralVF()

# environment setup
obstacles = torch.tensor([
    [1.0,  0.0, 0.5], # [px, py, radius]
    [4.0,  2.0, 1.0],
    [4.0, -2.0, 1.0],
    [7.0,  0.0, 1.5],
    [7.0,  4.0, 0.5],
    [7.0, -4.0, 0.5]
])

def smooth_blending_safety_filter(x, u_nom, gamma, lmbda):
    """
    Compute the smooth blending safety filter.
    Refer to the definition provided in the handout.
    You might find it useful to use functions from
    previous homeworks, which we have imported for you.
    These include:
      control_limits(.)
      f(.)
      g(.)
      vf.values(.)
      vf.gradients(.)
    NOTE: some of these functions expect batched inputs,
    but x, u_nom are not batched inputs in this case.
    
    args:
        x:      torch tensor with shape [13]
        u_nom:  torch tensor with shape [4]
        
    returns:
        u_sb:   torch tensor with shape [4]
    """
    num_obs = obstacles.shape[0]
    
    x_batch = x.unsqueeze(0).repeat(num_obs, 1)
    x_batch[:, 0] -= obstacles[:, 0]  # relative px
    x_batch[:, 1] -= obstacles[:, 1]  # relative py
    
    V_batch = vf.values(x_batch)
    dVdx_batch = vf.gradients(x_batch)
    
    radius_diff = obstacles[:, 2] - 0.5
    V_batch = V_batch - radius_diff
    
    min_idx = torch.argmin(V_batch)
    Vx = V_batch[min_idx].item()
    dVdx = dVdx_batch[min_idx].numpy()  # shape (13,)
    
    x_unbatched = x.unsqueeze(0)
    fx = f(x_unbatched).squeeze(0).numpy()  # shape (13,)
    gx = g(x_unbatched).squeeze(0).numpy()  # shape (13, 4)
    
    a = dVdx @ gx  # shape (4,)
    b = dVdx @ fx + gamma * Vx
    
    u_upper, u_lower = control_limits()
    u_upper = u_upper.numpy()
    u_lower = u_lower.numpy()
    u_nom_np = u_nom.numpy()
    
    u_var = cp.Variable(4)
    s_var = cp.Variable(1)
    
    cost = cp.sum_squares(u_var - u_nom_np) + lmbda * cp.square(s_var)
    constraints = [
        a @ u_var + b + s_var >= 0,
        u_var <= u_upper,
        u_var >= u_lower,
        s_var >= 0
    ]
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    
    if u_var.value is None:
        return u_nom
        
    return torch.tensor(u_var.value, dtype=torch.float32)