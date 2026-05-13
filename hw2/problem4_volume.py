import torch
import numpy as np

from problem4_helper import NeuralVF


# ------------------------------------------------------------
# Sample from state space
# ------------------------------------------------------------
def sample_states(N):
    x = np.zeros((N, 13))

    # px, py, pz ∈ [-3, 3]
    x[:, 0:3] = np.random.uniform(-3, 3, size=(N, 3))

    # quaternion ∈ [-1, 1]
    x[:, 3:7] = np.random.uniform(-1, 1, size=(N, 4))

    # velocities ∈ [-5, 5]
    x[:, 7:10] = np.random.uniform(-5, 5, size=(N, 3))

    # angular velocities ∈ [-5, 5]
    x[:, 10:13] = np.random.uniform(-5, 5, size=(N, 3))

    return torch.tensor(x, dtype=torch.float32)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    N = 20000  # keep small so it runs fast on CPU

    print("Loading value function...")
    vf = NeuralVF(ckpt_path="outputs/vf.ckpt")
    print("Loaded.")

    print("Sampling...")
    x = sample_states(N)

    print("Evaluating...")
    V = vf.values(x)

    safe_fraction = (V > 0).float().mean().item()

    print("\n=== RESULT ===")
    print(f"HJ safe set fraction: {safe_fraction:.6f}")


if __name__ == "__main__":
    main()