import torch
import numpy as np


# ------------------------------------------------------------
# Load CBF model WITHOUT neural_clbf dependency
# ------------------------------------------------------------
def load_cbf_model(path='outputs/cbf.ckpt'):
    import torch
    import pytorch_lightning.callbacks.model_checkpoint as mc

    # 🔑 allow Lightning objects
    torch.serialization.add_safe_globals([mc.ModelCheckpoint])

    device = torch.device('cpu')

    ckpt = torch.load(path, map_location=device, weights_only=False)

    state_dict = ckpt['state_dict']

    # remove "model." prefix
    clean_state_dict = {
        k.replace("model.", ""): v for k, v in state_dict.items()
    }

    # infer architecture
    layers = [v.shape for k, v in clean_state_dict.items() if "weight" in k]
    sizes = [layers[0][1]] + [l[0] for l in layers]

    modules = []
    for i in range(len(sizes) - 2):
        modules.append(torch.nn.Linear(sizes[i], sizes[i+1]))
        modules.append(torch.nn.ReLU())
    modules.append(torch.nn.Linear(sizes[-2], sizes[-1]))

    model = torch.nn.Sequential(*modules)
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    return model


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------
def sample_states(N):
    x = np.zeros((N, 13))

    x[:, 0:3] = np.random.uniform(-3, 3, size=(N, 3))
    x[:, 3:7] = np.random.uniform(-1, 1, size=(N, 4))
    x[:, 7:10] = np.random.uniform(-5, 5, size=(N, 3))
    x[:, 10:13] = np.random.uniform(-5, 5, size=(N, 3))

    return torch.tensor(x, dtype=torch.float32)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    N = 200000

    print("Loading CBF model...")
    model = load_cbf_model()
    print("Model loaded.")

    print("Sampling states...")
    x = sample_states(N)

    print("Evaluating CBF...")
    with torch.no_grad():
        h = model(x).squeeze()

    safe_fraction = (h > 0).float().mean().item()

    print("\n=== HW1 CBF Volume Result ===")
    print(f"CBF safe set fraction: {safe_fraction:.6f}")


if __name__ == "__main__":
    main()