"""
MC-dropout inference for streamflow prediction intervals — same approach
as your flood-emulator's uncertainty module, applied here to give
confidence bounds on forecasted streamflow instead of a single number.
"""
import torch
import numpy as np
import sys
sys.path.append("src")
from model import RainfallRunoffLSTM


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def mc_dropout_predict(model, x, n_samples=30, device="cuda"):
    model.eval()
    enable_mc_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x.to(device))
            preds.append(pred.cpu().numpy())
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load("best_rainfall_runoff_lstm.pt", map_location=device)

    model = RainfallRunoffLSTM().to(device)
    model.load_state_dict(checkpoint["model_state"])

    x = torch.rand(4, 365, 5)  # dummy batch — replace with real held-out sequences
    mean_pred, std_pred = mc_dropout_predict(model, x, n_samples=30, device=device)

    print("Mean predicted (normalized):", mean_pred)
    print("Uncertainty (std):", std_pred)