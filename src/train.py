"""
Training loop with BASIN-held-out validation — generalization to unseen
basins, not just unseen time periods. This is the rigor that impresses
reviewers in hydrology-ML papers, and it directly mirrors your flood-
emulator's terrain-cluster held-out approach.
"""
import torch
from torch.utils.data import DataLoader
import yaml
import sys
import numpy as np
from tqdm import tqdm

sys.path.append("src")
from dataset import CamelsDataset, load_basin_list, basin_held_out_split
from model import RainfallRunoffLSTM
from metrics import nse


def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    all_obs, all_sim = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            all_sim.extend(pred * y_std + y_mean)
            all_obs.extend(y.numpy() * y_std + y_mean)
    return nse(all_obs, all_sim)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_basins = load_basin_list(cfg["data"]["camels_root"], cfg["data"]["basin_list_size"])
    train_basins, held_out_basins = basin_held_out_split(
        all_basins, cfg["data"]["held_out_basins_frac"]
    )
    print(f"Train basins: {len(train_basins)}, Held-out basins: {len(held_out_basins)}")

    train_ds = CamelsDataset(
        cfg["data"]["camels_root"], train_basins, cfg["data"]["train_date_range"],
        seq_len=cfg["data"]["sequence_length"], forcing_type=cfg["data"]["forcing_type"]
    )
    scaler = train_ds.get_scaler()

    val_ds = CamelsDataset(
        cfg["data"]["camels_root"], train_basins, cfg["data"]["val_date_range"],
        seq_len=cfg["data"]["sequence_length"], forcing_type=cfg["data"]["forcing_type"],
        scaler=scaler
    )
    held_out_ds = CamelsDataset(
        cfg["data"]["camels_root"], held_out_basins, cfg["data"]["test_date_range"],
        seq_len=cfg["data"]["sequence_length"], forcing_type=cfg["data"]["forcing_type"],
        scaler=scaler
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
    held_out_loader = DataLoader(held_out_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    model = RainfallRunoffLSTM(
        input_size=5,
        hidden_size=cfg["training"]["hidden_size"],
        num_layers=cfg["training"]["num_layers"],
        dropout_p=cfg["training"]["dropout_p"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    criterion = torch.nn.MSELoss()

    best_val_nse = -np.inf
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_nse = evaluate(model, val_loader, device, scaler[2], scaler[3])
        print(f"Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.4f}, val_NSE={val_nse:.4f}")

        if val_nse > best_val_nse:
            best_val_nse = val_nse
            torch.save({"model_state": model.state_dict(), "scaler": scaler},
                       "best_rainfall_runoff_lstm.pt")

    held_out_nse = evaluate(model, held_out_loader, device, scaler[2], scaler[3])
    print(f"\n=== Final held-out-BASIN NSE (never seen in training): {held_out_nse:.4f} ===")


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    train(cfg)