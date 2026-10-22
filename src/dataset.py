"""
CAMELS-US PyTorch Dataset: loads daily forcing (precip, temp, PET) +
static basin attributes, builds sliding-window sequences, and streamflow
targets. Splits by BASIN (not just time) for the held-out generalization set.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import glob


FORCING_COLS = ["prcp(mm/day)", "tmax(C)", "tmin(C)", "srad(W/m2)", "vp(Pa)"]


def load_basin_list(camels_root, n_basins=None):
    forcing_dir = os.path.join(camels_root, "basin_mean_forcing", "daymet")
    basin_files = glob.glob(os.path.join(forcing_dir, "**", "*.txt"), recursive=True)
    basin_ids = sorted([os.path.basename(f).split("_")[0] for f in basin_files])
    if n_basins:
        basin_ids = basin_ids[:n_basins]
    return basin_ids


def load_forcing(camels_root, basin_id, forcing_type="daymet"):
    pattern = os.path.join(camels_root, "basin_mean_forcing", forcing_type, "**", f"{basin_id}_*.txt")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    df = pd.read_csv(files[0], sep=r"\s+", skiprows=4,
                      names=["Year", "Mnth", "Day", "Hr"] + FORCING_COLS)
    df["date"] = pd.to_datetime(df[["Year", "Mnth", "Day"]].rename(
        columns={"Year": "year", "Mnth": "month", "Day": "day"}))
    return df.set_index("date")[FORCING_COLS]


def load_streamflow(camels_root, basin_id):
    pattern = os.path.join(camels_root, "usgs_streamflow", "**", f"{basin_id}_streamflow_qc.txt")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    df = pd.read_csv(files[0], sep=r"\s+",
                      names=["basin_id", "Year", "Mnth", "Day", "QObs", "flag"])
    df["date"] = pd.to_datetime(df[["Year", "Mnth", "Day"]].rename(
        columns={"Year": "year", "Mnth": "month", "Day": "day"}))
    df["QObs"] = df["QObs"].replace(-999, np.nan)   # CAMELS missing-value code
    return df.set_index("date")["QObs"]


class CamelsDataset(Dataset):
    def __init__(self, camels_root, basin_ids, date_range, seq_len=365,
                 forcing_type="daymet", scaler=None):
        self.seq_len = seq_len
        self.samples = []  # list of (forcing_window, target)
        self.basin_ids_used = []

        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])

        for basin_id in basin_ids:
            forcing = load_forcing(camels_root, basin_id, forcing_type)
            flow = load_streamflow(camels_root, basin_id)
            if forcing is None or flow is None:
                continue

            merged = forcing.join(flow, how="inner")
            merged = merged.loc[start - pd.Timedelta(days=seq_len):end].dropna()
            if len(merged) < seq_len + 1:
                continue

            x_vals = merged[FORCING_COLS].values.astype(np.float32)
            y_vals = merged["QObs"].values.astype(np.float32)

            for i in range(seq_len, len(merged)):
                self.samples.append((x_vals[i - seq_len:i], y_vals[i], basin_id))

        if scaler is None:
            all_x = np.concatenate([s[0] for s in self.samples], axis=0)
            self.x_mean, self.x_std = all_x.mean(axis=0), all_x.std(axis=0) + 1e-6
            all_y = np.array([s[1] for s in self.samples])
            self.y_mean, self.y_std = all_y.mean(), all_y.std() + 1e-6
        else:
            self.x_mean, self.x_std, self.y_mean, self.y_std = scaler

    def get_scaler(self):
        return (self.x_mean, self.x_std, self.y_mean, self.y_std)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, basin_id = self.samples[idx]
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        return torch.from_numpy(x_norm), torch.tensor(y_norm, dtype=torch.float32)


def basin_held_out_split(basin_ids, held_out_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)
    shuffled = basin_ids.copy()
    rng.shuffle(shuffled)
    n_held_out = int(len(shuffled) * held_out_frac)
    return shuffled[n_held_out:], shuffled[:n_held_out]   # train_basins, held_out_basins