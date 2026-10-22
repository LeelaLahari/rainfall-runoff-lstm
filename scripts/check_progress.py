"""Quick sanity check: confirms data downloaded correctly and basin count."""
import sys
sys.path.append("src")
from dataset import load_basin_list
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

basins = load_basin_list(cfg["data"]["camels_root"])
print(f"Total basins found: {len(basins)}")
print(f"First 5 basin IDs: {basins[:5]}")