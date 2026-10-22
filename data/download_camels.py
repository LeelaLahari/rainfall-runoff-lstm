"""
Downloads CAMELS-US meteorological forcing + streamflow data from Zenodo.
Source: https://zenodo.org/records/15529996
File: basin_timeseries_v1p2_metForcing_obsFlow.zip
Run once; saves to Google Drive so it persists across Colab sessions.
"""
import os
import subprocess
import zipfile
import argparse

ZENODO_RECORD = "15529996"
ZIP_NAME = "basin_timeseries_v1p2_metForcing_obsFlow.zip"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{ZIP_NAME}"


def download(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, ZIP_NAME)

    if os.path.exists(zip_path):
        print(f"Already downloaded: {zip_path}")
    else:
        print(f"Downloading from {ZENODO_URL} ...")
        subprocess.run(["wget", "-O", zip_path, ZENODO_URL], check=True)

    extract_dir = os.path.join(out_dir, "CAMELS_US")
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")
    else:
        print(f"Already extracted: {extract_dir}")

    return extract_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="/content/drive/MyDrive/rainfall-runoff-lstm/data")
    args = parser.parse_args()
    path = download(args.out_dir)
    print(f"\nCAMELS-US ready at: {path}")
    print("Expected subfolders: basin_mean_forcing/, usgs_streamflow/")