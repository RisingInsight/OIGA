"""OIGA data preparation for FLAIR-HUB TIFF shards.

This script converts TIFF tar shards into normalized float32 NumPy tar shards.
The output contains four modalities per sample: RGB, NIR, DEM, and SPOT.

Usage:
    python preprocess_oiga.py --input_dir ./datasets_preprocess/FLAIR_ALL_MODALITIES/train \
        --output_dir ./datasets_preprocess/FLAIR_OIGA_NPY/train --shard_size 350
"""

import argparse
import glob
import io
import json
import os

import numpy as np
import tifffile
import webdataset as wds


def normalize_to_float32(image):
    """Scale an image to float32 in [0, 1] according to its dtype."""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    if image.dtype in (np.float32, np.float64):
        return image.astype(np.float32)
    raise ValueError(f"Unsupported dtype: {image.dtype}")


def convert_shards(input_dir, output_dir, shard_size):
    os.makedirs(output_dir, exist_ok=True)
    input_shards = sorted(glob.glob(os.path.join(input_dir, "dataset-*.tar")))
    if not input_shards:
        raise FileNotFoundError(f"No tar shards found in {input_dir}")

    sink = wds.ShardWriter(os.path.join(output_dir, "dataset-%06d.tar"), maxcount=shard_size)
    total = 0
    for shard_path in input_shards:
        print(f"Processing {os.path.basename(shard_path)}...")
        dataset = wds.WebDataset(shard_path)
        for sample in dataset:
            key = sample["__key__"]
            output = {"__key__": key}
            for modality in ("rgb", "nir", "dem", "spot"):
                raw = sample[f"{modality}.tif"]
                image = tifffile.imread(io.BytesIO(raw))
                image = normalize_to_float32(image)
                if image.ndim == 2:
                    image = image[np.newaxis, ...]
                elif image.ndim == 3 and image.shape[2] < image.shape[0]:
                    image = image.transpose(2, 0, 1)
                output[f"{modality}.npy"] = image.tobytes()
            sink.write(output)
            total += 1
            if total % 500 == 0:
                print(f"  Converted {total} samples...")
    sink.close()

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"total_samples": total}, f, indent=2)
    print(f"Done. Total {total} samples written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TIFF tar shards to normalized float32 NumPy shards")
    parser.add_argument("--input_dir", required=True, help="Directory containing dataset-*.tar (TIFF)")
    parser.add_argument("--output_dir", required=True, help="Output directory for NumPy tar shards")
    parser.add_argument("--shard_size", type=int, default=350)
    args = parser.parse_args()
    convert_shards(args.input_dir, args.output_dir, args.shard_size)
