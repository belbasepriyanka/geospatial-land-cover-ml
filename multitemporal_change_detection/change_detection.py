"""Multi-temporal categorical land-cover change detection.

This module compares two classified rasters after checking their grid geometry.
It writes a transition-code raster and CSV summaries that can be used for
mapping, QA/QC, and downstream analysis.

Example
-------
python change_detection.py \
    --before data/landcover_2018.tif \
    --after data/landcover_2024.tif \
    --out-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def validate_grids(before: rasterio.io.DatasetReader, after: rasterio.io.DatasetReader) -> None:
    """Require matching CRS, transform, shape, and pixel size."""
    checks = {
        "CRS": before.crs == after.crs,
        "transform": before.transform == after.transform,
        "width": before.width == after.width,
        "height": before.height == after.height,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(
            "Input rasters are not on the same analysis grid. "
            f"Failed checks: {', '.join(failed)}. Align/resample before comparison."
        )


def build_transition_codes(before_arr: np.ndarray, after_arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Encode each valid transition as before_class * 1000 + after_class."""
    transitions = np.full(before_arr.shape, -9999, dtype=np.int32)
    transitions[valid] = before_arr[valid].astype(np.int32) * 1000 + after_arr[valid].astype(np.int32)
    return transitions


def transition_table(before_arr: np.ndarray, after_arr: np.ndarray, valid: np.ndarray) -> pd.DataFrame:
    """Return pixel counts for every observed class-to-class transition."""
    df = pd.DataFrame(
        {
            "class_before": before_arr[valid].astype(int),
            "class_after": after_arr[valid].astype(int),
        }
    )
    out = (
        df.value_counts(["class_before", "class_after"])
        .rename("pixel_count")
        .reset_index()
        .sort_values(["class_before", "class_after"])
    )
    out["changed"] = out["class_before"] != out["class_after"]
    return out


def class_summary(before_arr: np.ndarray, after_arr: np.ndarray, valid: np.ndarray, pixel_area: float) -> pd.DataFrame:
    """Summarize persistence, gain, loss, and net change by class."""
    classes = np.union1d(np.unique(before_arr[valid]), np.unique(after_arr[valid])).astype(int)
    rows = []
    for cls in classes:
        before_mask = valid & (before_arr == cls)
        after_mask = valid & (after_arr == cls)
        persistent = before_mask & after_mask
        loss = before_mask & ~after_mask
        gain = ~before_mask & after_mask & valid

        before_n = int(before_mask.sum())
        after_n = int(after_mask.sum())
        rows.append(
            {
                "class": cls,
                "pixels_before": before_n,
                "pixels_after": after_n,
                "persistent_pixels": int(persistent.sum()),
                "loss_pixels": int(loss.sum()),
                "gain_pixels": int(gain.sum()),
                "net_pixel_change": after_n - before_n,
                "area_before": before_n * pixel_area,
                "area_after": after_n * pixel_area,
                "net_area_change": (after_n - before_n) * pixel_area,
            }
        )
    return pd.DataFrame(rows)


def run(before_path: Path, after_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(before_path) as before, rasterio.open(after_path) as after:
        validate_grids(before, after)

        before_arr = before.read(1)
        after_arr = after.read(1)

        before_valid = np.ones(before_arr.shape, dtype=bool)
        after_valid = np.ones(after_arr.shape, dtype=bool)
        if before.nodata is not None:
            before_valid &= before_arr != before.nodata
        if after.nodata is not None:
            after_valid &= after_arr != after.nodata
        valid = before_valid & after_valid

        transitions = build_transition_codes(before_arr, after_arr, valid)

        profile = before.profile.copy()
        profile.update(dtype="int32", nodata=-9999, count=1, compress="deflate")
        with rasterio.open(out_dir / "transition_codes.tif", "w", **profile) as dst:
            dst.write(transitions, 1)

        transitions_df = transition_table(before_arr, after_arr, valid)
        transitions_df.to_csv(out_dir / "transition_matrix_long.csv", index=False)

        pixel_area = abs(before.transform.a * before.transform.e)
        summary_df = class_summary(before_arr, after_arr, valid, pixel_area)
        summary_df.to_csv(out_dir / "class_change_summary.csv", index=False)

        changed_pixels = int(((before_arr != after_arr) & valid).sum())
        valid_pixels = int(valid.sum())
        pct_changed = 100.0 * changed_pixels / valid_pixels if valid_pixels else np.nan

        qa = pd.DataFrame(
            [
                {
                    "crs": str(before.crs),
                    "width": before.width,
                    "height": before.height,
                    "pixel_width": before.transform.a,
                    "pixel_height": abs(before.transform.e),
                    "valid_pixels": valid_pixels,
                    "changed_pixels": changed_pixels,
                    "percent_changed": pct_changed,
                }
            ]
        )
        qa.to_csv(out_dir / "qa_summary.csv", index=False)

        print(f"Wrote outputs to: {out_dir}")
        print(f"Valid pixels: {valid_pixels:,}")
        print(f"Changed pixels: {changed_pixels:,} ({pct_changed:.2f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two classified land-cover rasters.")
    parser.add_argument("--before", type=Path, required=True, help="Earlier classified raster")
    parser.add_argument("--after", type=Path, required=True, help="Later classified raster")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.before, args.after, args.out_dir)
