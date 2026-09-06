#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final submission blender.

Final = 0.60 * GC_final + 0.40 * GC_atmp3b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gc-final", type=Path, required=True)
    parser.add_argument("--gc-atmp3b", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/final_ensemble.csv"),
    )
    parser.add_argument("--weight-final", type=float, default=0.60)
    parser.add_argument("--weight-atmp3b", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not np.isclose(args.weight_final + args.weight_atmp3b, 1.0):
        raise ValueError("Ensemble weights must sum to 1.0")

    a = pd.read_csv(args.gc_final)
    b = pd.read_csv(args.gc_atmp3b)

    if list(a.columns) != list(b.columns):
        raise ValueError("Submission columns do not match.")
    if len(a) != len(b):
        raise ValueError("Submission row counts do not match.")
    if "nins" not in a.columns:
        raise ValueError("Expected target column 'nins'.")

    out = a.copy()
    out["nins"] = (
        args.weight_final * a["nins"].to_numpy(dtype=float)
        + args.weight_atmp3b * b["nins"].to_numpy(dtype=float)
    )
    out["nins"] = np.clip(out["nins"], 0.0, None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, float_format="%.5f")

    print(
        f"Saved: {args.output.resolve()} "
        f"(GC_final={args.weight_final:.0%}, "
        f"GC_atmp3b={args.weight_atmp3b:.0%})"
    )


if __name__ == "__main__":
    main()
