#!/usr/bin/env python3
"""Save a PNG grid of obs_rgb/rgb from a scripts/export_demos.py file for a visual check.

Rows are the first and last frame, columns the first few demos. Uses only h5py,
numpy and the standard library (no PIL on the Mac venv).
"""

import argparse
import struct
import zlib
from pathlib import Path

import h5py
import numpy as np


def write_png(path: Path, image: np.ndarray) -> None:
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6))
                     + chunk(b"IEND", b""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path)
    parser.add_argument("png", type=Path)
    parser.add_argument("--demos", type=int, default=5)
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as file:
        count = min(args.demos, len(file.keys()))
        frames = [file[f"traj_{i}/obs_rgb/rgb"] for i in range(count)]
        rows = [np.concatenate([f[index][..., :3] for f in frames], axis=1) for index in (0, -1)]
    grid = np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.uint8)
    write_png(args.png, grid)
    print(f"{args.png}: {grid.shape[1]}x{grid.shape[0]}, demos 0-{count - 1}, first and last frame")


if __name__ == "__main__":
    main()
