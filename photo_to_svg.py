"""
Convert a photographed invitation into a single vector SVG that fully
captures the border pattern and all text, corrected for the camera's
skew/rotation — ready to extrude onto a base rectangle for the 3D model.

Pipeline:
  1. Deskew the card and trim the photographic drop-shadow (card_utils),
     giving the corrected orientation and a clean crop.
  2. Threshold the whole card into a bilevel ink/paper bitmap (border +
     text combined) and drop small speckle noise.
  3. Hand the bitmap to potrace, which fits smooth vector paths to it.

Requires the `potrace` command-line tool (e.g. `brew install potrace`).
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from card_utils import clean_ink_mask, deskew_card, find_shadow_trim, ink_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("input", nargs="?", default="wedding_invite.png")
    parser.add_argument("-o", "--output", default="wedding_invite.svg")
    parser.add_argument("--min-area", type=int, default=2, help="drop ink specks smaller than this many px (edge-touching shadow residue is always dropped)")
    parser.add_argument("--turdsize", type=int, default=2, help="potrace speckle suppression (output curves)")
    parser.add_argument("--alphamax", type=float, default=1.0, help="potrace corner threshold")
    parser.add_argument("--opttolerance", type=float, default=0.2, help="potrace curve-fit tolerance")
    parser.add_argument("--keep-bitmap", help="also save the intermediate bilevel bitmap to this path")
    args = parser.parse_args()

    if shutil.which("potrace") is None:
        raise SystemExit("potrace not found on PATH — install it, e.g. `brew install potrace`.")

    img = cv2.imread(args.input)
    if img is None:
        raise SystemExit(f"Could not read image: {args.input}")

    card = deskew_card(img)
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    ink = ink_mask(gray)

    row_density = ink.mean(axis=1)
    col_density = ink.mean(axis=0)
    h, w = ink.shape
    top = find_shadow_trim(row_density)
    bottom = find_shadow_trim(row_density[::-1])
    left = find_shadow_trim(col_density)
    right = find_shadow_trim(col_density[::-1])

    trimmed = card[top:h - bottom, left:w - right]
    tgray = cv2.cvtColor(trimmed, cv2.COLOR_BGR2GRAY)
    tink = ink_mask(tgray).astype(np.uint8) * 255
    tink = clean_ink_mask(tink, min_area=args.min_area)

    # potrace traces dark pixels as foreground on a white background
    bitmap = np.where(tink > 0, 0, 255).astype(np.uint8)

    if args.keep_bitmap:
        cv2.imwrite(args.keep_bitmap, bitmap)

    with tempfile.TemporaryDirectory() as tmp:
        bmp_path = Path(tmp) / "ink.bmp"
        cv2.imwrite(str(bmp_path), bitmap)

        cmd = [
            "potrace", str(bmp_path),
            "--svg",
            "--group",
            "--turdsize", str(args.turdsize),
            "--alphamax", str(args.alphamax),
            "--opttolerance", str(args.opttolerance),
            "-o", args.output,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"potrace failed:\n{result.stderr}")

    print(f"Card corrected to {tink.shape[1]}x{tink.shape[0]}px, saved vector trace to {args.output}")


if __name__ == "__main__":
    main()
