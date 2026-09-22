# Copyright (c) 2025 takotime808
"""
Extract handwritten/inked text from a photo with no printed border and no
clean background to deskew against (e.g. marker on cardboard, a sticky
note, a whiteboard) -- cases where extract_text.py's card-locating pipeline
(card_utils.locate_card) picks the wrong region, either because there's no
border band to key off of or because uneven lighting across the subject
confuses the single global Otsu threshold used to find the card boundary.

Uses a per-pixel adaptive threshold instead of one global threshold, so it
stays robust to lighting gradients across the frame. A light morphological
open plus a connected-component size filter drops paper/cardboard grain
speckle while keeping full ink strokes. No deskewing or border detection
is attempted -- crop the input beforehand if the frame has background/
clutter you don't want picked up as noise.
"""

import argparse

import cv2
import numpy as np


def extract_ink(gray, block_size=35, c=12, open_size=3, min_area=40):
    if block_size % 2 == 0:
        block_size += 1
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block_size, c,
    )
    if open_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[:, 4]
    keep = np.ones(n, dtype=bool)
    keep[0] = False
    keep &= areas >= min_area
    return (keep[labels] * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="text_freeform.png")
    parser.add_argument("--block-size", type=int, default=35, help="adaptive-threshold neighborhood size (px, odd)")
    parser.add_argument("--c", type=float, default=12, help="adaptive-threshold constant subtracted from the local mean; raise to keep less ink, lower to keep more")
    parser.add_argument("--open-size", type=int, default=3, help="morphological opening kernel size (px); 0 disables")
    parser.add_argument("--min-area", type=int, default=40, help="drop ink specks smaller than this many px (paper/cardboard grain)")
    parser.add_argument("--keep-color", action="store_true", help="keep original ink color instead of solid black")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise SystemExit(f"Could not read image: {args.input}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = extract_ink(gray, args.block_size, args.c, args.open_size, args.min_area)

    if args.keep_color:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[:, :, 3] = mask

    cv2.imwrite(args.output, rgba)
    print(f"Saved text-only image to {args.output}")


if __name__ == "__main__":
    main()
