# Copyright (c) 2025 takotime808
"""
Extract only the text inside the border of a photographed wedding
invitation, discarding the border pattern and blank paper.

Uses the same card-locating pipeline as extract_border.py (deskew, shadow
trim, border-thickness detection) to find the interior region, then keeps
just the ink pixels within it. Output is the same canvas size as
extract_border.py's output, so the two layers line up for the 3D model.

Speckle noise (isolated pixels from paper texture) below --min-area is
dropped before saving.
"""

import argparse

import cv2
import numpy as np

from card_utils import clean_ink_mask, ink_mask, locate_card


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("input", nargs="?", default="wedding_invite.png")
    parser.add_argument("-o", "--output", default="wedding_invite_text.png")
    parser.add_argument("--top", type=int, help="override border thickness (px)")
    parser.add_argument("--bottom", type=int, help="override border thickness (px)")
    parser.add_argument("--left", type=int, help="override border thickness (px)")
    parser.add_argument("--right", type=int, help="override border thickness (px)")
    parser.add_argument("--pad", type=int, default=4, help="extra px kept past the detected border band")
    parser.add_argument("--min-area", type=int, default=2, help="drop ink specks smaller than this many px (edge-touching shadow residue is always dropped)")
    parser.add_argument("--keep-color", action="store_true", help="keep original ink color instead of solid black")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise SystemExit(f"Could not read image: {args.input}")

    trimmed, top, bottom, left, right = locate_card(
        img,
        pad=args.pad,
        overrides=(args.top, args.bottom, args.left, args.right),
    )
    th, tw = trimmed.shape[:2]
    print(f"Interior region (px) — top:{top} bottom:{bottom} left:{left} right:{right}")

    interior = trimmed[top:th - bottom, left:tw - right]
    igray = cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY)
    iink = ink_mask(igray).astype(np.uint8) * 255
    iink = clean_ink_mask(iink, min_area=args.min_area)

    mask = np.zeros((th, tw), dtype=np.uint8)
    mask[top:th - bottom, left:tw - right] = iink

    if args.keep_color:
        rgba = cv2.cvtColor(trimmed, cv2.COLOR_BGR2BGRA)
    else:
        rgba = np.zeros((th, tw, 4), dtype=np.uint8)
    rgba[:, :, 3] = mask

    cv2.imwrite(args.output, rgba)
    print(f"Saved text-only image to {args.output}")


if __name__ == "__main__":
    main()
