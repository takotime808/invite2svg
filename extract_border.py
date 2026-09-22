# Copyright (c) 2025 takotime808
"""
Extract only the decorative border pattern from a photographed wedding
invitation, discarding the card interior (text) and the background it was
photographed against.

Pipeline:
  1. Find the card in the photo (largest bright rectangular region) and
     perspective-warp it to a straight, cropped image.
  2. Trim the soft photographic drop-shadow that hugs the card's edges.
  3. Profile ink density row-by-row / column-by-column near each edge to
     find how thick the printed border band is on that side.
  4. Keep only the pixels within that border band; make everything else
     (the blank interior + text) transparent.

Works on any card with a border pattern hugging the edges and a mostly
blank/text interior. If auto-detection misjudges a side, override it with
--top/--bottom/--left/--right (in pixels, measured on the deskewed card).
"""

import argparse

import cv2
import numpy as np

from card_utils import locate_card


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("input", nargs="?", default="wedding_invite.png")
    parser.add_argument("-o", "--output", default="wedding_invite_border.png")
    parser.add_argument("--top", type=int, help="override border thickness (px)")
    parser.add_argument("--bottom", type=int, help="override border thickness (px)")
    parser.add_argument("--left", type=int, help="override border thickness (px)")
    parser.add_argument("--right", type=int, help="override border thickness (px)")
    parser.add_argument("--pad", type=int, default=4, help="extra px kept past the detected band")
    parser.add_argument("--enter-thresh", type=float, default=0.15)
    parser.add_argument("--exit-thresh", type=float, default=0.10)
    parser.add_argument("--exit-run", type=int, default=20)
    parser.add_argument("--max-frac", type=float, default=0.3, help="cap search to this fraction of each dimension")
    parser.add_argument("--debug", action="store_true", help="also save the deskewed card and a boundary overlay")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise SystemExit(f"Could not read image: {args.input}")

    trimmed, top, bottom, left, right = locate_card(
        img,
        pad=args.pad,
        enter_thresh=args.enter_thresh,
        exit_thresh=args.exit_thresh,
        exit_run=args.exit_run,
        max_frac=args.max_frac,
        overrides=(args.top, args.bottom, args.left, args.right),
    )
    th, tw = trimmed.shape[:2]
    print(f"Detected border thickness (px) — top:{top} bottom:{bottom} left:{left} right:{right}")

    mask = np.zeros((th, tw), dtype=np.uint8)
    mask[:top, :] = 255
    mask[th - bottom:, :] = 255
    mask[:, :left] = 255
    mask[:, tw - right:] = 255

    rgba = cv2.cvtColor(trimmed, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    cv2.imwrite(args.output, rgba)
    print(f"Saved border-only image to {args.output}")

    if args.debug:
        deskewed_path = args.output.rsplit(".", 1)[0] + "_deskewed.png"
        cv2.imwrite(deskewed_path, trimmed)

        overlay = trimmed.copy()
        cv2.rectangle(overlay, (left, top), (tw - right, th - bottom), (0, 0, 255), 3)
        overlay_path = args.output.rsplit(".", 1)[0] + "_overlay.png"
        cv2.imwrite(overlay_path, overlay)
        print(f"Saved debug images: {deskewed_path}, {overlay_path}")


if __name__ == "__main__":
    main()
