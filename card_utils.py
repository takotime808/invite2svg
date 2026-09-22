# Copyright (c) 2025 takotime808
"""Shared image-processing pipeline for locating a photographed card and its
printed border band. Used by extract_border.py and extract_text.py."""

import cv2
import numpy as np


def order_corners(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def deskew_card(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Could not find a card-shaped region in the image.")
    largest = max(contours, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(largest))
    src = order_corners(box)

    width = int(max(np.linalg.norm(src[2] - src[3]), np.linalg.norm(src[1] - src[0])))
    height = int(max(np.linalg.norm(src[1] - src[2]), np.linalg.norm(src[0] - src[3])))
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (width, height))


def ink_mask(gray):
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray < thresh


def find_shadow_trim(density, low=0.05, run=5):
    """How many pixels of soft drop-shadow hug this edge before the flat paper begins."""
    for i in range(len(density) - run):
        if np.all(density[i:i + run] <= low):
            return i
    return 0


def smooth(density, window=5):
    kernel = np.ones(window) / window
    return np.convolve(density, kernel, mode="same")


def find_border_thickness(density, enter_thresh, exit_thresh, exit_run, max_frac):
    """How many pixels from this edge the printed border pattern extends."""
    n = len(density)
    max_search = int(n * max_frac)
    started = False
    end = 0
    i = 0
    while i < max_search:
        if density[i] > enter_thresh:
            started = True
            end = i
        elif started:
            j = i
            while j < n and density[j] <= exit_thresh:
                j += 1
            if j - i >= exit_run:
                break
        i += 1
    return end + 1


def clean_ink_mask(ink, min_area=2, edge_margin=20):
    """Drop noise from a bilevel ink mask (0/255, uint8): specks within
    edge_margin px of the canvas edge (photographic shadow residue that
    survived shadow-trimming, which fades out gradually rather than
    stopping exactly on the boundary pixel) and specks smaller than
    min_area (sensor/paper-texture dust). Genuine text marks (dots,
    hyphens, accents) sit well inset from the edge, so they're left alone.
    """
    h, w = ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    x, y, bw, bh, area = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3], stats[:, 4]
    near_edge = (x <= edge_margin) | (y <= edge_margin) | (x + bw >= w - edge_margin) | (y + bh >= h - edge_margin)
    keep = np.ones(n, dtype=bool)
    keep[0] = False
    keep &= ~near_edge
    keep &= area >= min_area
    return (keep[labels] * 255).astype(np.uint8)


def locate_card(img, pad=4, enter_thresh=0.15, exit_thresh=0.10, exit_run=20, max_frac=0.3,
                 overrides=(None, None, None, None)):
    """Deskew the card and measure its border-band thickness on each side.

    Returns (trimmed_card_bgr, top, bottom, left, right).
    """
    card = deskew_card(img)
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    ink = ink_mask(gray)

    row_density = ink.mean(axis=1)
    col_density = ink.mean(axis=0)
    h, w = ink.shape

    top_trim = find_shadow_trim(row_density)
    bottom_trim = find_shadow_trim(row_density[::-1])
    left_trim = find_shadow_trim(col_density)
    right_trim = find_shadow_trim(col_density[::-1])

    trimmed = card[top_trim:h - bottom_trim, left_trim:w - right_trim]
    tgray = cv2.cvtColor(trimmed, cv2.COLOR_BGR2GRAY)
    tink = ink_mask(tgray)

    row_d = smooth(tink.mean(axis=1))
    col_d = smooth(tink.mean(axis=0))

    top_o, bottom_o, left_o, right_o = overrides

    def thickness(density, override):
        if override is not None:
            return override
        return find_border_thickness(density, enter_thresh, exit_thresh, exit_run, max_frac) + pad

    top = thickness(row_d, top_o)
    bottom = thickness(row_d[::-1], bottom_o)
    left = thickness(col_d, left_o)
    right = thickness(col_d[::-1], right_o)

    return trimmed, top, bottom, left, right
