# Image to SVG

### Wedding invite layer extraction

Turns a photographed wedding invitation into assets for a 3D model —
either as separate transparent PNG layers (border, text) or as a single
vector SVG capturing both — corrected for the camera's skew/rotation.

All three scripts share `card_utils.py`, which deskews the card, trims
the photographic drop-shadow, and profiles ink density near each edge to
find the border band thickness.


## Usage (python)

```sh
python extract_border.py wedding_invite.png -o wedding_invite_border.png
python extract_text.py   wedding_invite.png -o wedding_invite_text.png
python photo_to_svg.py   wedding_invite.png -o wedding_invite.svg
```

- **`extract_border.py`** keeps only the printed border band (transparent elsewhere).
- **`extract_text.py`** keeps only the ink inside the border, rendered as solid black on transparent (use `--keep-color` to keep the original ink color instead). Its output lines up with `extract_border.py`'s — same canvas size, since both measure the same border thickness.
- **`photo_to_svg.py`** vectorizes the whole deskewed card (border + text combined) into one SVG via [potrace](http://potrace.sourceforge.net/) (`brew install potrace`), ready to extrude onto a base rectangle.

Common options:
- `--top / --bottom / --left / --right` — override a side's border thickness (px) if auto-detection misjudges it (`extract_border.py`, `extract_text.py`)
- `--pad` — extra px kept past the detected border band
- `--min-area` — drop ink specks smaller than this many px; edge-touching photographic shadow residue is always dropped regardless of size
- `--debug` (`extract_border.py`) — also save the deskewed card and a boundary-overlay image for verification
- `--turdsize` / `--alphamax` / `--opttolerance` (`photo_to_svg.py`) — passed through to potrace to tune curve smoothing

## UI

`streamlit_app.py` wraps the same pipeline in a browser UI: upload a photo,
pick full-card / border-only / text-only, tune the same options via sliders,
and download the resulting SVG.

```sh
pip install -r requirements.txt   # streamlit, opencv-python-headless, numpy
streamlit run streamlit_app.py
```

Requires `potrace` on PATH, same as `photo_to_svg.py`. Deploying to Streamlit
Community Cloud picks up `packages.txt` to install it automatically.
