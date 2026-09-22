# Copyright (c) 2025 takotime808
"""
Streamlit page that extrudes a 3"x5" base plate and the invite's SVG
artwork into a single watertight 3D solid, either raised above the plate
(emboss) or cut into it (engrave/deboss), ready to export as an STL for
3D printing.

Run with:
    streamlit run streamlit_app.py
(this page is auto-discovered from the sidebar)
"""

from pathlib import Path

import streamlit as st

from card3d_utils import (
    MM_PER_INCH,
    build_invite_mesh,
    mesh_to_plotly_figure,
    mesh_to_stl_bytes,
    parse_svg_polygons,
)

st.set_page_config(page_title="3D Wedding Invite", page_icon="🧊", layout="wide")

st.title("3D Wedding Invite")
st.markdown("**NOTE:** Web renderings looks need work. Downloaded SVGs look good.")
st.caption(
    "Extrudes a 3\"x5\" base plate and lays the invite's SVG artwork on top of it, "
    "either raised (emboss) or cut in (engrave), as a single watertight solid ready to 3D print."
)

uploaded_svg = st.file_uploader("Upload an SVG (optional)", type=["svg"])

svg_text = None
svg_name = "wedding_invite_3d"
if uploaded_svg is not None:
    svg_text = uploaded_svg.read().decode("utf-8")
    svg_name = Path(uploaded_svg.name).stem
    st.caption(f"Using uploaded file: {uploaded_svg.name}")
elif "generated_svg" in st.session_state:
    svg_text = st.session_state["generated_svg"]
    svg_name = st.session_state.get("generated_svg_name", svg_name)
    st.caption("Using the SVG generated on the **Photo → SVG** page.")
else:
    st.info("Upload an SVG above, or generate one first on the **Photo → SVG** page.")

with st.sidebar:
    st.header("Plate size")
    plate_w_in = st.number_input("Width (in)", min_value=1.0, max_value=12.0, value=5.0, step=0.5)
    plate_h_in = st.number_input("Height (in)", min_value=1.0, max_value=12.0, value=3.0, step=0.5)
    margin_mm = st.slider("Artwork margin (mm)", 0.0, 15.0, 3.0, step=0.5)

    st.header("Extrusion")
    base_thickness_mm = st.slider("Base plate thickness (mm)", 1.0, 10.0, 3.0, step=0.5)
    feature_height_mm = st.slider("Emboss/engrave depth (mm)", 0.2, 5.0, 1.0, step=0.1)
    mode_label = st.radio(
        "Artwork direction",
        ["Raised (emboss)", "Indented (engrave / deboss)"],
        help="Raised sticks the SVG artwork up out of the plate. Indented cuts it into the plate as a cavity.",
    )
    mode = "raised" if mode_label.startswith("Raised") else "indented"

if svg_text:
    if st.button("Generate 3D Model", type="primary"):
        with st.spinner("Parsing SVG and building the 3D solid..."):
            try:
                polygons = parse_svg_polygons(svg_text)
                if not polygons:
                    st.error("Could not find any filled shapes in that SVG.")
                    st.stop()

                plate_width_mm = plate_w_in * MM_PER_INCH
                plate_height_mm = plate_h_in * MM_PER_INCH

                mesh = build_invite_mesh(
                    polygons,
                    plate_width_mm=plate_width_mm,
                    plate_height_mm=plate_height_mm,
                    base_thickness_mm=base_thickness_mm,
                    feature_height_mm=feature_height_mm,
                    mode=mode,
                    margin_mm=margin_mm,
                )
            except ValueError as e:
                st.error(str(e))
                st.stop()

        st.success(
            f"Built a {plate_w_in:g}\"x{plate_h_in:g}\" solid "
            f"({'watertight' if mesh.is_watertight else 'NOT watertight — check the SVG'})."
        )

        st.plotly_chart(mesh_to_plotly_figure(mesh), use_container_width=True)

        st.download_button(
            "Download STL",
            data=mesh_to_stl_bytes(mesh),
            file_name=f"{svg_name}_3d.stl",
            mime="model/stl",
        )
