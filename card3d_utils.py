# Copyright (c) 2025 takotime808
"""SVG -> 3D solid pipeline for the 3D wedding invite page.

Parses an arbitrary SVG (in particular, potrace's output from
photo_to_svg.py, which wraps its paths in a `<g transform="...">` that
must be resolved, not just each path's own `d` attribute) into filled
polygons with holes, fits them onto a flat base plate, and extrudes the
artwork either raised above the plate (emboss) or cut into it (deboss).

The final solid is assembled directly face-by-face (background top cap,
base sides/bottom, per-feature walls/caps) rather than via a 3D boolean:
a real 3D CSG library (e.g. manifold3d) internally re-triangulates any
face it modifies, and for a "big flat face with hundreds of small holes
punched in it" (exactly what a plate covered in embossed/engraved text
looks like) that re-triangulation reliably produces long spurious sliver
triangles running across otherwise-empty regions of the plate -- a real
but purely cosmetic defect (their area nets to zero, so it doesn't affect
volume/printability), but a bad one to ship in an interactive viewer.
Building the boundary ourselves sidesteps it entirely, and lets us use
a quality-constrained triangulation (`triangle`, with Steiner points)
everywhere, which avoids the same defect in the input geometry too.
"""

import io

import numpy as np
import svgelements as se
import trimesh
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

MM_PER_INCH = 25.4

# Quality-constrained Delaunay triangulation: p = respect input segments,
# q30 = no angle below 30 degrees (avoids slivers), a2 = cap triangle area
# at 2mm^2 (avoids long edges across otherwise-empty regions), Y = never
# add new points on the boundary (so two faces sharing a boundary ring,
# triangulated separately, still end up with matching, weldable edges).
_TRIANGLE_ARGS = "pq30a2Y"
_BOUNDARY_ONLY_ARGS = "pY"


def _as_polygons(geom):
    """Flatten a shapely geometry (possibly a Multi*/GeometryCollection
    result of a 2D boolean op or a buffer(0) validity fix) into a flat
    list of Polygons."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        out = []
        for g in geom.geoms:
            out.extend(_as_polygons(g))
        return out
    return []


def parse_svg_polygons(svg_text, samples_per_curve=8):
    """Parse all <path> elements of an SVG into a list of shapely Polygons
    (with holes), in the SVG's user-space coordinates, with all inherited
    transforms (group transforms, viewBox, etc.) already applied.

    Holes (e.g. the counter of a letter "o") are resolved by nesting depth
    rather than by trusting winding direction, since that varies by SVG
    producer.
    """
    svg = se.SVG.parse(io.StringIO(svg_text))

    rings = []
    for element in svg.elements():
        if isinstance(element, se.Path) and len(element) > 0:
            rings.extend(_flatten_path(element, samples_per_curve))

    if not rings:
        return []

    raw_polys = []
    for ring in rings:
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        raw_polys.append(poly)

    areas = [p.area for p in raw_polys]
    # a point just inside each ring's own boundary, rather than shapely's
    # representative_point(): for a ring like the outer boundary of a
    # letter "O", representative_point() often lands in the *middle* of
    # the shape -- which is exactly where that letter's own counter/hole
    # ring sits, making the hole falsely "contain" its own parent ring.
    # Hugging the boundary keeps the test point out of any child hole.
    reps = [_point_near_boundary(ring, poly) for ring, poly in zip(rings, raw_polys)]

    containers = [[] for _ in raw_polys]
    for i, rep in enumerate(reps):
        if rep is None:
            continue
        for j, poly in enumerate(raw_polys):
            if i == j or poly.is_empty:
                continue
            if poly.contains(rep):
                containers[i].append(j)
    depths = [len(c) for c in containers]

    polygons = []
    for i, depth in enumerate(depths):
        if depth % 2 != 0:
            continue  # this ring is a hole; it gets attached to its parent below
        holes = []
        for k, dk in enumerate(depths):
            if dk != depth + 1:
                continue
            same_depth_containers = [c for c in containers[k] if depths[c] == depth]
            if same_depth_containers and min(same_depth_containers, key=lambda c: areas[c]) == i:
                holes.append(rings[k])
        poly = Polygon(rings[i], holes)
        if not poly.is_valid:
            # a hole touching/crossing the exterior at a point ("keyhole"
            # topology) is valid under SVG's fill rules but not under the
            # strict simple-polygon rules shapely/GEOS enforce; repair it.
            poly = poly.buffer(0)
        polygons.extend(p for p in _as_polygons(poly) if p.area > 1e-9)

    return polygons


def _point_near_boundary(ring_coords, poly):
    """A point just inside `poly` (built from `ring_coords` alone), close
    to its boundary -- unlike Polygon.representative_point(), which for a
    ring shaped like a thin annulus (e.g. a letter "O") tends to land
    near the centroid, i.e. inside that same letter's own counter/hole.
    """
    coords = list(ring_coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 3:
        return None
    span = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 1.0)
    eps = 1e-4 * span
    for i in range(min(len(coords), 20)):
        p0, p1 = coords[i], coords[(i + 1) % len(coords)]
        edge = p1 - p0
        elen = np.linalg.norm(edge)
        if elen < 1e-9:
            continue
        mid = (p0 + p1) / 2
        normal = np.array([-edge[1], edge[0]]) / elen
        for sign in (1, -1):
            candidate = Point(mid + sign * normal * eps)
            if poly.contains(candidate):
                return candidate
    return poly.representative_point() if not poly.is_empty else None


def _flatten_path(path, samples_per_curve):
    """Flatten one svgelements Path (possibly containing multiple
    subpaths) into a list of closed point-rings, sampling curves."""
    rings = []
    current = []
    for seg in path.segments(transformed=True):
        if isinstance(seg, se.Move):
            if len(current) >= 3:
                rings.append(current)
            current = [(seg.end.x, seg.end.y)]
        elif isinstance(seg, se.Close):
            if len(current) >= 3:
                rings.append(current)
            current = []
        elif isinstance(seg, se.Line):
            current.append((seg.end.x, seg.end.y))
        else:
            for i in range(1, samples_per_curve + 1):
                pt = seg.point(i / samples_per_curve)
                current.append((pt.x, pt.y))
    if len(current) >= 3:
        rings.append(current)
    return rings


def fit_polygons_to_plate(polygons, plate_width_mm, plate_height_mm, margin_mm):
    """Rescale/translate polygons (SVG space, y-down) to sit centered
    within [0, plate_width_mm] x [0, plate_height_mm] (mesh space, y-up),
    preserving aspect ratio and leaving `margin_mm` clear on each side."""
    minx, miny, maxx, maxy = _combined_bounds(polygons)
    bbox_w, bbox_h = maxx - minx, maxy - miny
    if bbox_w <= 0 or bbox_h <= 0:
        raise ValueError("SVG artwork has zero-size bounding box.")

    available_w = plate_width_mm - 2 * margin_mm
    available_h = plate_height_mm - 2 * margin_mm
    if available_w <= 0 or available_h <= 0:
        raise ValueError("Margin is too large for the plate size.")

    scale = min(available_w / bbox_w, available_h / bbox_h)
    offset_x = (plate_width_mm - bbox_w * scale) / 2
    offset_y = (plate_height_mm - bbox_h * scale) / 2

    def transform_ring(coords):
        out = []
        for x, y in coords:
            nx = (x - minx) * scale
            ny = (y - miny) * scale
            out.append((nx + offset_x, plate_height_mm - offset_y - ny))
        return out

    fitted = []
    for poly in polygons:
        exterior = transform_ring(poly.exterior.coords)
        holes = [transform_ring(ring.coords) for ring in poly.interiors]
        fitted.append(Polygon(exterior, holes))
    return fitted


def _combined_bounds(polygons):
    minx = min(p.bounds[0] for p in polygons)
    miny = min(p.bounds[1] for p in polygons)
    maxx = max(p.bounds[2] for p in polygons)
    maxy = max(p.bounds[3] for p in polygons)
    return minx, miny, maxx, maxy


def _triangulate_capped(polygon, z, upward):
    """Triangulate a shapely Polygon into a flat 3D cap at height `z`,
    oriented so its normal points +z (upward=True) or -z (upward=False)."""
    verts2d, faces = trimesh.creation.triangulate_polygon(
        polygon, engine="triangle", triangle_args=_TRIANGLE_ARGS
    )
    if len(faces) == 0:
        return None
    verts3d = np.column_stack([verts2d, np.full(len(verts2d), z)])
    a, b, c = verts2d[faces[0][0]], verts2d[faces[0][1]], verts2d[faces[0][2]]
    is_ccw = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0
    if is_ccw != upward:
        faces = faces[:, ::-1]
    return trimesh.Trimesh(vertices=verts3d, faces=faces, process=False)


def _drop_cap(mesh, z, flip):
    """Remove the faces of an extrude_polygon() solid lying entirely at
    height `z` (its top or bottom cap), keeping the rest (the other cap
    plus the side walls). `flip` reverses the winding of what's kept, to
    turn an outward-facing "peg" into an inward-facing "cavity"."""
    at_z = np.all(np.isclose(mesh.vertices[mesh.faces][:, :, 2], z, atol=1e-6), axis=1)
    faces = mesh.faces[~at_z]
    if flip:
        faces = faces[:, ::-1]
    return trimesh.Trimesh(vertices=mesh.vertices, faces=faces, process=False)


def build_invite_mesh(
    polygons,
    plate_width_mm,
    plate_height_mm,
    base_thickness_mm,
    feature_height_mm,
    mode="raised",
    margin_mm=3.0,
):
    """Build the final watertight solid: a plate with the given polygons
    either raised above it (mode="raised") or cut into it (mode="indented").
    """
    if not polygons:
        raise ValueError("No artwork polygons to extrude.")
    if mode not in ("raised", "indented"):
        raise ValueError(f"Unknown mode: {mode!r}")

    fitted = fit_polygons_to_plate(polygons, plate_width_mm, plate_height_mm, margin_mm)
    fitted = [p for p in fitted if not p.is_empty and p.area > 1e-9]
    if not fitted:
        raise ValueError("Artwork polygons produced no extrudable geometry.")

    # merge any touching/overlapping shapes (e.g. a border pattern that
    # touches itself, or text touching the border) into one canonical
    # list -- two shapes that are individually valid but merely touch at
    # a point are still invalid as two holes of the same polygon, so they
    # need to become one shape before anything downstream uses them.
    features = _as_polygons(unary_union(fitted))
    features = [f if f.is_valid else f.buffer(0) for f in features]
    features = [f for g in features for f in _as_polygons(g) if f.area > 1e-9]

    top_z = base_thickness_mm
    plate_rect = [(0, 0), (plate_width_mm, 0), (plate_width_mm, plate_height_mm), (0, plate_height_mm)]

    # the plate's top surface, everywhere a feature *isn't*. Computed via
    # an actual 2D boolean (not by handing shapely's Polygon() constructor
    # a list of feature rings as holes directly) because GEOS may still
    # consider two *individually* valid, merely-touching feature exteriors
    # invalid as two holes of the same polygon. Because a boolean op like
    # this can nudge boundary coordinates by a few microns even where nothing
    # actually intersects, this recomputed hole ring -- not the original
    # feature ring -- becomes the canonical boundary used below to build
    # the matching peg, so the two always share exact, weldable coordinates.
    footprint = unary_union([Polygon(f.exterior) for f in features])
    top_cap_main = Polygon(plate_rect).difference(footprint)

    top_cap_pieces = _as_polygons(top_cap_main)

    # one small "island" piece per feature-internal hole (e.g. the counter
    # of a letter "o", which stays at full plate height since it isn't
    # part of that feature's own filled shape) -- untouched by the boolean
    # above, so these coordinates are already exact.
    top_pieces = list(top_cap_pieces)
    top_pieces.extend(Polygon(list(hole.coords)) for f in features for hole in f.interiors)

    parts = []
    for raw_piece in top_pieces:
        if not raw_piece.is_valid:
            raw_piece = raw_piece.buffer(0)
        for piece in _as_polygons(raw_piece):
            if piece.is_empty or piece.area < 1e-9:
                continue
            cap = _triangulate_capped(piece, top_z, upward=True)
            if cap is not None:
                parts.append(cap)

    # base plate: bottom + side walls only (no top cap -- that's `parts` above)
    base_solid = trimesh.creation.extrude_polygon(
        Polygon(plate_rect), height=base_thickness_mm,
        engine="triangle", triangle_args=_BOUNDARY_ONLY_ARGS,
    )
    parts.append(_drop_cap(base_solid, top_z, flip=False))

    # each peg's outer boundary is one of top_cap_main's hole rings (exact
    # same coordinates the cap uses -- see above), carrying whichever
    # original features' own internal holes (letter counters) fall inside
    # it, so two features that got merged into one hole above also become
    # one peg. `f.representative_point()` is safe here (unlike the parsing
    # step) since `f` is already a proper Polygon with its holes attached.
    hole_rings = [ring for piece in top_cap_pieces for ring in piece.interiors]
    for hole_ring in hole_rings:
        outer_coords = list(hole_ring.coords)
        outer = Polygon(outer_coords)
        if outer.area < 1e-9:
            continue
        internal_holes = [
            list(hole.coords)
            for f in features
            if outer.contains(f.representative_point())
            for hole in f.interiors
        ]
        poly = Polygon(outer_coords, internal_holes)
        peg = trimesh.creation.extrude_polygon(poly, height=feature_height_mm, engine="triangle", triangle_args=_TRIANGLE_ARGS)
        if mode == "raised":
            # sits on top of the plate: keep the walls + top cap, drop the
            # (interior, touching the background cap) bottom
            peg.apply_translation([0, 0, top_z])
            parts.append(_drop_cap(peg, top_z, flip=False))
        else:
            # cut down from the top: keep the walls + bottom cap (now the
            # cavity floor, flipped to face up into the cavity), drop the
            # top (that's where the background cap's hole already is)
            peg.apply_translation([0, 0, top_z - feature_height_mm])
            parts.append(_drop_cap(peg, top_z, flip=True))

    result = trimesh.util.concatenate(parts)
    result.merge_vertices(digits_vertex=8)
    result.remove_unreferenced_vertices()
    return result


def mesh_to_plotly_figure(mesh):
    import plotly.graph_objects as go

    v, f = mesh.vertices, mesh.faces
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color="#f2ead6",
                flatshading=False,
                lighting=dict(ambient=0.55, diffuse=0.7, specular=0.35, roughness=0.6, fresnel=0.1),
                lightposition=dict(x=200, y=400, z=500),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="mm", yaxis_title="mm", zaxis_title="mm",
            camera=dict(eye=dict(x=1.4, y=-1.6, z=5.5)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
    )
    return fig


def mesh_to_stl_bytes(mesh):
    return mesh.export(file_type="stl")
