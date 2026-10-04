"""
Nissan Silvia S15 Spec R - procedural Blender model for Roblox
================================================================

Builds a detailed, Roblox-ready Nissan Silvia S15 Spec R in Blender from
real-world dimensions, then exports it for Roblox Studio's 3D Importer.

Run it either way:
    blender --background --python s15_spec_r.py              # build + export
    blender -b -P s15_spec_r.py -- --out ./export            # choose the folder
    blender -b -P s15_spec_r.py -- --no-export --render --open --samples 64
                                                             # preview renders
    blender -b -P s15_spec_r.py -- --body loft               # lofted body
    blender -b -P s15_spec_r.py -- --prepare-body raw.glb body/s15_body.glb
                                         # turn a raw image-to-3D model into
                                         # the body file the build loads
or open Blender > Scripting tab > open this file > Run Script (builds into a
new scene called S15_SpecR and exports next to the script).

Developed and tested with Blender 4.2 (bpy 4.2). Written to tolerate 3.6+,
but only 4.2 has actually been run. The default body (body/s15_body.glb, an
image-to-3D reconstruction, see README) needs the OpenVDB grid nodes that
Blender 4.2 has; without them the script falls back to the lofted body.

Output (default ./export next to this file):
    S15_SpecR.blend        editable scene, real scale (metres)
    S15_SpecR_Roblox.fbx   Roblox import file, 1 unit = 1 stud
    S15_SpecR.glb          glTF copy in metres
    parts.txt              part list with triangle counts

Reference dimensions (Nissan GF-S15 Silvia Spec R):
    length 4445 mm, width 1695 mm, height 1285 mm, wheelbase 2525 mm,
    track 1470 / 1460 mm, 215/45R17 tyres on 17x7 five-spoke wheels.

Coordinate system while modelling: X = lateral (+X is the car's left),
Y = longitudinal (the nose points to -Y, Blender's front view), Z = up,
origin on the ground in the middle of the wheelbase.
"""

import bpy  # noqa: E402  (bpy first: the pip build only exposes bmesh after it)
import bmesh
import math
import os
import sys
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

STUD = 0.28                    # metres per Roblox stud
PAINT_PRESETS = {
    "lightning_yellow": (0.78, 0.58, 0.004),    # matched to the reference set
    "pearl_white": (0.86, 0.86, 0.83),
    "sparkling_silver": (0.55, 0.56, 0.57),
    "brilliant_blue": (0.02, 0.10, 0.45),
    "super_black": (0.012, 0.012, 0.014),
    "active_red": (0.55, 0.02, 0.02),
}
PAINT = "lightning_yellow"
CLAY = False                   # preview option: neutral grey paint
TRI_LIMIT = 20000              # Roblox MeshPart triangle cap
# Outer body shape from a mesh file (.glb/.gltf/.obj/.fbx); relative paths
# are next to this script. The default is the prepared image-to-3D body
# (see README); if it is missing, or with None / --body loft, the body is
# lofted from the design curves below. See body_from_mesh().
BODY_MESH = "body/s15_body.glb"
BODY_VOXEL = 0.006             # remesh resolution for BODY_MESH (m)
BODY_TRIS = 55000              # BODY_MESH shell triangles before the cuts

# Real dimensions (metres)
LENGTH, WIDTH, HEIGHT = 4.445, 1.695, 1.285
WHEELBASE = 2.525
Y_FAX, Y_RAX = -WHEELBASE / 2, WHEELBASE / 2
Y_NOSE = Y_FAX - 0.975          # front bumper tip
Y_TAIL = Y_RAX + 0.945          # rear bumper tip
TRACK_F, TRACK_R = 1.470, 1.460

TYRE_W, TYRE_AR, RIM_IN = 0.215, 0.45, 17
RIM_R = RIM_IN * 0.0254 / 2
TYRE_R = RIM_R + TYRE_W * TYRE_AR
RIM_W = 7 * 0.0254
WHEEL_X_F = TRACK_F / 2 - 0.006
WHEEL_X_R = TRACK_R / 2 - 0.004
WHEEL_Z = TYRE_R
ARCH_R = 0.355                 # measured from the reference side view
ARCH_Z = WHEEL_Z - 0.005

# Cross-section resolution (half profile). FLOOR and TOP must match for the
# end caps to grid-fill cleanly.
N_FLOOR, N_SIDE, N_TOP = 12, 20, 12
K_SIDE0 = N_FLOOR
K_BELT = N_FLOOR + N_SIDE
K_TOP = K_BELT + N_TOP


# --------------------------------------------------------------------------
# Small maths helpers
# --------------------------------------------------------------------------

def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    if e0 == e1:
        return 0.0 if x < e0 else 1.0
    t = clamp((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def bump(x, c, w):
    """Smooth bump: 1 at c, 0 beyond +/- w."""
    t = clamp(1 - abs(x - c) / w)
    return t * t * (3 - 2 * t)


class Curve:
    """Monotone cubic (PCHIP) interpolation through (x, y) keys."""

    def __init__(self, keys):
        keys = sorted(keys)
        self.x = [k[0] for k in keys]
        self.y = [k[1] for k in keys]
        n = len(keys)
        h = [self.x[i + 1] - self.x[i] for i in range(n - 1)]
        d = [(self.y[i + 1] - self.y[i]) / h[i] for i in range(n - 1)]
        m = [0.0] * n
        m[0], m[-1] = d[0], d[-1]
        for i in range(1, n - 1):
            if d[i - 1] * d[i] <= 0:
                m[i] = 0.0
            else:
                w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
                m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
        self.h, self.m = h, m

    def __call__(self, x):
        xs = self.x
        if x <= xs[0]:
            return self.y[0] + self.m[0] * (x - xs[0])
        if x >= xs[-1]:
            return self.y[-1] + self.m[-1] * (x - xs[-1])
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid
        h = self.h[lo]
        t = (x - xs[lo]) / h
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * self.y[lo] + (t3 - 2 * t2 + t) * h * self.m[lo]
                + (-2 * t3 + 3 * t2) * self.y[hi] + (t3 - t2) * h * self.m[hi])


def bezier(p0, p1, p2, p3, t):
    u = 1 - t
    return (p0 * (u * u * u) + p1 * (3 * u * u * t) + p2 * (3 * u * t * t) + p3 * (t * t * t))


def v2(x, y):
    return Vector((x, y))


def resample(points, n, closed=False):
    """Resample a polyline (list of Vectors) to n points evenly by arc length."""
    pts = list(points) + ([points[0]] if closed else [])
    seg = [(pts[i + 1] - pts[i]).length for i in range(len(pts) - 1)]
    total = sum(seg)
    count = n if closed else n - 1
    out, acc, i = [], 0.0, 0
    for j in range(n):
        target = total * j / count
        while i < len(seg) - 1 and acc + seg[i] < target:
            acc += seg[i]
            i += 1
        t = 0.0 if seg[i] == 0 else clamp((target - acc) / seg[i])
        out.append(pts[i].lerp(pts[i + 1], t))
    return out


def rounded_poly(corners, radius, steps=6):
    """2D polygon (list of (x, y)) with every corner filleted.

    radius may be a number or a list (one per corner)."""
    pts = [v2(*c) for c in corners]
    n = len(pts)
    radii = radius if isinstance(radius, (list, tuple)) else [radius] * n
    out = []
    for i in range(n):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        a, b = (p0 - p1), (p2 - p1)
        la, lb = a.length, b.length
        r = min(radii[i], la * 0.45, lb * 0.45)
        if r <= 1e-6:
            out.append(p1)
            continue
        s, e = p1 + a.normalized() * r, p1 + b.normalized() * r
        for k in range(steps + 1):
            t = k / steps
            out.append(s * ((1 - t) ** 2) + p1 * (2 * (1 - t) * t) + e * (t * t))
    return out


# --------------------------------------------------------------------------
# Scene / material helpers
# --------------------------------------------------------------------------

MATS = {}


def material(name, color, metallic=0.0, rough=0.5, alpha=1.0, emit=None,
             emit_strength=0.0, transmission=0.0, coat=0.0, ior=1.45):
    if name in MATS:
        return MATS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Roughness"].default_value = rough
    p.inputs["IOR"].default_value = ior
    for key in ("Coat Weight", "Clearcoat"):
        if key in p.inputs:
            p.inputs[key].default_value = coat
            break
    for key in ("Transmission Weight", "Transmission"):
        if key in p.inputs:
            p.inputs[key].default_value = transmission
            break
    if alpha < 1.0:
        p.inputs["Alpha"].default_value = alpha
        if hasattr(m, "blend_method"):
            m.blend_method = "BLEND"
    if emit:
        for key in ("Emission Color", "Emission"):
            if key in p.inputs:
                p.inputs[key].default_value = (*emit, 1)
                break
        p.inputs["Emission Strength"].default_value = emit_strength
    m.diffuse_color = (*color, alpha)
    MATS[name] = m
    return m


def setup_materials():
    paint = material("Paint", (0.20, 0.20, 0.21) if CLAY else PAINT_PRESETS[PAINT],
                     metallic=0.0, rough=0.12, coat=0.8)
    # slightly satin clear coat: blurs the reflections of the lumpy AI-body
    # surface (A-pillar) instead of mirroring every ripple
    for key in ("Coat Roughness", "Clearcoat Roughness"):
        if key in paint.node_tree.nodes["Principled BSDF"].inputs:
            paint.node_tree.nodes["Principled BSDF"].inputs[key].default_value = 0.1
            break
    material("BlackGloss", (0.01, 0.01, 0.012), rough=0.18, coat=0.5)
    material("BlackMatte", (0.018, 0.018, 0.02), rough=0.75)
    material("Rubber", (0.012, 0.012, 0.012), rough=0.85)
    material("Glass", (0.05, 0.065, 0.065), rough=0.02, alpha=0.38,
             transmission=0.6, ior=1.5)
    material("Chrome", (0.9, 0.9, 0.92), metallic=1.0, rough=0.08)
    material("Reflector", (0.75, 0.76, 0.78), metallic=1.0, rough=0.15)
    material("LensClear", (0.9, 0.92, 0.95), rough=0.02, alpha=0.25,
             transmission=0.9, ior=1.5)
    material("LensRed", (0.30, 0.01, 0.015), rough=0.05, alpha=0.5,
             transmission=0.3, emit=(0.5, 0.0, 0.0), emit_strength=0.1)
    material("BulbRed", (0.75, 0.02, 0.02), rough=0.25, emit=(0.8, 0.0, 0.0),
             emit_strength=0.6)
    material("LensAmber", (0.95, 0.42, 0.02), rough=0.1, alpha=0.85,
             transmission=0.3)
    material("LensSmoke", (0.07, 0.07, 0.08), rough=0.05, alpha=0.45,
             transmission=0.4)
    material("TailAmber", (0.45, 0.20, 0.02), rough=0.05, alpha=0.5, transmission=0.3)
    material("AmberReflector", (0.95, 0.45, 0.03), rough=0.3, emit=(0.9, 0.4, 0.0),
             emit_strength=0.15)
    material("Housing", (0.03, 0.03, 0.035), rough=0.55)
    material("Interior", (0.045, 0.045, 0.05), rough=0.8)
    material("InteriorTrim", (0.10, 0.10, 0.11), rough=0.5)
    material("Seat", (0.05, 0.05, 0.055), rough=0.9)
    material("SeatAccent", (0.14, 0.14, 0.15), rough=0.85)
    material("Rim", (0.62, 0.63, 0.65), metallic=0.9, rough=0.3)
    material("Brake", (0.35, 0.35, 0.36), metallic=1.0, rough=0.45)
    material("Caliper", (0.18, 0.18, 0.2), metallic=0.4, rough=0.4)
    material("Underbody", (0.03, 0.03, 0.032), rough=0.9)
    material("Exhaust", (0.55, 0.55, 0.57), metallic=1.0, rough=0.25)
    material("PlateWhite", (0.85, 0.86, 0.84), rough=0.4)
    material("PlateText", (0.02, 0.15, 0.05), rough=0.4)
    material("Badge", (0.8, 0.8, 0.82), metallic=1.0, rough=0.12)
    material("Mirror", (0.85, 0.87, 0.9), metallic=1.0, rough=0.02)
    material("Headliner", (0.32, 0.32, 0.31), rough=0.9)
    material("Carpet", (0.03, 0.03, 0.032), rough=1.0)
    material("Gauge", (0.01, 0.01, 0.012), rough=0.3)
    material("GaugeMark", (0.9, 0.9, 0.9), rough=0.4, emit=(0.9, 0.9, 0.95), emit_strength=0.4)
    material("Needle", (0.9, 0.12, 0.02), rough=0.4, emit=(0.9, 0.1, 0.0), emit_strength=0.4)
    material("Leather", (0.035, 0.035, 0.037), rough=0.55)
    material("EngineBlock", (0.42, 0.43, 0.44), metallic=0.6, rough=0.5)
    material("ValveCover", (0.55, 0.03, 0.03), metallic=0.3, rough=0.6)
    material("Aluminum", (0.78, 0.79, 0.8), metallic=1.0, rough=0.22)
    material("Hose", (0.02, 0.02, 0.022), rough=0.6)
    material("Coupler", (0.05, 0.12, 0.45), rough=0.5)
    material("Translucent", (0.85, 0.85, 0.8), rough=0.3, alpha=0.7)
    material("Radiator", (0.06, 0.06, 0.065), metallic=0.5, rough=0.6)
    material("Battery", (0.05, 0.05, 0.06), rough=0.5)
    material("Terminal", (0.7, 0.05, 0.05), rough=0.5)
    material("CoilPack", (0.08, 0.08, 0.09), rough=0.4)
    material("Turbine", (0.25, 0.22, 0.2), metallic=0.8, rough=0.6)
    material("Emitter", (0.95, 0.95, 0.9), rough=0.1, emit=(1.0, 0.97, 0.9),
             emit_strength=0.0)


def new_object(name, bm, mats, collection=None, smooth_angle=35.0):
    """Turn a bmesh into an object. `mats` is a list of material names."""
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for mname in mats:
        me.materials.append(MATS[mname])
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if smooth_angle is not None:
        shade_smooth(ob, smooth_angle)
    return ob


def shade_smooth(ob, angle_deg=35.0):
    """Smooth shading with sharp edges above angle_deg (version independent)."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    lim = math.radians(angle_deg)
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        if len(e.link_faces) != 2:
            e.smooth = False
        else:
            e.smooth = e.calc_face_angle(0.0) < lim
    bm.to_mesh(me)
    bm.free()
    if hasattr(me, "use_auto_smooth"):          # Blender <= 4.0
        me.use_auto_smooth = True
        me.auto_smooth_angle = math.pi


def tri_count(ob):
    return sum(len(p.vertices) - 2 for p in ob.data.polygons)


def apply_modifiers(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev)
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    bpy.data.meshes.remove(old)


def delete_object(ob):
    me = ob.data if ob.type == "MESH" else None
    bpy.data.objects.remove(ob, do_unlink=True)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)


# --------------------------------------------------------------------------
# Body shape definition (side, plan and section design curves)
# --------------------------------------------------------------------------

NOSE_LEN = 0.46           # length of the rounded nose zone
TAIL_LEN = 0.30

# bottom edge of the body side (skirts / bumpers)
Z_BOT = Curve([(-2.30, 0.205), (-2.0, 0.190), (-1.6, 0.185), (-1.0, 0.173),
               (0.0, 0.172), (0.9, 0.175), (1.5, 0.200), (1.9, 0.225),
               (2.30, 0.245)])
# widest half-width of the body (door/fender bulge)
W_MAX = Curve([(-2.30, 0.795), (-1.95, 0.812), (-1.55, 0.836), (-1.26, 0.846),
               (-0.95, 0.840), (-0.40, 0.8475), (0.55, 0.8475), (0.95, 0.844),
               (1.26, 0.8475), (1.62, 0.843), (1.95, 0.832), (2.30, 0.815)])
# height of the widest point
Z_BULGE = Curve([(-2.3, 0.53), (-1.2, 0.62), (0.0, 0.68), (1.2, 0.71),
                 (2.3, 0.72)])
# shoulder character line
Z_CREASE = Curve([(-2.30, 0.600), (-1.85, 0.655), (-1.20, 0.715),
                  (-0.50, 0.752), (0.30, 0.775), (1.00, 0.800), (1.60, 0.812),
                  (2.30, 0.815)])
# side-skirt top crease
Z_SKIRT = Curve([(-2.3, 0.31), (0.0, 0.30), (2.3, 0.31)])
# belt line: base of the side glass / top of the fenders
Z_BELT = Curve([(-2.30, 0.640), (-2.00, 0.703), (-1.80, 0.742),
                (-1.50, 0.792), (-1.20, 0.826), (-0.90, 0.850), (-0.73, 0.858),
                (-0.40, 0.861), (-0.10, 0.868), (0.20, 0.880), (0.45, 0.893),
                (0.70, 0.910), (1.00, 0.922), (1.30, 0.932), (1.60, 0.936),
                (2.00, 0.928), (2.30, 0.918)])
# centre-line top silhouette: hood, windshield, roof, backlight, deck
Z_TOP = Curve([(-2.30, 0.655), (-2.15, 0.688), (-2.00, 0.722), (-1.80, 0.766),
               (-1.60, 0.812), (-1.40, 0.849), (-1.20, 0.873), (-1.00, 0.888),
               (-0.85, 0.895), (-0.77, 0.899), (-0.73, 0.903), (-0.69, 0.921),
               (-0.50, 1.018), (-0.30, 1.126), (-0.15, 1.200), (-0.06, 1.238),
               (0.05, 1.266), (0.20, 1.280), (0.35, 1.285), (0.50, 1.281),
               (0.65, 1.262), (0.80, 1.242), (0.90, 1.220), (1.00, 1.175),
               (1.20, 1.088), (1.40, 1.010), (1.48, 0.975), (1.53, 0.962),
               (1.62, 0.956), (1.90, 0.952), (2.30, 0.944)])
# roof-rail / A-pillar / C-pillar / hood-seam guide line (half-width, height)
X_RAIL = Curve([(-2.30, 0.675), (-1.60, 0.692), (-1.00, 0.668), (-0.73, 0.656),
                (-0.06, 0.586), (0.50, 0.598), (0.88, 0.598), (1.20, 0.628),
                (1.50, 0.652), (1.90, 0.645), (2.30, 0.628)])
Z_RAIL = Curve([(-2.30, 0.652), (-2.00, 0.715), (-1.80, 0.754), (-1.50, 0.804),
                (-1.20, 0.838), (-0.90, 0.862), (-0.73, 0.872), (-0.50, 0.968),
                (-0.25, 1.088), (-0.13, 1.150), (-0.02, 1.192), (0.10, 1.211),
                (0.50, 1.219), (0.88, 1.196), (1.10, 1.098), (1.30, 1.020),
                (1.48, 0.962), (1.55, 0.950), (1.90, 0.942), (2.30, 0.932)])
# 0 = hood/deck style top, 1 = glasshouse style top
GREEN = Curve([(-2.4, 0.0), (-0.78, 0.0), (-0.56, 1.0), (1.30, 1.0),
               (1.50, 0.0), (2.4, 0.0)])

# side-skirt step only exists between the wheel arches
SKIRT_AMT = Curve([(-2.4, 0.0), (-0.92, 0.0), (-0.80, 1.0), (0.82, 1.0),
                   (0.94, 0.0), (2.4, 0.0)])


def half_profile(y):
    """Half cross-section at station y: list of K_TOP+1 2D points (x, z)
    from the bottom centre-line round the side to the top centre-line."""
    zb = Z_BOT(y)
    wm = W_MAX(y)
    # fender flares over the wheels
    wm += 0.004 * bump(y, Y_FAX, 0.55) + 0.003 * bump(y, Y_RAX, 0.55)
    zbul = Z_BULGE(y)
    zc = Z_CREASE(y)
    zs = max(Z_SKIRT(y), zb + 0.10)
    zbelt = Z_BELT(y)
    skirt = SKIRT_AMT(y)
    g = GREEN(y)

    def side_x(z):
        if z < zbul:
            t = (zbul - z) / max(zbul - zs, 0.05)
            return wm - 0.026 * t * t
        t = (z - zbul) / max(zc - zbul, 0.05)
        return wm - 0.004 * t * t

    pts = []
    # floor (flat underside)
    xf = wm - 0.065
    for i in range(N_FLOOR + 1):
        pts.append(v2(xf * i / N_FLOOR, zb))
    # side: bottom roll-under, skirt, door, crease, upper flank
    x_sk = side_x(zs + 0.012) + 0.010 * skirt          # skirt face
    side = [
        v2(wm - 0.040, zb + 0.004),
        v2(wm - 0.026, zb + 0.016),
        v2(x_sk - 0.004, zb + 0.040),
        v2(x_sk - 0.001, lerp(zb, zs, 0.6)),
        v2(x_sk, zs - 0.010),
        v2(x_sk, zs),                                    # skirt top edge
        v2(side_x(zs + 0.010), zs + 0.006),              # step in
    ]
    for t in (0.12, 0.30, 0.50, 0.70, 0.86):
        z = lerp(zs + 0.03, zc - 0.03, t)
        side.append(v2(side_x(z), z))
    xc = side_x(zc)
    side.append(v2(xc - 0.0005, zc - 0.012))
    side.append(v2(xc, zc))                              # shoulder crease
    side.append(v2(xc - 0.010, zc + 0.008))
    # upper flank leans in towards the belt
    xbelt = xc - 0.012 - (zbelt - zc) * lerp(0.48, 0.60, g)
    p_cr = v2(xc - 0.012, zc + 0.014)
    p_bt = v2(xbelt, zbelt)
    for t in (0.0, 0.3, 0.6, 0.85):
        q = p_cr.lerp(p_bt, t)
        q.x += 0.006 * math.sin(math.pi * t)              # slight convexity
        side.append(q)
    side.append(p_bt)
    assert len(side) == N_SIDE, len(side)
    pts.extend(side)

    # top: belt -> rail (glass / fender roll) -> centre (roof / hood / deck)
    B = p_bt
    R = v2(X_RAIL(y), Z_RAIL(y))
    T = v2(0.0, Z_TOP(y))
    glass_dir = (R - B).normalized()
    flank_dir = v2(-0.62, 0.78).normalized()
    hood_dir = v2(-1.0, 0.03).normalized()
    roof_dir = v2(-0.56, 0.83).normalized()
    tB = flank_dir.lerp(glass_dir, g).normalized()
    tR = hood_dir.lerp(glass_dir, g).normalized()
    tR2 = hood_dir.lerp(roof_dir, g).normalized()
    dBR = (R - B).length
    dRT = (R - T).length
    c1 = B + tB * dBR * 0.35
    c2 = R - tR * dBR * 0.35
    c3 = R + tR2 * dRT * lerp(0.30, 0.10, g)
    c4 = T + v2(1.0, 0.0) * dRT * 0.45
    nb = 6
    for i in range(1, nb + 1):
        t = i / nb
        pts.append(bezier(B, c1, c2, R, t))
    nr = N_TOP - nb
    for i in range(1, nr + 1):
        t = i / nr
        t = t ** lerp(1.0, 1.35, g)                       # denser at the roof edge
        pts.append(bezier(R, c3, c4, T, t))
    pts[-1] = v2(0.0, T.y)
    assert len(pts) == K_TOP + 1, len(pts)
    return pts


# --- rounded ends ---------------------------------------------------------

def _ellipse_drop(t):
    """Roll-over curve: flat at t=0, finite slope at t=1 (no folds)."""
    t = clamp(t, 0.0, 1.0)
    return t ** 2.4


# cap edge heights (z of the end-cap top / bottom at the centre line)
NOSE_ZC, NOSE_BZ = 0.40, 0.80
NOSE_CAP_TOP = NOSE_ZC + (Z_TOP(Y_NOSE) - NOSE_ZC) * NOSE_BZ
NOSE_CAP_BOT = NOSE_ZC + (Z_BOT(Y_NOSE) - NOSE_ZC) * NOSE_BZ
TAIL_ZC, TAIL_BZ = 0.58, 0.86
TAIL_CAP_TOP = TAIL_ZC + (Z_TOP(Y_TAIL) - TAIL_ZC) * TAIL_BZ
TAIL_CAP_BOT = TAIL_ZC + (Z_BOT(Y_TAIL) - TAIL_ZC) * TAIL_BZ


def nose_setback(z):
    """How far (m) the front face recedes at height z (side-view shape):
    near-vertical bumper face rolling over into the hood, tucked chin."""
    if z >= 0.50:
        return 0.065 * _ellipse_drop((z - 0.50) / (NOSE_CAP_TOP - 0.50))
    if z <= 0.34:
        return 0.040 * _ellipse_drop((0.34 - z) / (0.34 - NOSE_CAP_BOT))
    return 0.0


def tail_setback(z):
    crease = 0.11 * max(0.0, z - 0.615)          # horizontal bumper crease line
    if z >= 0.58:
        return (crease + 0.006 * clamp((z - 0.58) / 0.20)
                + 0.040 * _ellipse_drop((z - 0.77) / (TAIL_CAP_TOP - 0.77)))
    if z <= 0.43:
        return 0.065 * _ellipse_drop((0.43 - z) / (0.43 - TAIL_CAP_BOT))
    return 0.0


END = {
    # d: direction pointing into the car; phi_min: where the end cap starts
    "nose": dict(y0=Y_NOSE, d=1.0, L=NOSE_LEN, phi_min=math.asin(0.24), bz=NOSE_BZ,
                 zc=NOSE_ZC, zround=0.55, setback=nose_setback, n=18),
    "tail": dict(y0=Y_TAIL, d=-1.0, L=TAIL_LEN, phi_min=math.asin(0.34), bz=TAIL_BZ,
                 zc=TAIL_ZC, zround=0.45, setback=tail_setback, n=12),
}


def end_y(E, phi):
    return E["y0"] + E["d"] * E["L"] * (1 - math.cos(phi))


def body_stations():
    """List of (y, phase) where phase is None or ('nose'|'tail', angle)."""
    st = []
    E = END["nose"]
    for j in range(E["n"]):
        a = lerp(E["phi_min"], math.pi / 2, j / E["n"])
        st.append((end_y(E, a), ("nose", a)))
    y_a, y_b = Y_NOSE + NOSE_LEN, Y_TAIL - TAIL_LEN
    keys = [y_a, -1.95, -1.70, -1.45, -1.20, -1.00, -0.88, -0.80, -0.76,
            -0.73, -0.71, -0.68, -0.58, -0.45, -0.30, -0.15, -0.08, -0.02, 0.05,
            0.15, 0.30, 0.45, 0.60, 0.75, 0.86, 0.92, 1.00, 1.10, 1.22, 1.34,
            1.44, 1.48, 1.52, 1.57, 1.65, 1.80, y_b]
    keys = sorted(k for k in keys if y_a <= k <= y_b)
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        n = max(1, int(math.ceil((b - a) / 0.09)))
        for j in range(n):
            st.append((a + (b - a) * j / n, None))
    E = END["tail"]
    for j in range(E["n"] + 1):
        a = lerp(math.pi / 2, E["phi_min"], j / E["n"])
        st.append((end_y(E, a), ("tail", a)))
    return st


def end_point(E, p, y_station, phi, z_lo=-9.0, z_hi=9.0):
    """Map a full-section point onto the rounded nose/tail.

    The setback is evaluated with z clamped to the ring's floor/belt corners
    so the hood/deck and floor groups move rigidly (no folds)."""
    u = clamp((phi - E["phi_min"]) / E["zround"])
    zs = E["bz"] + (1 - E["bz"]) * (1 - math.cos(u * math.pi / 2))
    # setback fade; exponent kept low so rings never fold over each other
    w = min(1.0, math.cos(phi) / math.cos(E["phi_min"])) ** 1.15
    x = p.x * math.sin(phi)
    z = E["zc"] + (p.y - E["zc"]) * zs
    zl = E["zc"] + (z_lo - E["zc"]) * zs
    zh = E["zc"] + (z_hi - E["zc"]) * zs
    y = y_station + E["d"] * E["setback"](clamp(z, zl, zh)) * w
    return Vector((x, y, z))


END_BULGE = 0.6     # share of the plan-view nose/tail curvature carried by the rings


def ring_points(y, phase):
    """Full 3D half ring for a station (list of Vector)."""
    prof = half_profile(y)
    if phase is None:
        return [Vector((p.x, y, p.y)) for p in prof]
    kind, a = phase
    E = END[kind]
    xf, xt = max(prof[N_FLOOR].x, 1e-3), max(prof[K_BELT].x, 1e-3)
    out = []
    z_lo, z_hi = prof[N_FLOOR].y, prof[K_BELT].y
    for k, p in enumerate(prof):
        q = end_point(E, p, y, a, z_lo, z_hi)
        if k <= N_FLOOR or k >= K_BELT:
            f = clamp(p.x / (xf if k <= N_FLOOR else xt))
            sa = f * math.sin(a)
            delta = E["L"] * (math.sqrt(max(0.0, 1 - sa * sa)) - math.cos(a))
            q.y -= E["d"] * END_BULGE * delta
        out.append(q)
    return out


def build_loft(step=1):
    """Closed, manifold body solid (before any cut-outs).

    step=2 builds a half-resolution version (used for the inner offset)."""
    stations = body_stations()
    if step > 1:
        stations = stations[::step] if (len(stations) - 1) % step == 0 \
            else stations[::step] + [stations[-1]]
    nf, ns, nt = N_FLOOR // step, N_SIDE // step, N_TOP // step
    K = nf + ns + nt
    k_side0 = nf
    rings = []
    for y, ph in stations:
        r = ring_points(y, ph)
        rings.append(r[::step])
    bm = bmesh.new()
    loops = []
    for r in rings:
        right = [bm.verts.new(p) for p in r]                     # +X, k=0..K
        left = [bm.verts.new((-p.x, p.y, p.z)) for p in r[K - 1:0:-1]]
        loops.append(right + left)                               # 2K verts
    M = 2 * K
    for i in range(len(loops) - 1):
        a, b = loops[i], loops[i + 1]
        for j in range(M):
            j2 = (j + 1) % M
            bm.faces.new((a[j], a[j2], b[j2], b[j]))

    def kidx(k, side):
        if side > 0 or k == 0 or k == K:
            return k
        return 2 * K - k

    def cap(loop, ring, E, front):
        cols, rows = 2 * nf, ns
        s0 = math.sin(E["phi_min"])

        def bpt(i, j):
            if i == 0:
                k, sd = abs(nf - j), (1 if j >= nf else -1)
            elif i == rows:
                k, sd = K - abs(nf - j), (1 if j >= nf else -1)
            elif j == 0:
                k, sd = k_side0 + i, -1
            else:
                k, sd = k_side0 + i, 1
            v = ring[k]
            return Vector((v.x * sd, v.y, v.z)), loop[kidx(k, sd)]

        def coons(f):
            out = {}
            for i in range(1, rows):
                for j in range(1, cols):
                    s, t = j / cols, i / rows
                    out[i, j] = ((1 - t) * f(0, j) + t * f(rows, j) + (1 - s) * f(i, 0)
                                 + s * f(i, cols)
                                 - ((1 - s) * (1 - t) * f(0, 0) + s * (1 - t) * f(0, cols)
                                    + (1 - s) * t * f(rows, 0) + s * t * f(rows, cols)))
            return out

        bnd = {}
        for i in range(rows + 1):
            for j in range(cols + 1):
                if i in (0, rows) or j in (0, cols):
                    bnd[i, j] = bpt(i, j)

        def wn(i):
            return max(abs(bnd[i, cols][0].x), 1e-3) / s0

        z_lo, z_hi = ring[nf].z, ring[K - nt].z

        def shape(x, z, i):
            q = clamp(abs(x) / wn(i), 0, 0.999)
            return E["d"] * (E["L"] * (1 - math.sqrt(1 - q * q))
                             + E["setback"](clamp(z, z_lo, z_hi)))

        pos = coons(lambda i, j: bnd[i, j][0])
        grid = {k: v[1] for k, v in bnd.items()}
        for (i, j), p in pos.items():
            grid[i, j] = bm.verts.new(p)
        for i in range(rows):
            for j in range(cols):
                q = (grid[i, j], grid[i, j + 1], grid[i + 1, j + 1], grid[i + 1, j])
                bm.faces.new(q if front else q[::-1])

    cap(loops[0], rings[0], END["nose"], True)
    cap(loops[-1], rings[-1], END["tail"], False)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    relax_ends(bm, iters=10 // step)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if bm.calc_volume(signed=True) < 0:          # make sure normals face out
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    check_self_intersections(bm, f"loft(step={step})")
    return bm


def relax_ends(bm, iters=10, lam=0.45):
    """Weighted Laplacian relax of the nose-top / tail-top roll-overs, where
    the analytic rings meet the end caps (irons out small wrinkles)."""
    def weight(co):
        wn = smoothstep(Y_NOSE + 0.42, Y_NOSE + 0.22, co.y) * smoothstep(0.50, 0.60, co.z)
        wt = 0.6 * smoothstep(Y_TAIL - 0.30, Y_TAIL - 0.14, co.y) * smoothstep(0.80, 0.88, co.z)
        return max(wn, wt)
    W = {v: weight(v.co) for v in bm.verts}
    verts = [v for v, w in W.items() if w > 0]
    for _ in range(iters):
        new = {}
        for v in verts:
            nb = [e.other_vert(v).co for e in v.link_edges]
            avg = sum(nb, Vector()) / len(nb)
            new[v] = v.co.lerp(avg, lam * W[v])
        for v, c in new.items():
            v.co = c


def inner_offset(bm, t, iters=10):
    """Offset a closed loft inwards by t and relax it (removes folds at the
    creases). Used to hollow the body into a shell."""
    bm.normal_update()
    target = {v: v.co - v.normal * t for v in bm.verts}
    for v in bm.verts:
        v.co = target[v]
    for _ in range(iters):
        new = {}
        for v in bm.verts:
            nb = [e.other_vert(v).co for e in v.link_edges]
            avg = sum(nb, Vector()) / len(nb)
            new[v] = v.co.lerp(avg, 0.5)
        for v, c in new.items():
            v.co = c
    bm.normal_update()
    return bm


# --------------------------------------------------------------------------
# Generic mesh builders
# --------------------------------------------------------------------------

def revolve(profile, segments, closed=False, angle=None, start=0.0, mat=0,
            bm=None, xform=None):
    """Revolve a 2D profile [(radius, axial), ...] about the X axis.

    `angle` limits the sweep (radians) and caps nothing; `closed` joins the
    last profile point to the first."""
    own = bm is None
    bm = bm or bmesh.new()
    full = angle is None
    sweep = 2 * math.pi if full else angle
    nseg = segments if full else segments + 1
    rings = []
    for i in range(nseg):
        th = start + sweep * i / segments
        c, s_ = math.cos(th), math.sin(th)
        ring = []
        for r, a in profile:
            p = Vector((a, r * c, r * s_))
            if xform is not None:
                p = xform @ p
            ring.append(bm.verts.new(p))
        rings.append(ring)
    n = len(profile)
    faces = []
    for i in range(segments):
        r0, r1 = rings[i], rings[(i + 1) % nseg]
        for j in range(n if closed else n - 1):
            j2 = (j + 1) % n
            q = (r0[j], r0[j2], r1[j2], r1[j])
            if len(set(q)) == 4:
                f = bm.faces.new(q)
                f.material_index = mat
                faces.append(f)
    if not full:
        # cap both ends of a partial sweep with n-gons
        for ring in (rings[0], rings[-1]):
            if closed and len(ring) >= 3:
                f = bm.faces.new(ring)
                f.material_index = mat
                faces.append(f)
    if own:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm if own else faces


def add_box(bm, center, size, mat=0, rot=None, bevel=0.0):
    """Axis-aligned (optionally rotated) box into bm. Returns the new faces."""
    geom = bmesh.ops.create_cube(bm, size=1.0)
    verts = geom["verts"]
    m = Matrix.Diagonal((*size, 1.0))
    if rot is not None:
        m = rot.to_4x4() @ m
    m = Matrix.Translation(center) @ m
    bmesh.ops.transform(bm, matrix=m, verts=verts)
    faces = list({f for v in verts for f in v.link_faces})
    for f in faces:
        f.material_index = mat
    if bevel > 0:
        edges = list({e for f in faces for e in f.edges})
        res = bmesh.ops.bevel(bm, geom=edges, offset=bevel, segments=2,
                              affect="EDGES", profile=0.5)
        for f in res["faces"]:
            f.material_index = mat
        faces = list({f for v in res["verts"] for f in v.link_faces} | set(f for f in faces if f.is_valid))
    return [f for f in faces if f.is_valid]


def add_cylinder(bm, p0, p1, r, segs=16, mat=0, r1=None, cap=True):
    """Cylinder/cone between two points."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    L = d.length
    q = d.normalized().to_track_quat("Z", "Y")
    m = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4()
    geom = bmesh.ops.create_cone(bm, cap_ends=cap, cap_tris=False, segments=segs,
                                 radius1=r, radius2=r if r1 is None else r1,
                                 depth=L, matrix=m)
    faces = list({f for v in geom["verts"] for f in v.link_faces})
    for f in faces:
        f.material_index = mat
    return faces


def loft_tube(bm, centers, sections, mat=0, cap=True):
    """Loft closed cross-sections (lists of Vectors, same count) in order."""
    rings = [[bm.verts.new(p) for p in sec] for sec in sections]
    n = len(rings[0])
    faces = []
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        for j in range(n):
            j2 = (j + 1) % n
            f = bm.faces.new((a[j], a[j2], b[j2], b[j]))
            f.material_index = mat
            faces.append(f)
    if cap:
        for r in (rings[0], rings[-1]):
            f = bm.faces.new(r)
            f.material_index = mat
            faces.append(f)
    return faces


def prism_from_poly(poly2d, axis, a0, a1, mat=0, bm=None):
    """Extrude a 2D polygon along axis 'x' | 'y' | 'z' between a0 and a1.

    poly2d is in the plane of the two remaining axes, in (x,y,z) order:
    axis 'x' -> (y, z); axis 'y' -> (x, z); axis 'z' -> (x, y)."""
    own = bm is None
    bm = bm or bmesh.new()

    def P(u, v, a):
        if axis == "x":
            return Vector((a, u, v))
        if axis == "y":
            return Vector((u, a, v))
        return Vector((u, v, a))

    pts = [Vector((p[0], p[1])) for p in poly2d]
    # drop duplicates
    clean = [pts[0]]
    for p in pts[1:]:
        if (p - clean[-1]).length > 1e-6:
            clean.append(p)
    if (clean[0] - clean[-1]).length < 1e-6:
        clean.pop()
    lo = [bm.verts.new(P(p.x, p.y, a0)) for p in clean]
    hi = [bm.verts.new(P(p.x, p.y, a1)) for p in clean]
    n = len(clean)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((lo[i], lo[j], hi[j], hi[i])).material_index = mat
    bm.faces.new(lo).material_index = mat
    bm.faces.new(hi[::-1]).material_index = mat
    if own:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def bm_from_object(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.transform(ob.matrix_world)
    return bm


def boolean(target, cutter, op="DIFFERENCE", use_self=False):
    """Apply an exact boolean (materials transferred from the cutter)."""
    mod = target.modifiers.new("bool", "BOOLEAN")
    mod.operation = op
    mod.object = cutter
    mod.solver = "EXACT"
    if hasattr(mod, "material_mode"):
        mod.material_mode = "TRANSFER"
    if hasattr(mod, "use_self"):
        mod.use_self = use_self
    with bpy.context.temp_override(object=target, active_object=target,
                                   selected_objects=[target]):
        bpy.ops.object.modifier_apply(modifier=mod.name)


def cutter_object(name, bm, mat="Housing"):
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if bm.calc_volume(signed=True) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    ob = new_object(name, bm, [mat], smooth_angle=None)
    ob.display_type = "WIRE"
    ob.hide_render = True
    return ob


def check_self_intersections(bm, name):
    bm.faces.ensure_lookup_table()
    t = BVHTree.FromBMesh(bm)
    bad = 0
    for a, b in t.overlap(t):
        if a < b and not (set(bm.faces[a].verts) & set(bm.faces[b].verts)):
            bad += 1
    if bad:
        print(f"[warn] {name}: {bad} self-intersecting face pairs")
    return bad == 0


def check_manifold(bm, name):
    bad = [e for e in bm.edges if not e.is_manifold]
    if bad:
        print(f"[warn] {name}: {len(bad)} non-manifold edges")
    return not bad


# --------------------------------------------------------------------------
# Wheels
# --------------------------------------------------------------------------

def tyre_profile():
    """Closed (radius, axial) profile of a 215/45R17 tyre, +axial = outboard."""
    hw = TYRE_W / 2
    R = TYRE_R
    rb = RIM_R + 0.010
    side = [
        (rb, hw - 0.030), (rb + 0.008, hw - 0.012), (rb + 0.022, hw - 0.002),
        (rb + 0.040, hw + 0.004), (rb + 0.060, hw + 0.004), (R - 0.030, hw - 0.002),
        (R - 0.016, hw - 0.010), (R - 0.006, hw - 0.020), (R - 0.001, hw - 0.032),
    ]
    tread = []
    grooves = (0.068, 0.024)
    a = hw - 0.034
    # walk the tread from outboard to inboard with 4 circumferential grooves
    edges = []
    for g in grooves:
        edges += [g + 0.006, g - 0.006]
    for g in grooves:
        edges += [-g + 0.006, -g - 0.006]
    edges = sorted(set(edges), reverse=True)
    tread.append((R, a))
    gi = 0
    while gi < len(edges):
        e0, e1 = edges[gi], edges[gi + 1]
        tread += [(R, e0 + 0.0015), (R - 0.0025, e0), (R - 0.008, e0 - 0.001),
                  (R - 0.008, e1 + 0.001), (R - 0.0025, e1), (R, e1 - 0.0015)]
        gi += 2
    tread.append((R, -a))
    inner = [(r, -ax) for r, ax in reversed(side)]
    liner = [(rb - 0.004, -hw + 0.034), (rb - 0.004, hw - 0.034)]
    return side + tread + inner + liner


def build_tyre(segments=64):
    prof = tyre_profile()
    bm = revolve(prof, segments, closed=True)
    # shoulder tread blocks: notch every other block on the outer ribs
    hw = TYRE_W / 2
    for v in bm.verts:
        a = v.co.x
        r = math.hypot(v.co.y, v.co.z)
        if r > TYRE_R - 0.0012 and abs(a) > 0.076:
            th = math.atan2(v.co.z, v.co.y)
            block = int((th + math.pi) / (2 * math.pi) * segments) % 4
            if block < 1:
                k = (TYRE_R - 0.006) / r
                v.co.y *= k
                v.co.z *= k
    return bm


def build_rim(spokes=5, segments=72):
    """17x7 five split-spoke wheel. Local axis +X = outboard face."""
    bm = bmesh.new()
    a_out = RIM_W / 2 + 0.010
    a_in = -RIM_W / 2 - 0.004
    R = RIM_R
    barrel = [
        (R - 0.018, a_out - 0.030), (R - 0.013, a_out - 0.010),
        (R - 0.006, a_out - 0.003), (R + 0.004, a_out - 0.0005),
        (R + 0.016, a_out - 0.001), (R + 0.0205, a_out - 0.006),
        (R + 0.019, a_out - 0.012), (R + 0.005, a_out - 0.016),
        (R, a_out - 0.030), (R - 0.004, a_out - 0.060),
        (R - 0.022, a_out - 0.075), (R - 0.022, a_in + 0.040),
        (R - 0.004, a_in + 0.025), (R, a_in + 0.012), (R + 0.016, a_in + 0.004),
        (R + 0.016, a_in), (R - 0.004, a_in), (R - 0.026, a_in + 0.030),
        (R - 0.026, a_out - 0.072), (R - 0.020, a_out - 0.034),
    ]
    revolve(barrel, 60, closed=True, mat=0, bm=bm)
    # centre hub: concave face, cap and lug bosses
    hub = [(0.0, a_out - 0.030), (0.030, a_out - 0.030), (0.034, a_out - 0.033),
           (0.036, a_out - 0.036), (0.080, a_out - 0.041), (0.090, a_out - 0.047),
           (0.092, a_out - 0.066), (0.0, a_out - 0.066)]
    revolve(hub, 40, closed=True, mat=0, bm=bm)
    cap = [(0.0, a_out - 0.024), (0.026, a_out - 0.024), (0.0295, a_out - 0.027),
           (0.0295, a_out - 0.034), (0.0, a_out - 0.034)]
    revolve(cap, 28, closed=True, mat=2, bm=bm)
    # lug nuts on a 5x114.3 pattern
    pcd = 0.1143 / 2
    for i in range(5):
        th = 2 * math.pi * i / 5 + math.pi / 5
        c = Vector((a_out - 0.040, pcd * math.cos(th), pcd * math.sin(th)))
        add_cylinder(bm, c, c + Vector((0.016, 0, 0)), 0.0105, segs=6, mat=2)
        add_cylinder(bm, c + Vector((0.016, 0, 0)), c + Vector((0.021, 0, 0)),
                     0.0090, segs=6, mat=2, r1=0.005)
        add_cylinder(bm, c - Vector((0.004, 0, 0)), c, 0.0135, segs=12, mat=1)
    # five broad spokes that flare into the rim, gentle twist, concave face
    # (reference wheel: ~6 cm wide mid-spoke, ~11 cm where it meets the rim)
    r_root, r_tip = 0.075, R - 0.010
    for i in range(spokes):
        th0 = 2 * math.pi * i / spokes
        secs = []
        n = 9
        for k in range(n):
            t = k / (n - 1)
            r = lerp(r_root, r_tip, t)
            th = th0 + 0.06 * t ** 1.3                       # twist towards the rim
            half_w = lerp(0.030, 0.027, t) + 0.029 * smoothstep(0.55, 1.0, t) ** 1.5
            depth = lerp(0.034, 0.022, t)
            face_a = lerp(a_out - 0.042, a_out - 0.012, t ** 0.8)   # concave
            cy, sy = math.cos(th), math.sin(th)
            radial = Vector((0, cy, sy))
            tang = Vector((0, -sy, cy))
            cen = radial * r
            sec = []
            for (u, w) in ((-1, 0), (-0.9, 0.7), (-0.45, 1), (0.45, 1), (0.9, 0.7),
                           (1, 0), (0.7, -1), (-0.7, -1)):
                p = cen + tang * (u * half_w)
                if w > 0:
                    p.x = face_a - (1 - w) * 0.006
                elif w == 0:
                    p.x = face_a - depth * 0.5
                else:
                    p.x = face_a - depth
                sec.append(p)
            secs.append(sec)
        loft_tube(bm, None, secs, mat=0)
    # valve stem
    th = 2 * math.pi * 0.5 / spokes
    c = Vector((a_out - 0.020, (R - 0.010) * math.cos(th), (R - 0.010) * math.sin(th)))
    add_cylinder(bm, c, c + Vector((0.018, 0, 0)), 0.0035, segs=8, mat=1)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    return bm


def build_brake():
    """Ventilated disc + hat. Local axis +X outboard."""
    bm = bmesh.new()
    a0 = RIM_W / 2 - 0.085
    disc = [(0.085, a0 + 0.013), (0.146, a0 + 0.013), (0.148, a0 + 0.011),
            (0.148, a0 - 0.011), (0.146, a0 - 0.013), (0.085, a0 - 0.013)]
    revolve(disc, 64, closed=True, mat=0, bm=bm)
    hat = [(0.0, a0 + 0.040), (0.082, a0 + 0.040), (0.086, a0 + 0.034),
           (0.086, a0 + 0.010), (0.075, a0 + 0.010), (0.075, a0 + 0.034),
           (0.0, a0 + 0.034)]
    revolve(hat, 40, closed=True, mat=1, bm=bm)
    return bm


def build_caliper():
    """Twin-piston style caliper hugging the disc at the trailing edge."""
    bm = bmesh.new()
    a0 = RIM_W / 2 - 0.085
    prof = [(0.104, a0 + 0.034), (0.160, a0 + 0.034), (0.166, a0 + 0.026),
            (0.166, a0 - 0.026), (0.160, a0 - 0.034), (0.104, a0 - 0.034),
            (0.104, a0 - 0.016), (0.150, a0 - 0.016), (0.150, a0 + 0.016),
            (0.104, a0 + 0.016)]
    # trailing side, slightly above centre (angle measured from +Y towards +Z)
    revolve(prof, 10, closed=True, angle=math.radians(62),
            start=math.radians(-18), mat=0, bm=bm)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


WHEELS = {
    "FL": (Vector((WHEEL_X_F, Y_FAX, WHEEL_Z)), 1),
    "FR": (Vector((-WHEEL_X_F, Y_FAX, WHEEL_Z)), -1),
    "RL": (Vector((WHEEL_X_R, Y_RAX, WHEEL_Z)), 1),
    "RR": (Vector((-WHEEL_X_R, Y_RAX, WHEEL_Z)), -1),
}


def place_wheel_part(name, bm, mats, pos, side, smooth=35.0):
    m = Matrix.Diagonal((side, 1.0, 1.0, 1.0))
    bm.transform(m)
    if side < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    ob = new_object(name, bm, mats, smooth_angle=smooth)
    ob.location = pos
    return ob


def build_wheels():
    out = {}
    for key, (pos, side) in WHEELS.items():
        out["Tyre_" + key] = place_wheel_part("Tyre_" + key, build_tyre(), ["Rubber"], pos, side, 40)
        out["Rim_" + key] = place_wheel_part("Rim_" + key, build_rim(), ["Rim", "Chrome", "Badge"], pos, side, 40)
        out["Disc_" + key] = place_wheel_part("Disc_" + key, build_brake(), ["Brake", "Underbody"], pos, side)
        cal = place_wheel_part("Caliper_" + key, build_caliper(), ["Caliper"], pos, side)
        out["Caliper_" + key] = cal
        for k in ("Tyre_", "Rim_", "Disc_"):
            out[k + key]["assembly"] = "Wheel_" + key
        cal["assembly"] = "Caliper_" + key
    return out


def cut_wheel_wells(body):
    bm = bmesh.new()
    for key, (pos, side) in WHEELS.items():
        x0, x1 = (0.46, 1.4) if side > 0 else (-1.4, -0.46)
        c0 = Vector((x0, pos.y, ARCH_Z))
        c1 = Vector((x1, pos.y, ARCH_Z))
        add_cylinder(bm, c0, c1, ARCH_R, segs=72)
    cut = cutter_object("cut_wheelwells", bm, "Underbody")
    boolean(body, cut)
    delete_object(cut)


# --------------------------------------------------------------------------
# CSG helpers: orthographic volumes, copies, surface patches
# --------------------------------------------------------------------------

BIG = 4.0


def copy_object(ob, name):
    c = ob.copy()
    c.data = ob.data.copy()
    c.name = name
    bpy.context.scene.collection.objects.link(c)
    return c


def offset_poly(pts, d):
    """Offset a closed 2D polygon (list of (x, y)) by d (+ = outward)."""
    P = [v2(*p) for p in pts]
    n = len(P)
    area = sum(P[i].x * P[(i + 1) % n].y - P[(i + 1) % n].x * P[i].y for i in range(n))
    sgn = 1.0 if area > 0 else -1.0          # CCW -> outward normal is (dy, -dx)
    out = []
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        t1 = (b - a).normalized()
        t2 = (c - b).normalized()
        n1 = v2(t1.y, -t1.x) * sgn
        n2 = v2(t2.y, -t2.x) * sgn
        nn = (n1 + n2)
        if nn.length < 1e-6:
            nn = n1
        nn.normalize()
        cosh = max(0.3, nn.dot(n1))
        out.append(b + nn * (d / cosh))
    out = [(p.x, p.y) for p in out]
    if d > 0 and _self_intersects(out):      # a notch narrower than 2d folds over:
        return offset_poly(_convex_hull(pts), d)   # grow its (enclosing) hull instead
    return out


def _self_intersects(pts):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue                     # neighbours share a vertex
            c, e = pts[j], pts[(j + 1) % n]
            if (cross(a, b, c) * cross(a, b, e) < 0) and (cross(c, e, a) * cross(c, e, b) < 0):
                return True
    return False


def _convex_hull(pts):
    P = sorted(set((float(x), float(y)) for x, y in pts))

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h
    return half(P)[:-1] + half(P[::-1])[:-1]


def volume(name, front=None, side=None, plan=None, xr=(-BIG, BIG),
           yr=(-BIG, BIG), zr=(-1.0, BIG), mat="Housing", grow=0.0):
    """Closed volume = intersection of orthographic prisms.

    front: (x, z) outline extruded along Y over yr
    side:  (y, z) outline extruded along X over xr
    plan:  (x, y) outline extruded along Z over zr"""
    parts = []
    if front is not None:
        parts.append(prism_from_poly(offset_poly(front, grow) if grow else front, "y", *yr))
    if side is not None:
        parts.append(prism_from_poly(offset_poly(side, grow) if grow else side, "x", *xr))
    if plan is not None:
        parts.append(prism_from_poly(offset_poly(plan, grow) if grow else plan, "z", *zr))
    if not parts:
        bm = bmesh.new()
        add_box(bm, Vector(((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2)),
                (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]))
        parts.append(bm)
    base = cutter_object(name, parts[0], mat)
    for k, bm in enumerate(parts[1:]):
        other = cutter_object(name + f"_p{k}", bm, mat)
        boolean(base, other, "INTERSECT")
        delete_object(other)
    return base


def mirror_x(ob):
    """Mirror an object's mesh across X (in place) keeping normals outward."""
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.transform(Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0)))
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    return ob


def both_sides(ob):
    """Join ob with its X mirror (for symmetric cutters)."""
    m = copy_object(ob, ob.name + "_m")
    mirror_x(m)
    join_into(ob, [m])
    return ob


def join_into(target, others):
    others = [o for o in others if o is not None and o != target]
    if not others:
        return target
    with bpy.context.temp_override(active_object=target, object=target,
                                   selected_objects=[target] + others,
                                   selected_editable_objects=[target] + others):
        bpy.ops.object.join()
    return target


def fit_smooth(bm, degree=2, tol=0.004):
    """Project a gently curved patch (a glass pane) onto a least-squares
    polynomial height field over its mean plane, ignoring lumps (at most
    the worst third of the points are left out of the fit)."""
    import numpy as np
    P = np.array([v.co[:] for v in bm.verts])
    if len(P) < 12:
        return
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    U, W, N = vt[0], vt[1], vt[2]
    u, w, h = (P - c) @ U, (P - c) @ W, (P - c) @ N
    su, sw = max(np.abs(u).max(), 1e-6), max(np.abs(w).max(), 1e-6)
    A = np.stack([(u / su) ** i * (w / sw) ** j for i in range(degree + 1)
                  for j in range(degree + 1 - i)], 1)
    keep = np.ones(len(P), bool)
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(A[keep], h[keep], rcond=None)
        res = np.abs(h - A @ coef)
        keep = res <= max(tol, np.percentile(res, 67))
    P2 = c + np.outer(u, U) + np.outer(w, W) + np.outer(A @ coef, N)
    for v, p in zip(bm.verts, P2):
        v.co = Vector(p)
    # lumps that overhung fold flat when projected: drop those faces (others
    # cover the same spot), they would wreck the solidify
    bm.normal_update()
    n = Vector(N[:]) if np.mean([f.normal.dot(Vector(N[:])) for f in bm.faces]) > 0 else -Vector(N[:])
    bad = [f for f in bm.faces if f.calc_area() < 1e-9 or f.normal.dot(n) < 0.2]
    bmesh.ops.delete(bm, geom=bad, context="FACES")
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")


def surface_patch(source, cutter, name, inset=0.004, keep_mat=0, thickness=0.0,
                  mats=None, smooth=False):
    """The part of `source`'s surface inside `cutter` (e.g. a lens or a glass
    pane), pulled in along the normals by `inset`, optionally given thickness.
    smooth: refit it as one smooth curved surface (glass on a mesh body)."""
    tmp = copy_object(source, name)
    tmp.modifiers.clear()
    boolean(tmp, cutter, "INTERSECT")
    bm = bmesh.new()
    bm.from_mesh(tmp.data)
    drop = [f for f in bm.faces if f.material_index != keep_mat]
    bmesh.ops.delete(bm, geom=drop, context="FACES")
    if smooth:
        fit_smooth(bm)
        # a smooth pane needs few triangles
        bm.to_mesh(tmp.data)
        bm.free()
        decimate(tmp, min(1.0, 600 / max(1, len(tmp.data.polygons))))
        bm = bmesh.new()
        bm.from_mesh(tmp.data)
    bm.normal_update()
    for v in bm.verts:
        v.co -= v.normal * inset
    if thickness > 0:
        res = bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=thickness)
    for f in bm.faces:
        f.material_index = 0
    me = tmp.data
    me.materials.clear()
    for m in (mats or ["Glass"]):
        me.materials.append(MATS[m])
    bm.to_mesh(me)
    bm.free()
    shade_smooth(tmp, 40)
    tmp.hide_render = False
    tmp.display_type = "TEXTURED"
    return tmp


# --------------------------------------------------------------------------
# Body design outlines (all for the +X / left side; mirrored as needed)
# --------------------------------------------------------------------------

def sample(fn, a, b, n):
    return [fn(a + (b - a) * i / (n - 1)) for i in range(n)]


def fillet(points, radii, steps=5):
    """Polygon with selected corners filleted. radii: dict index -> radius."""
    pts = [v2(*p) for p in points]
    n = len(pts)
    out = []
    for i in range(n):
        r = radii.get(i, 0.0)
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        a, b = p0 - p1, p2 - p1
        if r <= 0 or a.length < 1e-6 or b.length < 1e-6:
            out.append(p1)
            continue
        r = min(r, a.length * 0.48, b.length * 0.48)
        s_, e_ = p1 + a.normalized() * r, p1 + b.normalized() * r
        for k in range(steps + 1):
            t = k / steps
            out.append(s_ * ((1 - t) ** 2) + p1 * (2 * (1 - t) * t) + e_ * (t * t))
    return [(p.x, p.y) for p in out]


def side_dlo_door():
    """Door glass daylight opening, side view (y, z)."""
    y0, y1 = -0.530, 0.545
    bottom = sample(lambda y: (y, Z_BELT(y) + 0.008), y0, y1, 6)
    top = sample(lambda y: (y, Z_RAIL(y) - 0.026), y1, -0.04, 6)
    apil = sample(lambda y: (y, Z_RAIL(y) - 0.028), -0.07, y0, 7)
    pts = bottom + top + apil
    radii = {0: 0.012, 5: 0.02, 6: 0.03, 11: 0.06, 17: 0.015}
    return fillet(pts, radii)


def side_dlo_quarter():
    """Fixed rear quarter glass: straight raked rear edge (from the reference)."""
    y0, y_tip = 0.612, 1.055
    bottom = sample(lambda y: (y, Z_BELT(y) + 0.008), y_tip, y0, 5)
    top = [(y0, Z_RAIL(y0) - 0.026), (0.70, 1.158), (0.88, 1.085), (1.00, 1.030),
           (y_tip, 0.995)]
    pts = bottom + top
    radii = {0: 0.012, 4: 0.012, 5: 0.025, 6: 0.04, 9: 0.02}
    return fillet(pts, radii)


def screen_edge_x(y):
    """x of the windscreen's side edge (plan view, +X side) at y."""
    if BODY_MESH:     # the reconstructed screen tapers more (measured on body/s15_body.glb)
        return 0.645 - 0.2875 * (y + 0.62)
    return X_RAIL(y) - 0.032


def plan_windshield():
    yb, yt = -0.700, -0.085
    if BODY_MESH:
        yt = -0.110
    side = sample(lambda y: (screen_edge_x(y), y), yb, yt, 7)
    pts = side + [(-x, y) for x, y in reversed(side)]
    radii = {0: 0.05, 6: 0.06, 7: 0.06, 13: 0.05}
    return fillet(pts, radii)


def plan_backlight():
    yt, yb = 0.905, 1.475
    side = sample(lambda y: (X_RAIL(y) - 0.040, y), yt, yb, 7)
    if BODY_MESH:     # the reconstruction's glass wraps back in plan (measured on
        side = sample(lambda y: (X_RAIL(y) - 0.040, y), yt, 1.535, 7)   # its texture)
        arc = [(0.588, 1.560), (0.556, 1.580), (0.500, 1.608), (0.450, 1.628),
               (0.400, 1.642), (0.300, 1.661), (0.150, 1.668)]
        half = side + arc
        pts = half + [(0.0, 1.670)] + [(-x, y) for x, y in reversed(half)]
        n = len(pts)
        return fillet(pts, {0: 0.07, 6: 0.02, n - 7: 0.02, n - 1: 0.07})
    pts = side + [(-x, y) for x, y in reversed(side)]
    radii = {0: 0.07, 6: 0.05, 7: 0.05, 13: 0.07}
    return fillet(pts, radii)


def side_door_outline():
    """Door skin outline, side view (y, z), including the window frame."""
    ya, yb = -0.690, 0.575
    apil = sample(lambda y: (y, Z_RAIL(y) - 0.007), ya, -0.08, 8)
    roof = sample(lambda y: (y, Z_RAIL(y) - 0.007), -0.05, yb, 5)
    rear = [(0.585, Z_BELT(0.585) - 0.012), (0.594, 0.70), (0.600, 0.315)]
    bottom = [(-0.640, 0.315)]
    front = [(-0.655, 0.60), (-0.672, 0.80), (ya, Z_BELT(ya) - 0.004)]
    pts = apil + roof + rear + bottom + front
    n_ap = len(apil)
    radii = {n_ap - 1: 0.05, n_ap: 0.05, n_ap + len(roof) - 1: 0.02,
             n_ap + len(roof) + 2: 0.03, n_ap + len(roof) + 3: 0.03}
    return fillet(pts, radii)


def plan_hood_outline():
    """Hood skin outline, plan view (x, y)."""
    yr = -0.800
    side = sample(lambda y: (X_RAIL(y) - 0.006, y), yr, -1.86, 7)
    front = [(0.62, -1.905), (0.46, -1.995), (0.31, -2.065), (0.15, -2.110), (0.0, -2.125)]
    half = side + front
    pts = half + [(-x, y) for x, y in reversed(half[:-1])]
    radii = {0: 0.02, len(pts) - 1: 0.02}
    return fillet(pts, radii)


def trunk_outlines():
    """Boot lid: plan outline (x, y) and rear outline (x, z). The lid ends in a
    straight edge between the tail lights' lower inner corners."""
    yf = 1.530
    plan = fillet([(-0.62, yf), (0.62, yf), (0.62, 2.6), (-0.62, 2.6)],
                  {0: 0.03, 1: 0.03})
    if BODY_MESH:     # front edge follows the rear glass, 4.5 cm behind it
        front = [(0.620, 1.592), (0.588, 1.605), (0.556, 1.625), (0.500, 1.653),
                 (0.450, 1.673), (0.400, 1.687), (0.300, 1.706), (0.150, 1.713)]
        plan = fillet([(-x, y) for x, y in front] + [(0.0, 1.715)] + front[::-1]
                      + [(0.62, 2.6), (-0.62, 2.6)], {0: 0.02, 16: 0.02})
    if BODY_MESH:     # lid edge runs just above the tail lights, then straight across
        half = [(0.326, 0.700), (0.329, 0.731), (0.346, 0.755), (0.367, 0.776),
                (0.390, 0.799), (0.414, 0.821), (0.438, 0.843), (0.462, 0.862),
                (0.486, 0.875), (0.508, 0.889), (0.62, 0.890), (0.62, 1.6)]
        rear = half + [(-x, z) for x, z in reversed(half)]
        return plan, fillet(rear, {0: 0.006, 1: 0.006, 22: 0.006, 23: 0.006})
    rear = [(-0.240, 0.702), (0.240, 0.702), (0.240, 0.795), (0.30, 0.807),
            (0.42, 0.829), (0.62, 0.869), (0.62, 1.6), (-0.62, 1.6), (-0.62, 0.869),
            (-0.42, 0.829), (-0.30, 0.807), (-0.240, 0.795)]
    rear = fillet(rear, {0: 0.012, 1: 0.012, 2: 0.012, 11: 0.012})
    return plan, rear


def head_front():
    """Headlight outline, front view (x, z), +X side: short squared inner end
    on the bonnet line, tall rounded outer end (from the reference)."""
    if BODY_MESH:     # measured on the reconstruction's texture; the outer end
        pts = [(0.320, 0.598), (0.338, 0.560), (0.40, 0.565), (0.50, 0.568),   # runs on
               (0.60, 0.571), (0.70, 0.577), (0.740, 0.581), (0.762, 0.598),   # for the
               (0.830, 0.600), (0.830, 0.666), (0.766, 0.664), (0.742, 0.680),  # wrap
               (0.700, 0.707), (0.665, 0.702), (0.600, 0.688), (0.530, 0.672),
               (0.470, 0.662), (0.400, 0.651), (0.372, 0.640), (0.345, 0.622)]
        return fillet(pts, {0: 0.008, 1: 0.01, 11: 0.012, 12: 0.01})
    pts = [(0.312, 0.590), (0.318, 0.528), (0.45, 0.530), (0.60, 0.536),
           (0.72, 0.548), (0.79, 0.566), (0.826, 0.600), (0.815, 0.645),
           (0.77, 0.667), (0.66, 0.660), (0.52, 0.638), (0.40, 0.610)]
    return fillet(pts, {0: 0.008, 1: 0.008, 5: 0.03, 6: 0.025, 7: 0.03, 8: 0.02})


def head_plan():
    if BODY_MESH:     # lamp's rear edge seen from above, 6 cm deeper for the housing
        return [(0.30, -2.6), (0.98, -2.6), (0.98, -1.66), (0.775, -1.662),
                (0.742, -1.734), (0.695, -1.77), (0.607, -1.842), (0.50, -1.914),
                (0.428, -1.95), (0.375, -1.986), (0.33, -2.03)]
    return [(0.29, -2.6), (0.98, -2.6), (0.98, -1.79), (0.78, -1.83),
            (0.55, -1.93), (0.40, -1.99), (0.29, -2.03)]


def head_side():
    """Headlight wrap seen from the side (y, z); open towards the front."""
    return [(-2.60, 0.552), (-2.10, 0.557), (-2.00, 0.567), (-1.921, 0.574),
            (-1.876, 0.575), (-1.831, 0.587), (-1.786, 0.604), (-1.741, 0.624),
            (-1.722, 0.645), (-1.741, 0.667), (-1.786, 0.680), (-1.831, 0.690),
            (-1.876, 0.700), (-1.92, 0.712), (-2.60, 0.712)]


def tail_rear():
    """Tail light outline, rear view (x, z), +X side (wraps onto the side):
    tall outer end, blunt rounded inner end (reference)."""
    if BODY_MESH:     # measured: pointed inner end, sloping top, wrap at the corner
        pts = [(0.333, 0.700), (0.70, 0.699), (0.97, 0.699), (0.97, 0.886),
               (0.80, 0.886), (0.775, 0.842), (0.757, 0.839), (0.733, 0.857),
               (0.70, 0.867), (0.65, 0.878), (0.585, 0.882), (0.513, 0.881),
               (0.493, 0.866), (0.469, 0.854), (0.445, 0.835), (0.421, 0.813),
               (0.397, 0.791), (0.374, 0.768), (0.353, 0.748), (0.336, 0.724)]
        return fillet(pts, {0: 0.004, 11: 0.012})
    pts = [(0.252, 0.788), (0.258, 0.728), (0.40, 0.708), (0.60, 0.698),
           (0.76, 0.698), (0.86, 0.703), (0.97, 0.712), (0.97, 0.878),
           (0.82, 0.886), (0.62, 0.862), (0.42, 0.822), (0.30, 0.800)]
    return fillet(pts, {0: 0.02, 1: 0.02, 5: 0.02, 8: 0.02})


def tail_plan():
    if BODY_MESH:     # lamp's front edge seen from above, 6 cm deeper for the housing
        return [(0.30, 2.08), (0.364, 2.07), (0.568, 2.033), (0.619, 1.961),
                (0.694, 1.925), (0.765, 1.90), (0.98, 1.89), (0.98, 2.7), (0.30, 2.7)]
    return [(0.18, 2.13), (0.40, 2.112), (0.62, 2.045), (0.74, 1.965),
            (0.84, 1.905), (0.98, 1.885), (0.98, 2.7), (0.18, 2.7)]


def tail_side():
    """Tail light wrap seen from the side (y, z); open towards the back."""
    return [(1.949, 0.839), (1.994, 0.874), (2.039, 0.880), (2.06, 0.886),
            (2.70, 0.886), (2.70, 0.697), (2.054, 0.698), (2.009, 0.702), (1.964, 0.769)]


def intake_main():
    """Central intake, front view (x, z): flat top, bulging sides."""
    if BODY_MESH:     # measured on the reconstruction
        return fillet([(-0.31, 0.252), (0.31, 0.252), (0.362, 0.298), (0.375, 0.348),
                       (0.34, 0.405), (-0.34, 0.405), (-0.375, 0.348), (-0.362, 0.298)],
                      {0: 0.03, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.03, 5: 0.03, 6: 0.02, 7: 0.02})
    return fillet([(-0.30, 0.252), (0.30, 0.252), (0.352, 0.295), (0.365, 0.345),
                   (0.33, 0.395), (-0.33, 0.395), (-0.365, 0.345), (-0.352, 0.295)],
                  {0: 0.03, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.03, 5: 0.03, 6: 0.02, 7: 0.02})


def intake_outer():
    """Outer bumper opening (honeycomb), front view (x, z), +X side."""
    pts = fillet([(0.50, 0.266), (0.625, 0.268), (0.662, 0.300), (0.668, 0.355),
                  (0.648, 0.392), (0.478, 0.386), (0.462, 0.330)],
                 {0: 0.025, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.03, 5: 0.025, 6: 0.02})
    if BODY_MESH:     # measured on the reconstruction
        pts = offset_poly([(0.471, 0.339), (0.468, 0.375), (0.488, 0.395), (0.524, 0.400),
                           (0.596, 0.402), (0.668, 0.396), (0.683, 0.372), (0.685, 0.336),
                           (0.676, 0.300), (0.655, 0.281), (0.619, 0.275), (0.547, 0.272),
                           (0.511, 0.274), (0.487, 0.289), (0.476, 0.313)], 0.003)
    return pts


def side_marker():
    """Amber side repeater just behind the front arch (rounded rectangle)."""
    return fillet([(-0.952, 0.538), (-0.898, 0.538), (-0.898, 0.562), (-0.952, 0.562)],
                  {0: 0.008, 1: 0.008, 2: 0.008, 3: 0.008}, steps=3)


def ellipse(cx, cy, a, b, n=24):
    return [(cx + a * math.cos(2 * math.pi * i / n), cy + b * math.sin(2 * math.pi * i / n))
            for i in range(n)]


# --------------------------------------------------------------------------
# Body construction: shell, cavities, openings, panels
# --------------------------------------------------------------------------

def wheel_cutters(extra_r=0.0, x_in=0.46, mat="Underbody", name="cut_wheels",
                  only=None):
    bm = bmesh.new()
    for key, (pos, side) in WHEELS.items():
        if only and key[0] not in only:
            continue
        x0, x1 = (x_in, 1.4) if side > 0 else (-1.4, -x_in)
        add_cylinder(bm, Vector((x0, pos.y, ARCH_Z)), Vector((x1, pos.y, ARCH_Z)),
                     ARCH_R + extra_r, segs=72, mat=0)
    return cutter_object(name, bm, mat)


def build_cavities(inner):
    """Cabin, engine bay and boot cavities, carved from the inner offset."""
    wt = wheel_cutters(0.030, 0.43, name="cut_wheels_thick", mat="Interior")
    wt_bay = wheel_cutters(0.030, 0.43, name="cut_wheels_bay", mat="Paint")
    hl_env, tl_env = lamp_volumes(0.022, "_env")
    both_sides(hl_env)
    both_sides(tl_env)
    cab_side = [(-0.80, 0.24), (0.95, 0.24), (1.05, 0.86), (1.44, 0.86),
                (1.44, 1.7), (-0.74, 1.7), (-0.74, 0.84), (-0.80, 0.84)]
    cab_front = [(-0.70, 0.20), (0.70, 0.20), (0.70, 0.40), (1.2, 0.47),
                 (1.2, 1.7), (-1.2, 1.7), (-1.2, 0.47), (-0.70, 0.40)]
    cab = volume("cav_cabin", front=cab_front, side=cab_side, mat="Interior")
    boolean(cab, inner, "INTERSECT")
    boolean(cab, wt)
    bay = volume("cav_bay", side=[(-2.07, 0.22), (-0.88, 0.22), (-0.88, 1.4),
                                  (-2.07, 1.4)], mat="Paint")
    boolean(bay, inner, "INTERSECT")
    boolean(bay, wt_bay)
    boolean(bay, hl_env)
    boot = volume("cav_boot", side=[(1.12, 0.33), (2.6, 0.33), (2.6, 1.6),
                                    (1.46, 1.6), (1.46, 0.82), (1.12, 0.82)],
                  mat="Interior")
    boolean(boot, inner, "INTERSECT")
    boolean(boot, wt)
    boolean(boot, tl_env)
    for ob in (wt, wt_bay, hl_env, tl_env):
        delete_object(ob)
    return cab, bay, boot


def lamp_volumes(grow=0.0, tag=""):
    """Headlight and tail light volumes for both sides (one object each)."""
    hl = volume("vol_head" + tag, front=head_front(), plan=head_plan(), grow=grow,
                side=head_side() if BODY_MESH else None,
                mat="Reflector", yr=(-2.7, -1.6), zr=(0.3, 1.2))
    tl = volume("vol_tail" + tag, front=tail_rear(), plan=tail_plan(), grow=grow,
                side=tail_side() if BODY_MESH else None,
                mat="Housing", yr=(1.8, 2.8), zr=(0.55, 1.2))
    return hl, tl


# --------------------------------------------------------------------------
# Body shape from a mesh file (BODY_MESH)
# --------------------------------------------------------------------------
# An image-to-3D model gets the surfacing right but arrives as one fused,
# arbitrarily scaled blob with the wheels, wing and mirrors moulded on. It is
# oriented, fitted to the real dimensions and axle positions, made symmetric
# and watertight, stripped of the parts this script models itself, and then
# goes through exactly the same cuts as the lofted body. Only its shape is
# used; textures are ignored.

def _verts(ob):
    import numpy as np
    v = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", v)
    return v.reshape(-1, 3)


def _set_verts(ob, V):
    ob.data.vertices.foreach_set("co", V.ravel())
    ob.data.update()


def import_mesh_file(path, name):
    """Import every mesh in a model file as one object (world space)."""
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise ValueError(f"unsupported body mesh: {path}")
    new = [o for o in bpy.data.objects if o not in before]
    prepared = any(o.get("s15_body") or (o.type == "MESH" and o.data.get("s15_body"))
                   for o in new)
    bm = bmesh.new()
    for o in new:
        if o.type == "MESH":
            t = bmesh.new()
            t.from_mesh(o.data)
            t.transform(o.matrix_world)
            add_bm(bm, t)
    meshes = {o.data for o in new if o.type == "MESH"}
    for o in new:
        bpy.data.objects.remove(o, do_unlink=True)
    for me in meshes:
        bpy.data.meshes.remove(me)
    if not bm.faces:
        raise ValueError(f"no mesh found in {path}")
    ob = new_object(name, bm, ["Paint"], smooth_angle=None)
    if prepared:
        ob["s15_body"] = 1
    return ob


def fit_body_mesh(ob):
    """Z up, nose to -Y, wheels on the real axles, real length/width/height.

    The overhangs are stretched to the real ones with a scale that eases in
    from 1 at the axle line, so the wheelbase section is only scaled
    uniformly."""
    import numpy as np
    V = _verts(ob)
    ext = V.max(0) - V.min(0)
    if ext[0] > ext[1]:                          # length along X: turn 90 deg
        V = V[:, [1, 0, 2]] * np.array([-1.0, 1.0, 1.0])
    lo, hi = V.min(0), V.max(0)
    V -= np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    H = V[:, 2].max()

    def width(z0, z1):
        m = (V[:, 2] >= z0) & (V[:, 2] <= z1)
        return np.ptp(V[m, 0]) if m.any() else 0.0

    if width(0.85 * H, H) > width(0.0, 0.15 * H):    # roof is narrower than the floor
        V[:, 2] = H - V[:, 2]
        V[:, 0] *= -1
    L = np.ptp(V[:, 1])
    y0 = V[:, 1].min()

    def top(a, b):
        m = (V[:, 1] >= y0 + a * L) & (V[:, 1] <= y0 + b * L)
        return V[m, 2].max()

    if top(0.0, 0.1) > top(0.9, 1.0):            # nose is lower than the boot
        V[:, 0] *= -1
        V[:, 1] *= -1
    # wheel centres from the tyre contact patches
    c = V[V[:, 2] < 0.012 * L, 1]
    mid = (c.min() + c.max()) / 2
    f, r = c[c < mid], c[c >= mid]
    yf, yr = (f.min() + f.max()) / 2, (r.min() + r.max()) / 2
    V *= WHEELBASE / (yr - yf)
    V[:, 1] -= (yf + yr) / 2 * WHEELBASE / (yr - yf)
    band = (V[:, 2] > 0.25) & (V[:, 2] < 0.80)       # below mirrors, above the floor
    nose, tail = V[band, 1].min(), V[band, 1].max()
    # height and width
    V[:, 2] *= HEIGHT / V[:, 2].max()
    side = (V[:, 2] > 0.40) & (V[:, 2] < 0.80)
    V[:, 0] *= (WIDTH / 2) / np.abs(V[side, 0]).max()
    # overhangs: y -> axle + (y - axle) + k * ease integral
    grid = np.linspace(-4.0, 4.0, 8001)
    out = grid.copy()
    for axle, tip, real in ((Y_FAX, nose, Y_NOSE), (Y_RAX, tail, Y_TAIL)):
        sgn = 1.0 if real > axle else -1.0
        u = np.clip((grid - axle) * sgn / 0.5, 0.0, 1.0)
        w = u * u * (3 - 2 * u)                  # 0 at the axle, 1 from 0.5 m out
        I = np.cumsum(w) * (grid[1] - grid[0])
        I -= np.interp(axle, grid, I)
        I_tip = np.interp(tip, grid, I)
        k = ((real - axle) - (tip - axle)) / I_tip if abs(I_tip) > 1e-9 else 0.0
        zone = (grid - axle) * sgn > 0
        out[zone] += k * I[zone]
    V[:, 1] = np.interp(V[:, 1], grid, out)
    _set_verts(ob, V)
    print(f"[body] fitted: axles {yf:.3f}/{yr:.3f} -> {Y_FAX:.3f}/{Y_RAX:.3f}, "
          f"overhangs {Y_FAX - nose:.3f}/{tail - Y_RAX:.3f} (scaled) -> "
          f"{Y_FAX - Y_NOSE:.3f}/{Y_TAIL - Y_RAX:.3f}")


def symmetrize_x(ob):
    """Keep the -X (right) half and mirror it: the stock exhaust exits on the
    left, and this script adds its own."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                           plane_co=(0, 0, 0), plane_no=(-1, 0, 0), clear_inner=True)
    bmesh.ops.mirror(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                     merge_dist=1e-5, axis="X")
    bm.to_mesh(ob.data)
    bm.free()


def taubin(ob, iters=6, lam=0.5, mu=-0.53):
    """Non-shrinking smoothing (irons out reconstruction ripples)."""
    import numpy as np
    V = _verts(ob)
    E = np.empty(len(ob.data.edges) * 2, dtype=np.int64)
    ob.data.edges.foreach_get("vertices", E)
    E = E.reshape(-1, 2)
    deg = np.maximum(np.bincount(E.ravel(), minlength=len(V)), 1)[:, None]

    def lap(P):
        S = np.zeros_like(P)
        np.add.at(S, E[:, 0], P[E[:, 1]])
        np.add.at(S, E[:, 1], P[E[:, 0]])
        return S / deg - P

    for _ in range(iters):
        V = V + lam * lap(V)
        V = V + mu * lap(V)
    _set_verts(ob, V)


# large, gently curved panels: their outer skin is pulled onto a smooth
# fitted surface, which removes the reconstruction's lumps that glossy paint
# shows up (and what is left of the moulded wing's feet on the boot lid).
# "top" panels are heights z(x, y) seen from above, "rear" ones depths y(x, z)
# seen from behind; (half-width in x, range of the other coordinate)
FAIR_PANELS = {"hood": ("top", 0.64, -1.98, -0.90), "roof": ("top", 0.52, -0.10, 0.88),
               "boot": ("top", 0.64, 1.50, 2.10), "tail": ("rear", 0.50, 0.71, 0.87)}


def fair_panels(ob, edge=0.05, deg_x=4, deg_v=7):
    import numpy as np
    from mathutils.bvhtree import BVHTree
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    V = _verts(ob)
    N = np.empty(len(V) * 3)
    ob.data.vertices.foreach_get("normal", N)
    N = N.reshape(-1, 3)
    for name, (kind, xh, v0, v1) in FAIR_PANELS.items():
        # h: the coordinate that is faired, v: the one along the panel
        h, v, ray = (2, 1, Vector((0, 0, -1))) if kind == "top" else (1, 2, Vector((0, -1, 0)))
        sgn = -ray[h]                           # +1: the outer skin is the highest h
        x = V[:, 0]
        cand = (np.abs(x) < xh) & (V[:, v] > v0) & (V[:, v] < v1) & (sgn * N[:, h] > 0.4)
        idx = np.nonzero(cand)[0]
        outer = []
        for i in idx:                           # outer skin only (not the far side)
            o = Vector(V[i])
            o[h] = sgn * 5.0
            hit = tree.ray_cast(o, ray)[0]
            outer.append(hit is not None and sgn * (V[i, h] - hit[h]) > -0.004)
        sel = idx[np.array(outer, bool)] if len(idx) else idx
        if name == "boot":                      # not the rear glass (cut away; and
            bl = offset_poly(plan_backlight(), 0.012)   # its edge is a step)
            sel = sel[[not _in_poly(V[i, 0], V[i, 1], bl) for i in sel]]
        xn = V[sel, 0] / xh
        vn = (V[sel, v] - v0) / (v1 - v0) * 2 - 1
        A = np.stack([xn ** (2 * i) * vn ** j for i in range(deg_x) for j in range(deg_v)], 1)
        hv = V[sel, h]
        keep = np.ones(len(sel), bool)
        for _ in range(6):                      # robust: ignore the lumps
            coef, *_ = np.linalg.lstsq(A[keep], hv[keep], rcond=None)
            r = A @ coef - hv
            keep = np.abs(r) < max(3 * 1.4826 * np.median(np.abs(r[keep])), 0.002)
        d = np.minimum.reduce([xh - np.abs(V[sel, 0]), V[sel, v] - v0, v1 - V[sel, v]])
        w = (smoothstep_np(0.0, edge, d) * smoothstep_np(0.4, 0.7, sgn * N[sel, h])
             * (1 - smoothstep_np(0.02, 0.035, np.abs(r))))   # (where the fit fails)
        V[sel, h] += w * r
        print(f"[body] faired {name}: {len(sel)} vertices, "
              f"rms {1000 * np.sqrt(np.mean((w * r) ** 2)):.1f} mm, "
              f"max {1000 * np.abs(w * r).max():.1f} mm")
    _set_verts(ob, V)


def self_intersections(ob):
    """Faces of ob that cut through a face they share no vertex with."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    tree = BVHTree.FromBMesh(bm)
    bad = set()
    for a, b in tree.overlap(tree):
        if a != b and not set(bm.faces[a].verts) & set(bm.faces[b].verts):
            bad.update((a, b))
    faces = [[v.index for v in bm.faces[i].verts] for i in bad]
    bm.free()
    return faces


def untangle(ob, V0, rounds=4):
    """Put the vertices around any self-intersection back where they were
    before smoothing (V0), widening the patch until the mesh is clean."""
    import numpy as np
    E = np.empty(len(ob.data.edges) * 2, dtype=np.int64)
    ob.data.edges.foreach_get("vertices", E)
    E = E.reshape(-1, 2)
    for k in range(rounds):
        faces = self_intersections(ob)
        if not faces:
            return
        mask = np.zeros(len(V0), bool)
        mask[[i for f in faces for i in f]] = True
        for _ in range(2 + 2 * k):              # grow the patch
            grow = mask.copy()
            grow[E[mask[E[:, 0]], 1]] = True
            grow[E[mask[E[:, 1]], 0]] = True
            mask = grow
        V = _verts(ob)
        V[mask] = V0[mask]
        _set_verts(ob, V)
        print(f"[body] untangled {len(faces)} self-intersecting faces ({mask.sum()} vertices)")
    if self_intersections(ob):
        print("[body] WARNING: self-intersections left; booleans may fail")


def smoothstep_np(a, b, t):
    import numpy as np
    t = np.clip((t - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _robust_fit(A, b, iters=5, tol=0.008):
    """Least squares that ignores samples sitting more than tol *above* the
    fit (spoiler pedestals, mirror stalks): returns fitted values."""
    import numpy as np
    keep = np.isfinite(b)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        fit = A @ coef
        keep = np.isfinite(b) & (b - fit < tol)
    return fit, keep


def field_cutter(name, us, vs, H, to_xyz, far):
    """Closed solid between the sampled surface h = H[j, i] over the grid
    (us[i], vs[j]) and the plane h = far. to_xyz(u, v, h) -> point."""
    bm = bmesh.new()
    nu, nv = len(us), len(vs)
    lo = [[bm.verts.new(to_xyz(us[i], vs[j], H[j, i])) for i in range(nu)] for j in range(nv)]
    hi = [[bm.verts.new(to_xyz(us[i], vs[j], far)) for i in range(nu)] for j in range(nv)]
    for j in range(nv - 1):
        for i in range(nu - 1):
            bm.faces.new((lo[j][i], lo[j + 1][i], lo[j + 1][i + 1], lo[j][i + 1]))
            bm.faces.new((hi[j][i], hi[j][i + 1], hi[j + 1][i + 1], hi[j + 1][i]))
    rim = ([(0, i) for i in range(nu)] + [(j, nu - 1) for j in range(1, nv)] +
           [(nv - 1, i) for i in range(nu - 2, -1, -1)] + [(j, 0) for j in range(nv - 2, 0, -1)])
    for k, a in enumerate(rim):
        b = rim[(k + 1) % len(rim)]
        bm.faces.new((lo[a[0]][a[1]], lo[b[0]][b[1]], hi[b[0]][b[1]], hi[a[0]][a[1]]))
    return cutter_object(name, bm, "Paint")


def strip_wipers(ob):
    """Shave the moulded wipers off the scuttle and the base of the screen
    (this script adds its own). The surface height is ray-cast from inside
    on dense rows across the car; each row is fitted with a smooth crown
    curve that ignores samples sitting on top of it. Only inside the
    screen's side edges: outboard of them the feet of the A-pillars stand
    above the crown, and shaving them off left the lower pillars flat with
    a step where the cut ended (a kinked highlight along the screen edge)."""
    import numpy as np
    bvh = bvh_of(ob)
    xs = np.linspace(-0.78, 0.78, 157)          # 1 cm: a clean edge at the pillars
    ys = np.arange(-0.86, -0.575, 0.005)
    H = np.full((len(ys), len(xs)), np.nan)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            hit = bvh.ray_cast(Vector((x, y, 0.50)), Vector((0, 0, 1)))
            if hit[0] is not None and hit[1].z > 0:      # exited the body (started inside)
                H[j, i] = hit[0].z
    A = np.stack([np.ones_like(xs), xs ** 2, xs ** 4, xs ** 6], 1)
    for j in range(len(ys)):
        ok = np.isfinite(H[j])
        if ok.sum() > 8:
            fit, keep = _robust_fit(A, H[j], tol=0.006)
            inside = np.abs(xs) < screen_edge_x(max(ys[j], -0.70)) - 0.005
            H[j] = np.where(keep | ~ok | ~inside, H[j], fit)
    # no body under a sample: leave it alone (the cutter rises out of the way)
    H = np.where(np.isfinite(H), H + 0.003, 1.9)
    cut = field_cutter("cut_wipers", xs, ys, H, lambda u, v, h: (u, v, h), 2.0)
    boolean(ob, cut)
    delete_object(cut)


def strip_spoiler(ob):
    """Cut a moulded rear spoiler off the boot lid (this script adds its own).
    The lid height is ray-cast from inside on dense rows across the car. Under
    the spoiler's feet (|x| 0.46..0.66) the rays exit through the spoiler, so
    there the height comes from a smooth crown fit to the rest of the row;
    behind the lid's rear lip the last lid height is held, so the cut never
    reaches the tail panel or the lights."""
    import numpy as np
    bvh = bvh_of(ob)
    xs = np.linspace(-0.70, 0.70, 57)
    ys = np.arange(1.66, Y_TAIL + 0.08, 0.008)
    H = np.full((len(ys), len(xs)), np.nan)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            hit = bvh.ray_cast(Vector((x, y, 0.50)), Vector((0, 0, 1)))
            if hit[0] is not None and hit[1].z > 0:      # exited the body (started inside)
                H[j, i] = hit[0].z
    A = np.stack([np.ones_like(xs), xs ** 2, xs ** 4, xs ** 6], 1)
    feet = (np.abs(xs) > 0.46) & (np.abs(xs) < 0.66)
    centre = np.abs(xs) < 0.30
    prev, prev_mid, held = None, None, False
    for j in range(len(ys)):
        ok = np.isfinite(H[j]) & ~feet
        if ok.sum() > 10:
            coef, *_ = np.linalg.lstsq(A[ok], H[j][ok], rcond=None)
            fit = A @ coef
            H[j] = np.where(feet & np.isfinite(H[j]) & (H[j] > fit + 0.006), fit, H[j])
        mid = np.nanmedian(H[j][centre]) if np.isfinite(H[j][centre]).any() else np.nan
        if prev is not None and (held or not np.isfinite(mid) or mid < prev_mid - 0.008):
            held = True
            H[j] = prev
        else:
            prev, prev_mid = H[j].copy(), mid
    # no body under a sample: leave it alone (the cutter rises out of the way)
    H = np.where(np.isfinite(H), H + 0.006, 1.9)
    cut = field_cutter("cut_spoiler", xs, ys, H, lambda u, v, h: (u, v, h), 2.0)
    boolean(ob, cut)
    delete_object(cut)


def strip_mirrors(ob):
    """Cut the moulded door mirrors off just outside the door/window surface
    (sampled from inside with the stalks fitted out). Never closer than
    10 cm (in plan) outboard of the screen's side edge: the fitted surface
    runs inside the A-pillar above the window line, and cutting there
    planed the pillar's outer side flat up to the cutter's top (z 1.10),
    leaving a notch where the cut ended."""
    import numpy as np
    bvh = bvh_of(ob)
    ys = np.linspace(-0.74, -0.14, 31)
    pts, hx = [], []
    for z in np.linspace(0.78, 1.10, 33):
        for y in ys:
            if z > Z_RAIL(y) - 0.02:             # above the window frame: outside the body
                continue
            hit = bvh.ray_cast(Vector((0.30, y, z)), Vector((1, 0, 0)))
            if hit[0] is not None and hit[1].x > 0:
                pts.append((y, z))
                hx.append(hit[0].x)
    P, hx = np.array(pts), np.array(hx)
    def basis(y, z):
        return np.stack([np.ones_like(y), y, z, y * z, z * z, y * y], -1)
    _, keep = _robust_fit(basis(P[:, 0], P[:, 1]), hx, tol=0.006)
    coef, *_ = np.linalg.lstsq(basis(P[keep, 0], P[keep, 1]), hx[keep], rcond=None)
    zs = np.linspace(0.845, 1.10, 18)            # below this is door skin (my mirror covers it)
    Y, Z = np.meshgrid(ys, zs)
    H = np.maximum(basis(Y, Z) @ coef + 0.005, screen_edge_x(Y) + 0.10)
    cut = field_cutter("cut_mirrors", zs, ys, H.T, lambda u, v, h: (h, v, u), 1.4)
    both_sides(cut)
    boolean(ob, cut)
    delete_object(cut)


def drop_islands(ob):
    """Keep only the largest connected piece. After the voxel remesh,
    anything not touching the body (wheels, cut-off spoiler tips, exhaust
    tips) is a separate piece, and the script models all of those itself."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    seen, islands = set(), []
    for v in bm.verts:
        if v in seen:
            continue
        stack, isl = [v], []
        seen.add(v)
        while stack:
            a = stack.pop()
            isl.append(a)
            for e in a.link_edges:
                b = e.other_vert(a)
                if b not in seen:
                    seen.add(b)
                    stack.append(b)
        islands.append(isl)
    small = sorted(islands, key=len)[:-1]
    if small:
        bmesh.ops.delete(bm, geom=[v for isl in small for v in isl], context="VERTS")
        print(f"[body] dropped {len(small)} loose pieces")
    bm.to_mesh(ob.data)
    bm.free()


def sdf_remesh(ob, isos, voxel):
    """Rebuild ob from its signed distance field (OpenVDB, via geometry
    nodes), once per entry of isos: the surface at that signed distance (m,
    negative = inwards) from the previous one. The result is watertight, and
    unlike moving vertices along normals, offsets never fold over."""
    try:
        bpy.context.preferences.experimental.use_new_volume_nodes = True
    except AttributeError:
        pass
    ng = bpy.data.node_groups.new("sdf_remesh", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    src = ng.nodes.new("NodeGroupInput").outputs[0]
    for iso in isos:
        to_grid = ng.nodes.new("GeometryNodeMeshToSDFGrid")
        to_grid.inputs["Voxel Size"].default_value = voxel
        to_grid.inputs["Band Width"].default_value = int(abs(iso) / voxel) + 3
        ng.links.new(src, to_grid.inputs["Mesh"])
        to_mesh = ng.nodes.new("GeometryNodeGridToMesh")
        to_mesh.inputs["Threshold"].default_value = iso
        ng.links.new(to_grid.outputs[0], to_mesh.inputs["Grid"])
        src = to_mesh.outputs[0]
    ng.links.new(src, ng.nodes.new("NodeGroupOutput").inputs[0])
    mats = list(ob.data.materials)
    mod = ob.modifiers.new("sdf", "NODES")
    mod.node_group = ng
    apply_modifiers(ob)
    bpy.data.node_groups.remove(ng)
    if not ob.data.polygons:
        raise RuntimeError(f"{ob.name}: distance-field remesh came out empty")
    ob.data.materials.clear()
    for m in mats:
        ob.data.materials.append(m)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    if bm.calc_volume(signed=True) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(ob.data)
    bm.free()


def _in_poly(x, y, poly):
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i - 1], poly[i]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def in_glass(grow=0.03):
    """Test (point, normal) -> True inside the (grown) glass areas."""
    sides = [offset_poly(side_dlo_door(), grow), offset_poly(side_dlo_quarter(), grow)]
    ws, bl = offset_poly(plan_windshield(), grow), offset_poly(plan_backlight(), grow)

    def test(p, n):
        if abs(n.x) > 0.5:
            return abs(p.x) > 0.35 and any(_in_poly(p.y, p.z, s) for s in sides)
        return ((p.z > 0.70 and _in_poly(p.x, p.y, ws)) or
                (p.z > 0.93 and _in_poly(p.x, p.y, bl)))
    return test


def conform(bm, target, reach=0.10, skip=None):
    """Move each vertex of bm along its normal onto target's outer surface
    when that is within reach. Skipped or unmatched vertices (open windows,
    spoiler pedestals, deep openings) get a smooth harmonic fill from their
    neighbours, so e.g. a window becomes a surface spanning its frame."""
    import numpy as np
    bvh = bvh_of(target)
    bm.normal_update()
    bm.verts.index_update()
    verts = bm.verts[:]
    P = np.array([v.co[:] for v in verts])
    N = np.array([v.normal[:] for v in verts])
    d = np.full(len(verts), np.nan)
    for i, v in enumerate(verts):
        p, n = v.co, v.normal
        if skip and skip(p, n):
            continue
        best = None
        hit = bvh.ray_cast(p, n, reach)
        if hit[0] is not None:
            if hit[1].dot(n) > 0.3:                # leaving the target: its outer face
                best = hit[3]
            else:                                  # under a thin panel: its outer face
                h2 = bvh.ray_cast(hit[0] + n * 1e-4, n, 0.02)
                if h2[0] is not None and h2[1].dot(n) > 0.3:
                    best = hit[3] + h2[3]
        hit = bvh.ray_cast(p, -n, reach)
        if hit[0] is not None and hit[1].dot(n) > 0.3 and (best is None or hit[3] < abs(best)):
            best = -hit[3]
        if best is not None:
            d[i] = best
    E = np.array([(e.verts[0].index, e.verts[1].index) for e in bm.edges])
    deg = np.maximum(np.bincount(E.ravel(), minlength=len(verts)), 1)

    def nb_mean(x):
        s = np.zeros_like(x)
        np.add.at(s, E[:, 0], x[E[:, 1]])
        np.add.at(s, E[:, 1], x[E[:, 0]])
        return s / deg

    known = np.isfinite(d)
    print(f"[body] conform: {known.mean() * 100:.0f}% of vertices matched")
    x = np.where(known, d, 0.0)
    # drop isolated outliers (a ray that found a seat through a window edge)
    k = known.astype(float)
    s = np.zeros(len(verts))
    np.add.at(s, E[:, 0], x[E[:, 1]])
    np.add.at(s, E[:, 1], x[E[:, 0]])
    c = np.zeros(len(verts))
    np.add.at(c, E[:, 0], k[E[:, 1]])
    np.add.at(c, E[:, 1], k[E[:, 0]])
    nbr = np.where(c > 0, s / np.maximum(c, 1), x)
    known &= np.abs(x - nbr) < 0.02
    x = np.where(known, d, np.nanmean(d))
    for _ in range(3000):
        x = np.where(known, x, nb_mean(x))
    # larger patches that sit well below their surroundings in the glasshouse
    # (sun visors, mirror, seat backs seen through an opening) are not skin
    upper = P[:, 2] > 0.85
    for _ in range(2):
        low = x.copy()
        for _ in range(300):
            low = nb_mean(low)
        known &= ~(upper & (np.abs(x - low) > 0.025))
        x = np.where(known, d, x)
        for _ in range(3000):
            x = np.where(known, x, nb_mean(x))
    for i, v in enumerate(verts):
        v.co = Vector(P[i] + N[i] * x[i])


def body_from_mesh(path):
    """Watertight body solid from a model file.

    Open windows (a mesh with a modelled interior) are closed with the
    lofted body pulled onto the mesh's window frames, which also fills the
    cabin so the usual cavities hollow it out. A small distance-field
    closing fills the mesh's own panel-line grooves (the panels are cut
    again here, where the doors and lids really split)."""
    ob = import_mesh_file(path, "Body")
    print(f"[body] {os.path.basename(path)}: {tri_count(ob)} triangles")
    bm = bmesh.new()                            # glTF splits vertices at UV seams
    bm.from_mesh(ob.data)                       # and sharp edges: weld them again
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-5 * max(ob.dimensions))
    bm.to_mesh(ob.data)
    bm.free()
    if ob.get("s15_body"):                      # already prepared (prepare_body)
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        check_manifold(bm, "body mesh")
        bm.free()
        shade_smooth(ob, 40)
        return ob
    fit_body_mesh(ob)                           # (finds the axles from the tyres)
    drop_islands(ob)                            # loose wheels
    symmetrize_x(ob)
    # cabin filler: the loft, matched to the mesh, a few mm under its skin
    bm = build_loft()
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
    conform(bm, ob, skip=in_glass())
    fill = new_object("CabinFill", bm, ["Paint"], smooth_angle=None)
    sdf_remesh(fill, [-0.004], 0.01)
    decimate(fill, min(1.0, 60000 / tri_count(fill)))
    box = volume("cabin_box", yr=(-0.80, 1.50), zr=(0.45, 2.0), mat="Paint")
    boolean(fill, box, "INTERSECT")
    delete_object(box)
    join_into(ob, [fill])
    sdf_remesh(ob, [0.01, -0.01], BODY_VOXEL)
    decimate(ob, min(1.0, 4 * BODY_TRIS / max(1, tri_count(ob))))
    strip_wipers(ob)
    strip_mirrors(ob)
    strip_spoiler(ob)
    drop_islands(ob)
    taubin(ob, 40)                             # iron out reconstruction ripples
    sdf_remesh(ob, [0.0], BODY_VOXEL)          # watertight (and untangled) again
    decimate(ob, min(1.0, BODY_TRIS / max(1, tri_count(ob))))
    V0 = _verts(ob)
    taubin(ob, 4)                              # light: more folds thin lips over
    fair_panels(ob)
    untangle(ob, V0)                           # (exact booleans fail on folds)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    check_manifold(bm, "body mesh")
    bm.free()
    shade_smooth(ob, 40)
    return ob


def prepare_body(src, dst):
    """Run body_from_mesh on a raw model file (e.g. straight from an
    image-to-3D service) and save the resulting body solid as a small glTF
    that later builds load as is."""
    reset_scene()
    setup_materials()
    global BODY_MESH
    BODY_MESH = src
    ob = body_from_mesh(src)
    ob["s15_body"] = 1
    ob.data["s15_body"] = 1
    shade_smooth(ob, 180)                       # no sharp edges: keeps vertices shared
    _select_only([ob], ob)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", use_selection=True,
                              export_materials="NONE", export_extras=True)
    print(f"[body] prepared {tri_count(ob)} triangles -> {dst}")


def resolve_body_mesh():
    """Absolute BODY_MESH path, or None (lofted body) if unset or missing."""
    global BODY_MESH
    if BODY_MESH and not os.path.isabs(BODY_MESH):
        here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        BODY_MESH = os.path.join(here, BODY_MESH)
    if BODY_MESH and not os.path.exists(BODY_MESH):
        print(f"[body] {BODY_MESH} not found, lofting the body instead")
        BODY_MESH = None
    if BODY_MESH and not hasattr(bpy.types, "GeometryNodeMeshToSDFGrid"):
        print("[body] this Blender has no SDF grid nodes (needs 4.2+), lofting the body instead")
        BODY_MESH = None


def build_body():
    """Returns dict with the body shell and the reference solid."""
    if BODY_MESH:
        body = body_from_mesh(BODY_MESH)
    else:
        body = new_object("Body", build_loft(), ["Paint"], smooth_angle=None)
    ref = copy_object(body, "BodyRef")          # untouched outer solid
    ref.hide_render = True

    def offset(name, t, mat, iters=10):
        if BODY_MESH:                           # exact offsets from the distance field
            ob = copy_object(body, name)
            sdf_remesh(ob, [-t], 0.01)
            decimate(ob, min(1.0, 25000 / tri_count(ob)))   # cavity walls: rarely seen
            ob.data.materials.clear()
            ob.data.materials.append(MATS[mat])
            return ob
        bm = build_loft(2)
        inner_offset(bm, t, iters=iters)
        return new_object(name, bm, [mat], smooth_angle=None)

    inner = offset("Inner", 0.028, "Interior")
    layer_in = offset("LayerIn", 0.036, "Paint")
    # the skin layer must poke out past the body surface: exact booleans
    # do not like coincident faces
    layer = offset("Layer", -0.015, "Paint", iters=4)
    boolean(layer, layer_in)
    delete_object(layer_in)
    for ob in (inner, layer):
        ob.hide_render = True
    cab, bay, boot = build_cavities(inner)
    delete_object(inner)
    w = wheel_cutters()
    boolean(body, w)
    delete_object(w)
    for c in (cab, bay, boot):
        boolean(body, c)
        delete_object(c)
    return {"body": body, "ref": ref, "layer": layer}


def cut_windows(body, ref):
    """Cut the glass openings and return the glass panes."""
    glass = {}
    # side glass: door glass (moves with each door) and fixed quarter glass
    for key, outline in (("Door", side_dlo_door()), ("Quarter", side_dlo_quarter())):
        for sd, xr in (("L", (0.35, 1.3)), ("R", (-1.3, -0.35))):
            cut = volume(f"cut_{key}_{sd}", side=outline, xr=xr, mat="BlackMatte")
            glass[f"Glass_{key}_{sd}"] = surface_patch(
                ref, cut, f"Glass_{key}_{sd}", inset=0.007, thickness=0.004,
                smooth=bool(BODY_MESH))
            boolean(body, cut)
            delete_object(cut)
    cut = volume("cut_windshield", plan=plan_windshield(), zr=(0.70, 1.8), mat="BlackMatte")
    glass["Glass_Windshield"] = surface_patch(ref, cut, "Glass_Windshield", inset=0.006,
                                        thickness=0.005, smooth=bool(BODY_MESH))
    boolean(body, cut)
    delete_object(cut)
    cut = volume("cut_backlight", plan=plan_backlight(), zr=(0.93, 1.8), mat="BlackMatte")
    glass["Glass_Rear"] = surface_patch(ref, cut, "Glass_Rear", inset=0.006, thickness=0.005,
                                        smooth=bool(BODY_MESH))
    boolean(body, cut)
    delete_object(cut)
    return glass


def ray_surface(bvh, origin, direction):
    hit = bvh.ray_cast(Vector(origin), Vector(direction).normalized())
    return hit[0]


def paint_faces(ob, test, mat_name):
    """Give the faces whose centre and normal pass test(c, n) a material."""
    me = ob.data
    if MATS[mat_name].name not in [m.name for m in me.materials if m]:
        me.materials.append(MATS[mat_name])
    idx = [m.name if m else "" for m in me.materials].index(MATS[mat_name].name)
    count = 0
    for poly in me.polygons:
        if test(poly.center, poly.normal):
            poly.material_index = idx
            count += 1
    return count


def split_faces(ob, chords, near):
    """Cut ob's faces along a chain of planes, so that a paint_faces
    boundary on them comes out as a clean line instead of the zigzag of
    the mesh's triangles. chords: [(plane_co, plane_no, select(c))], each
    plane cutting only the faces whose centre c passes select(c) and
    near(c, n) (re-evaluated after the previous cuts)."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    for co, no, select in chords:
        faces = [f for f in bm.faces if select(f.calc_center_median()) and
                 near(f.calc_center_median(), f.normal)]
        if not faces:
            continue
        edges = list({e for f in faces for e in f.edges})
        verts = list({v for f in faces for v in f.verts})
        bmesh.ops.bisect_plane(bm, geom=faces + edges + verts, dist=2e-4,
                               plane_co=Vector(co), plane_no=Vector(no).normalized())
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


def split_by_plane(ob, co, no, name_neg, name_pos):
    """Split a mesh object by a plane into two objects (neg / pos side)."""
    res = []
    for keep_pos, nm in ((False, name_neg), (True, name_pos)):
        c = copy_object(ob, nm)
        bm = bmesh.new()
        bm.from_mesh(c.data)
        g = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(bm, geom=g, plane_co=co, plane_no=no,
                               clear_inner=keep_pos, clear_outer=not keep_pos)
        bm.to_mesh(c.data)
        bm.free()
        res.append(c)
    delete_object(ob)
    return res


PLATE_RECESS_Y = Y_TAIL - 0.040     # back wall of the rear plate recess


def cut_lamps_and_openings(body, ref):
    """Headlight/tail-light pockets with lenses, bumper intakes,
    side marker pockets, rear plate recess."""
    out = {}
    hl, tl = lamp_volumes()
    for side, sx in (("L", 1), ("R", -1)):
        h = copy_object(hl, "vh_" + side)
        t = copy_object(tl, "vt_" + side)
        if sx < 0:
            mirror_x(h)
            mirror_x(t)
        out["Lens_Head_" + side] = surface_patch(ref, h, "Lens_Head_" + side, inset=0.003,
                                                 thickness=0.003, mats=["LensClear"])
        lens = surface_patch(ref, t, "LensT_" + side, inset=0.003, thickness=0.003,
                             mats=["LensRed"])
        # tail lamp zones (reference): amber indicator strip along the top,
        # clear reverse lamp at the inner end, red tail/brake below
        if BODY_MESH:   # smoked lens; reverse zone is the whole pointed inner end
            inner_, rest = split_by_plane(lens, Vector((0.488 * sx, 0, 0)), Vector((sx, 0, 0)),
                                          "Light_Reverse_" + side, "tmp_outer_" + side)
            outer, mid = split_by_plane(rest, Vector((0, 0, 0.806)), Vector((0, 0, 1)),
                                        "Light_Brake_" + side, "Light_Indicator_R" + side)
            zone_mats = (("LensSmoke", "TailAmber", "LensRed"))
        else:
            lower, mid = split_by_plane(lens, Vector((0, 0, 0.802)), Vector((0, 0, 1)),
                                        "tmp_lower_" + side, "Light_Indicator_R" + side)
            inner_, outer = split_by_plane(lower, Vector((0.40 * sx, 0, 0)), Vector((sx, 0, 0)),
                                           "Light_Reverse_" + side, "Light_Brake_" + side)
            zone_mats = ("LensClear", "LensAmber", "LensRed")
        for ob, mname in zip((inner_, mid, outer), zone_mats):
            ob.data.materials.clear()
            ob.data.materials.append(MATS[mname])
        out[inner_.name], out[mid.name], out[outer.name] = inner_, mid, outer
        boolean(body, h)
        boolean(body, t)
        delete_object(h)
        delete_object(t)
    delete_object(hl)
    delete_object(tl)
    # bumper openings (open into the engine bay behind)
    cut = volume("cut_intake", front=intake_main(), yr=(-2.8, -2.02), mat="BlackMatte")
    boolean(body, cut)
    delete_object(cut)
    cut = volume("cut_outer", front=intake_outer(), yr=(-2.8, -2.03), mat="BlackMatte")
    both_sides(cut)
    boolean(body, cut)
    delete_object(cut)
    if BODY_MESH:
        # the reconstruction's intake recesses and scuttle panel are black
        # trim, not body colour
        polys = [offset_poly(intake_main(), -0.003), offset_poly(intake_outer(), -0.003),
                 [(-x, z) for x, z in offset_poly(intake_outer(), -0.003)]]
        paint_faces(body, lambda c, n: -2.25 < c.y < -1.85 and
                    any(_in_poly(c.x, c.z, q) for q in polys), "BlackMatte")
        # scuttle: between the bonnet's rear edge and the screen, and only
        # inboard of the bonnet's side line carried on back to the screen
        # (parallel to the screen's edge). Outboard of it are the A-pillar
        # feet and the wing tops behind the bonnet's rear corners, which
        # stay body colour; the faces are split along that line first so
        # the black ends in a clean edge.
        def cowl_x(y):
            return screen_edge_x(y) - 0.043        # bonnet side line at y -0.80
        k = (screen_edge_x(-0.80) - screen_edge_x(-0.70)) / 0.10    # its slope, -dx/dy
        split_faces(body, [((sx * cowl_x(-0.75), -0.75, 0.85), (sx, k, 0.0),
                            lambda c, sx=sx: sx * c.x > 0) for sx in (1, -1)],
                    lambda c, n: -0.83 < c.y < -0.66 and c.z > 0.78 and
                    abs(abs(c.x) - cowl_x(c.y)) < 0.05)
        paint_faces(body, lambda c, n: -0.80 < c.y < -0.69 and c.z > 0.84 and
                    abs(c.x) < cowl_x(c.y) and n.z > 0.3, "BlackMatte")
    # side repeaters on the front fenders
    cut = volume("cut_marker", side=side_marker(), xr=(0.775, 1.3), mat="Housing")
    out["Light_Indicator_SL"] = surface_patch(ref, cut, "Light_Indicator_SL", inset=0.001,
                                              thickness=0.004, mats=["LensAmber"])
    m = copy_object(cut, "cut_marker_r")
    mirror_x(m)
    out["Light_Indicator_SR"] = surface_patch(ref, m, "Light_Indicator_SR", inset=0.001,
                                              thickness=0.004, mats=["LensAmber"])
    join_into(cut, [m])
    boolean(body, cut)
    delete_object(cut)
    # rear plate recess
    yb = PLATE_RECESS_Y
    cut = volume("cut_plate", front=fillet([(-0.19, 0.368), (0.19, 0.368), (0.19, 0.552),
                                            (-0.19, 0.552)], {0: 0.02, 1: 0.02, 2: 0.02, 3: 0.02}),
                 plan=[(-0.3, yb), (0.3, yb), (0.3, 2.6), (-0.3, 2.6)],
                 yr=(1.9, 2.7), zr=(0.25, 0.75), mat="Paint")
    boolean(body, cut)
    delete_object(cut)
    return out


def split_panel(body, layer, name, gap=0.0022, mats=None, exclude=None, **vol_kw):
    """Separate an opening panel (door/hood/boot lid) from the body shell.
    exclude: a volume that stays with the body whatever the outline says."""
    inner_v = volume("pv_" + name, grow=-gap, mat="Paint", **vol_kw)
    boolean(inner_v, layer, "INTERSECT")
    if exclude:
        boolean(inner_v, exclude)
    panel = copy_object(body, name)
    boolean(panel, inner_v, "INTERSECT")
    delete_object(inner_v)
    outer_v = volume("pg_" + name, grow=gap, mat="Paint", **vol_kw)
    boolean(outer_v, layer, "INTERSECT")
    if exclude:
        boolean(outer_v, exclude)
    saved, nm0 = body.data.copy(), nonmanifold_edges(body)
    boolean(body, outer_v)
    if nonmanifold_edges(body) > nm0:  # the exact solver sometimes doubles faces where
        old = body.data               # a cut meets cavity walls, and a non-manifold
        body.data = saved             # body breaks every later cut: redo it in
        bpy.data.meshes.remove(old)   # self-intersection mode
        boolean(body, outer_v, use_self=True)
        print(f"[panels] {name}: gap cut redone in self-intersection mode "
              f"({nonmanifold_edges(body)} non-manifold edges left)")
    else:
        bpy.data.meshes.remove(saved)
    delete_object(outer_v)
    return panel


def nonmanifold_edges(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    n = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return n


def build_panels(body, layer):
    panels = {}
    dl = split_panel(body, layer, "Door_L", side=side_door_outline(), xr=(0.30, 1.3))
    dr = split_panel(body, layer, "Door_R", side=side_door_outline(), xr=(-1.3, -0.30))
    panels["Door_L"], panels["Door_R"] = dl, dr
    # only the band under the bonnet top, and never the headlight housings
    # below its front corners
    ys = [-2.40 + 0.05 * k for k in range(37)]
    band = [(y, Z_TOP(max(y, -2.30)) - 0.10) for y in ys] + [(ys[-1], 2.0), (ys[0], 2.0)]
    hl, tl = lamp_volumes(0.026, "_hood")
    delete_object(tl)
    both_sides(hl)
    panels["Hood"] = split_panel(body, layer, "Hood", plan=plan_hood_outline(), side=band,
                                 zr=(0.50, 2.0), exclude=hl)
    delete_object(hl)
    plan, rear = trunk_outlines()
    panels["Trunk"] = split_panel(body, layer, "Trunk", plan=plan, front=rear,
                                  zr=(0.60, 2.0), yr=(1.40, 2.8))
    return panels


# --------------------------------------------------------------------------
# Exterior details
# --------------------------------------------------------------------------

def decimate(ob, ratio):
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.ratio = ratio
    with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob]):
        bpy.ops.object.modifier_apply(modifier=mod.name)
    shade_smooth(ob, 40)


def bvh_of(ob):
    bm = bm_from_object(ob)
    t = BVHTree.FromBMesh(bm)
    bm.free()
    return t


def ray(bvh, origin, direction):
    """(location, normal) of the first hit or (None, None)."""
    h = bvh.ray_cast(Vector(origin), Vector(direction).normalized())
    return (h[0], h[1]) if h[0] is not None else (None, None)


def orient_faces(faces, direction):
    for f in faces:
        if f.is_valid and f.normal.dot(direction) < 0:
            f.normal_flip()


def frame_matrix(origin, x_axis, z_axis):
    """4x4 matrix from an origin, a local X axis and a local Z (normal)."""
    x = Vector(x_axis).normalized()
    z = Vector(z_axis).normalized()
    y = z.cross(x).normalized()
    x = y.cross(z).normalized()
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = Vector(origin)
    return m


def bowl(bm, center, facing, R, depth, mat=0, segs=24):
    """Paraboloid reflector bowl opening towards `facing`."""
    prof = [(R, 0.0), (R * 0.93, -0.12 * depth), (R * 0.79, -0.36 * depth),
            (R * 0.56, -0.63 * depth), (R * 0.30, -0.87 * depth), (R * 0.08, -depth)]
    q = Vector((1, 0, 0)).rotation_difference(Vector(facing))
    xf = Matrix.Translation(center) @ q.to_matrix().to_4x4()
    faces = revolve(prof, segs, mat=mat, bm=bm, xform=xf)
    bm.normal_update()
    # inner surface must face the lens
    for f in faces:
        c = f.calc_center_median()
        axis_pt = Vector(center) + Vector(facing).normalized() * (Vector(facing).normalized().dot(c - Vector(center)))
        if f.normal.dot(axis_pt - c) < 0:
            f.normal_flip()
    return faces


def disc(bm, center, facing, r, mat=0, segs=24, dome=0.0, thick=0.0):
    """Closed lens disc / dome facing `facing`."""
    prof = [(0.0, dome), (r * 0.5, dome * 0.75), (r * 0.85, dome * 0.3), (r, 0.0),
            (r, -max(thick, 0.002)), (0.0, -max(thick, 0.002))]
    q = Vector((1, 0, 0)).rotation_difference(Vector(facing))
    xf = Matrix.Translation(center) @ q.to_matrix().to_4x4()
    return revolve(prof, segs, closed=True, mat=mat, bm=bm, xform=xf)


def ring(bm, center, facing, r_in, r_out, depth, mat=0, segs=32):
    prof = [(r_in, 0.0), (r_out, 0.0), (r_out, -depth), (r_in, -depth)]
    q = Vector((1, 0, 0)).rotation_difference(Vector(facing))
    xf = Matrix.Translation(center) @ q.to_matrix().to_4x4()
    return revolve(prof, segs, closed=True, mat=mat, bm=bm, xform=xf)


def clip_line_poly(p0, d, poly):
    """Inside intervals (t0, t1) of the line p0 + t d against a polygon."""
    ts = []
    n = len(poly)
    for i in range(n):
        a, b = v2(*poly[i]), v2(*poly[(i + 1) % n])
        e = b - a
        den = d.x * e.y - d.y * e.x
        if abs(den) < 1e-12:
            continue
        w = a - p0
        t = (w.x * e.y - w.y * e.x) / den
        u = (w.x * d.y - w.y * d.x) / den
        if 0 <= u < 1:
            ts.append(t)
    ts.sort()
    return [(ts[i], ts[i + 1]) for i in range(0, len(ts) - 1, 2)]


def diamond_grille(bm, poly, y, pitch=0.024, bar=0.0045, thick=0.008, mat=0,
                   facing=-1):
    """Diamond mesh grille filling a front/rear-view polygon at depth y."""
    poly = offset_poly(poly, 0.004)
    xs = [p[0] for p in poly]
    zs = [p[1] for p in poly]
    x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
    span = (x1 - x0) + (z1 - z0)
    for sgn in (1, -1):
        d = v2(1.0, sgn).normalized()
        nrm = v2(-d.y, d.x)
        k = -span
        while k <= span:
            p0 = v2((x0 + x1) / 2, (z0 + z1) / 2) + nrm * k
            for t0, t1 in clip_line_poly(p0, d, poly):
                a, b = p0 + d * t0, p0 + d * t1
                L = (b - a).length
                if L < 0.004:
                    continue
                c = (a + b) / 2
                ang = math.atan2(d.y, d.x)
                rot = Matrix.Rotation(-ang, 3, "Y")
                yy = y + (0.0 if sgn > 0 else facing * -0.003)
                add_box(bm, Vector((c.x, yy, c.y)), (L, thick, bar), mat=mat, rot=rot)
            k += pitch / math.sqrt(2)


def text_mesh(text, size, extrude, align="CENTER", shear=0.0):
    """Text -> bmesh in the XY plane (reads along +X, faces +Z)."""
    cu = bpy.data.curves.new("txt", "FONT")
    cu.body = text
    cu.size = size
    cu.extrude = extrude
    cu.align_x = align
    cu.align_y = "CENTER"
    cu.shear = shear
    cu.resolution_u = 3
    ob = bpy.data.objects.new("txt", cu)
    bpy.context.scene.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.curves.remove(cu)
    bm = bmesh.new()
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    return bm


def add_bm(dst, src, matrix=None, mat=0):
    """Append bmesh src into dst (optionally transformed)."""
    me = bpy.data.meshes.new("tmp")
    if matrix is not None:
        src.transform(matrix)
    src.to_mesh(me)
    src.free()
    for p in me.polygons:
        p.material_index = mat
    dst.from_mesh(me)
    bpy.data.meshes.remove(me)


def stroke_poly(points, w):
    """Closed outline of a polyline stroked with half-width w."""
    P = [v2(*p) for p in points]
    left, right = [], []
    for i, p in enumerate(P):
        a = P[max(i - 1, 0)]
        b = P[min(i + 1, len(P) - 1)]
        t = (b - a).normalized()
        n = v2(-t.y, t.x)
        left.append(p + n * w)
        right.append(p - n * w)
    return [(q.x, q.y) for q in left + right[::-1]]


def cut_seams(body, layer):
    """Painted-panel shut lines that do not open (bumpers, fuel flap)."""
    seams = [
        # front bumper / wing seam: headlight corner down into the wheel arch
        [(-1.865, 0.705), (-1.815, 0.640), (-1.745, 0.575), (-1.665, 0.520),
         (-1.590, 0.478), (-1.540, 0.455)],
        # rear bumper / quarter seam: tail light corner down into the arch
        [(1.990, 0.812), (1.950, 0.742), (1.860, 0.660), (1.740, 0.590),
         (1.630, 0.535), (1.560, 0.505)],
    ]
    for i, pts in enumerate(seams):
        vol = volume(f"seam{i}", side=stroke_poly(pts, 0.0019), xr=(0.40, 1.3), mat="Paint")
        both_sides(vol)
        boolean(vol, layer, "INTERSECT")
        boolean(body, vol)
        delete_object(vol)
    # fuel filler flap (car's right rear quarter)
    rr = fillet([(0.985, 0.705), (1.125, 0.705), (1.125, 0.800), (0.985, 0.800)],
                {0: 0.02, 1: 0.02, 2: 0.02, 3: 0.02})
    outer = volume("fuel_o", side=offset_poly(rr, 0.0018), xr=(-1.3, -0.822), mat="Paint")
    inner = volume("fuel_i", side=offset_poly(rr, -0.0018), xr=(-1.4, -0.80), mat="Paint")
    boolean(outer, inner)
    delete_object(inner)
    boolean(body, outer)
    delete_object(outer)


def build_exterior_details(ref):
    """Returns dict name -> object (each tagged with its assembly)."""
    bvh = bvh_of(ref)
    parts = {}

    def make(name, bm, mats, assembly="Body", smooth=40):
        ob = new_object(name, bm, mats, smooth_angle=smooth)
        ob["assembly"] = assembly
        parts[name] = ob
        return ob

    for side, sx in (("L", 1), ("R", -1)):
        F = "F" + side
        # ---- headlight internals --------------------------------------
        chrome, emit, amber, dark = bmesh.new(), bmesh.new(), bmesh.new(), bmesh.new()
        if BODY_MESH:   # measured: reflector bowl, projector bowl, amber indicator
            units = ((0.420, 0.607, 0.042, "high"), (0.565, 0.620, 0.050, "proj"),
                     (0.688, 0.625, 0.033, "ind"))
        else:
            units = ((0.445, 0.575, 0.030, "high"), (0.600, 0.592, 0.035, "proj"),
                     (0.762, 0.610, 0.025, "ind"))
        for (x, z, R, kind) in units:
            hit, _ = ray(bvh, (x * sx, -4.0, z), (0, 1, 0))
            ys = hit.y + 0.028
            c = Vector((x * sx, ys + 0.004, z))
            if BODY_MESH:                       # chrome bezel ring around each unit
                ring(chrome, c + Vector((0, -0.003, 0)), (0, -1, 0), R - 0.003, R + 0.004, 0.006)
            if kind == "ind":
                if BODY_MESH:
                    bowl(chrome, c + Vector((0, 0.004, 0)), (0, -1, 0), R, 0.035)
                    disc(amber, c + Vector((0, 0.006, 0)), (0, -1, 0), R * 0.72, dome=0.010)
                else:
                    bowl(amber, c, (0, -1, 0), R, 0.035)
                disc(amber, c + Vector((0, 0.020, 0)), (0, -1, 0), 0.008, dome=0.006)
                continue
            bowl(chrome, c, (0, -1, 0), R, 0.05)
            if kind == "proj":
                ring(chrome, c + Vector((0, -0.004, 0)), (0, -1, 0), 0.024, 0.031, 0.04)
                disc(emit, c + Vector((0, -0.006, 0)), (0, -1, 0), 0.024, dome=0.012, thick=0.01)
            else:
                disc(emit, c + Vector((0, 0.020, 0)), (0, -1, 0), 0.007, dome=0.006)
                add_cylinder(dark, c + Vector((0, 0.03, 0)), c + Vector((0, 0.05, 0)), 0.012, 12)
        make("HeadReflector_" + side, chrome, ["Reflector"])
        make("Light_Head_" + side, emit, ["Emitter"])
        make("Light_Indicator_" + F, amber, ["LensAmber"])
        make("HeadBulbs_" + side, dark, ["Housing"])

        # ---- tail light internals (rings behind the red lens) --------
        rings, red, clear = bmesh.new(), bmesh.new(), bmesh.new()
        if BODY_MESH:   # reference: three round reds, clear reverse at the pointed end
            units = ((0.700, 0.755, 0.036, "red"), (0.618, 0.755, 0.036, "red"),
                     (0.536, 0.755, 0.036, "red"), (0.440, 0.748, 0.031, "clear"))
            # amber oval reflector behind the top strip
            hit, _ = ray(bvh, (0.628 * sx, 4.0, 0.834), (0, -1, 0))
            ov = bmesh.new()
            prism_from_poly([(x * sx, z) for x, z in ellipse(0.628, 0.834, 0.118, 0.020, 28)],
                            "y", hit.y - 0.024, hit.y - 0.016, bm=ov)
            bmesh.ops.recalc_face_normals(ov, faces=ov.faces)
            make("TailAmberReflector_" + side, ov, ["AmberReflector"])
        else:
            units = ((0.555, 0.750, 0.037, "red"), (0.705, 0.752, 0.037, "red"),
                     (0.335, 0.752, 0.026, "clear"))
        for (x, z, r, kind) in units:
            hit, _ = ray(bvh, (x * sx, 4.0, z), (0, -1, 0))
            c = Vector((x * sx, hit.y - 0.020, z))
            ring(rings, c, (0, 1, 0), r, r + 0.006, 0.014)
            bowl(rings, c - Vector((0, 0.004, 0)), (0, 1, 0), r, 0.022)
            disc(red if kind == "red" else clear, c - Vector((0, 0.010, 0)), (0, 1, 0), r * 0.92,
                 dome=0.010, thick=0.004)
        make("TailReflector_" + side, rings, ["Reflector"])
        make("TailBulbs_" + side, red, ["BulbRed"])
        make("TailReverseBulb_" + side, clear, ["LensClear"])

        # ---- door mirror (rides on the door) ---------------------------
        mbm, mglass, mbase = bmesh.new(), bmesh.new(), bmesh.new()
        secs = []
        xs = [0.785, 0.805, 0.842, 0.885, 0.918, 0.938, 0.948]
        sc = [0.55, 0.82, 0.97, 1.0, 0.96, 0.82, 0.55]
        for x, k in zip(xs, sc):
            sec = []
            for i in range(16):
                a = 2 * math.pi * i / 16
                cy, cz = math.cos(a), math.sin(a)
                yy = 0.050 * k * cy
                if cy > 0:
                    yy = 0.034 * k * (abs(cy) ** 0.35)
                zz = 0.052 * k * (abs(cz) ** 0.8) * (1 if cz >= 0 else -1)
                sec.append(Vector((x * sx, -0.430 + yy, 0.925 + zz)))
            secs.append(sec)
        if sx < 0:
            secs = [s_[::-1] for s_ in secs]
        loft_tube(mbm, None, secs, mat=0)
        bmesh.ops.recalc_face_normals(mbm, faces=mbm.faces)
        add_box(mglass, Vector((0.870 * sx, -0.430 + 0.0345, 0.925)), (0.122, 0.003, 0.082),
                mat=0, bevel=0.0)
        # stalk + sail base (black triangle ahead of the door glass)
        add_box(mbase, Vector((0.780 * sx, -0.448, 0.900)), (0.044, 0.060, 0.030), bevel=0.006)
        add_box(mbase, Vector((0.766 * sx, -0.560, 0.888)), (0.016, 0.110, 0.048), bevel=0.004)
        d = "Door_" + side
        make("Mirror_" + side, mbm, ["Paint"], d)
        make("MirrorGlass_" + side, mglass, ["Mirror"], d)
        make("MirrorBase_" + side, mbase, ["BlackGloss"], d)

        # ---- door handle flap + key lock (pocket is cut in the door) -
        hb = bmesh.new()
        hit, nrm = ray(bvh, (2.0 * sx, 0.465, 0.762), (-sx, 0, 0))
        add_box(hb, hit - Vector((0.005 * sx, 0, 0)), (0.010, 0.112, 0.020), bevel=0.003)
        make("Handle_" + side, hb, ["Paint"], d)
        kb = bmesh.new()
        hit, nrm = ray(bvh, (2.0 * sx, 0.556, 0.763), (-sx, 0, 0))
        disc(kb, hit + Vector((0.001 * sx, 0, 0)), (sx, 0, 0), 0.0085, dome=0.002, thick=0.004)
        make("KeyLock_" + side, kb, ["Chrome"], d)

        # ---- B-pillar black appliqué ---------------------------------
        bp = volume("bp_" + side, side=[(0.589, Z_BELT(0.589) - 0.002), (0.611, Z_BELT(0.611) - 0.002),
                                        (0.611, Z_RAIL(0.611) - 0.022), (0.589, Z_RAIL(0.589) - 0.020)],
                    xr=(0.45, 1.3) if sx > 0 else (-1.3, -0.45))
        ob = surface_patch(ref, bp, "BPillar_" + side, inset=-0.0025, thickness=0.003,
                           mats=["BlackGloss"])
        delete_object(bp)
        ob["assembly"] = "Body"
        parts[ob.name] = ob

    # ---- grilles -----------------------------------------------------
    gr = bmesh.new()

    def deepest(poly):
        """y of the most rearward body point around an opening's rim."""
        ys = []
        for x, z in offset_poly(poly, 0.01):
            hit, _ = ray(bvh, (x, -4.0, z), (0, 1, 0))
            if hit is not None:
                ys.append(hit.y)
        return max(ys)

    yg = deepest(intake_main()) + 0.035
    for k in range(9):
        z = 0.262 + k * 0.0155
        for t0, t1 in clip_line_poly(v2(0.0, z), v2(1.0, 0.0), offset_poly(intake_main(), 0.004)):
            add_box(gr, Vector(((t0 + t1) / 2, yg, z)), (t1 - t0, 0.012, 0.004))
    add_box(gr, Vector((0.0, yg + 0.005, 0.323)), (0.012, 0.014, 0.15))
    for sx in (1, -1):
        poly = [(x * sx, z) for x, z in intake_outer()]
        if sx < 0:
            poly = poly[::-1]
        diamond_grille(gr, poly, deepest(poly) + 0.025)
    make("Grille", gr, ["BlackMatte"], smooth=None)

    # ---- wipers --------------------------------------------------------
    wb = bmesh.new()
    for px, L in ((-0.43, 0.56), (0.06, 0.50)):
        piv = Vector((px, -0.758, 0.0))
        hit, _ = ray(bvh, (px, -0.758, 3.0), (0, 0, -1))
        piv.z = hit.z + 0.012
        add_cylinder(wb, piv - Vector((0, 0, 0.02)), piv + Vector((0, 0, 0.012)), 0.014, 12)
        pts = []
        for i in range(9):
            t = i / 8
            x = px + 0.04 + L * t
            y = lerp(-0.742, -0.668, min(1.0, t * 3))
            hit, _ = ray(bvh, (x, y, 3.0), (0, 0, -1))
            pts.append(Vector((x, y, hit.z + 0.012)))
        for a, b in zip(pts[:-1], pts[1:]):
            add_cylinder(wb, a, b, 0.0055, 6)
        # blade
        bl = []
        for i in range(11):
            t = i / 10
            x = px + 0.06 + L * 0.95 * t
            hit, _ = ray(bvh, (x, -0.662, 3.0), (0, 0, -1))
            bl.append(Vector((x, -0.662, hit.z + 0.008)))
        for a, b in zip(bl[:-1], bl[1:]):
            add_box(wb, (a + b) / 2, ((b - a).length + 0.002, 0.012, 0.012),
                    rot=(b - a).to_track_quat("X", "Z").to_matrix())
    make("Wipers", wb, ["BlackMatte"])

    # ---- Spec R rear spoiler (rides on the boot lid) ------------------
    # (a mesh body's lumpy moulded spoiler is cut off in body_from_mesh and
    # this one goes where the reference has it: thin blade, long sloped fins)
    if BODY_MESH:
        half, le_y, le_z, chord, pitch0, x_ped, w_ped = 0.62, 1.935, 1.045, 0.195, 2.0, 0.555, 0.024
    else:
        half, le_y, le_z, chord, pitch0, x_ped, w_ped = 0.71, 1.870, 1.026, 0.248, 5.0, 0.47, 0.018
    wing, wlight = bmesh.new(), bmesh.new()
    foil = [(0.0, 0.0), (0.014, 0.011), (0.05, 0.019), (0.10, 0.022), (0.15, 0.020),
            (0.20, 0.014), (0.235, 0.007), (0.248, 0.004), (0.248, -0.002),
            (0.20, -0.002), (0.12, -0.004), (0.06, -0.005), (0.018, -0.004)]
    foil = [(cy * chord / 0.248, cz) for cy, cz in foil]
    secs = []
    span = [i / 20 for i in range(21)]
    for u in span:
        x = lerp(-half, half, u)
        ax = abs(x)
        rise = 0.014 * smoothstep(half - 0.19, half, ax) ** 1.4
        sweep = 0.010 * smoothstep(half - 0.19, half, ax)
        le = Vector((x, le_y + sweep, le_z + rise))
        pitch = math.radians(pitch0 + 5.0 * smoothstep(half - 0.16, half, ax))
        sec = []
        for (cy, cz) in foil:
            yy = cy * math.cos(pitch) + cz * math.sin(pitch)
            zz = cy * math.sin(pitch) + cz * math.cos(pitch)
            sec.append(le + Vector((0.0, yy, zz)))
        secs.append(sec)
    loft_tube(wing, None, secs, mat=0)
    for x0 in (-x_ped, x_ped):
        hit, _ = ray(bvh, (x0, 1.99, 3.0), (0, 0, -1))
        zd = hit.z
        if BODY_MESH:
            prof = fillet([(1.745, zd - 0.02), (2.045, zd - 0.02), (2.115, le_z + 0.002),
                           (1.955, le_z + 0.008), (1.935, le_z)], {2: 0.01, 3: 0.02, 4: 0.03})
        else:
            prof = fillet([(1.700, zd - 0.02), (2.085, zd - 0.02), (2.080, 1.040),
                           (1.905, 1.034), (1.870, 1.026)], {2: 0.01, 3: 0.02, 4: 0.03})
        prism_from_poly(prof, "x", x0 - w_ped, x0 + w_ped, bm=wing)
    bmesh.ops.recalc_face_normals(wing, faces=wing.faces)
    if BODY_MESH:       # stop lamp across the trailing edge
        te = Vector((0.0, le_y + chord * math.cos(math.radians(pitch0)) + 0.002,
                     le_z + chord * math.sin(math.radians(pitch0)) + 0.004))
        add_box(wlight, te, (0.21, 0.006, 0.020), bevel=0.002)
    else:
        add_box(wlight, Vector((0.0, 1.975, 1.050)), (0.22, 0.040, 0.006), bevel=0.002,
                rot=Matrix.Rotation(math.radians(-5), 3, "X"))
    make("Wing", wing, ["Paint"], "Trunk", smooth=45)
    make("Light_Brake_C", wlight, ["LensRed"], "Trunk")

    # ---- exhaust (single tip, car's left) ------------------------------
    ex, exd = bmesh.new(), bmesh.new()
    hit, _ = ray(bvh, (0.44, 4.0, 0.31), (0, -1, 0))
    yend = (hit.y if hit else 2.08) + 0.035
    tip0, tip1 = Vector((0.44, yend - 0.24, 0.248)), Vector((0.44, yend, 0.248))
    prof = [(0.040, 0.0), (0.046, 0.0), (0.0475, 0.232), (0.046, 0.240), (0.041, 0.240),
            (0.040, 0.232)]
    q = Vector((1, 0, 0)).rotation_difference(Vector((0, 1, 0)))
    xf = Matrix.Translation(tip0) @ q.to_matrix().to_4x4()
    revolve(prof, 32, closed=True, bm=ex, xform=xf)
    disc(exd, tip1 - Vector((0, 0.06, 0)), (0, 1, 0), 0.040, thick=0.004)
    make("ExhaustTip", ex, ["Exhaust"], smooth=50)
    make("ExhaustSoot", exd, ["BlackMatte"])

    # ---- license plates (JDM 330 x 165) --------------------------------
    for key, (ctr, facing) in {"Front": (None, -1), "Rear": (None, 1)}.items():
        if key == "Front":
            hit, _ = ray(bvh, (0.0, -4.0, 0.43), (0, 1, 0))
            hit2, _ = ray(bvh, (0.15, -4.0, 0.43), (0, 1, 0))
            yp = min(hit.y, hit2.y) - 0.010
            c = Vector((0.0, yp, 0.357))
            xa = Vector((1, 0, 0))
        else:
            c = Vector((0.0, PLATE_RECESS_Y + 0.0045, 0.460))
            xa = Vector((-1, 0, 0))
        m = frame_matrix(c, xa, Vector((0, facing, 0)))
        pl = bmesh.new()
        add_box(pl, Vector((0, 0, 0)), (0.330, 0.165, 0.006), bevel=0.004)
        add_box(pl, Vector((0, 0, 0.0035)), (0.318, 0.004, 0.0015), bevel=0.0)  # emboss rib
        pl.transform(m)
        make("Plate_" + key, pl, ["PlateWhite"])
        tb = bmesh.new()
        t1 = text_mesh("SILVIA 300", 0.036, 0.0012)
        t1.transform(Matrix.Translation((0, 0.050, 0.004)))
        t2 = text_mesh("15-15", 0.085, 0.0016)
        t2.transform(Matrix.Translation((0.012, -0.018, 0.004)))
        t3 = text_mesh("S", 0.040, 0.0012)
        t3.transform(Matrix.Translation((-0.125, -0.015, 0.004)))
        for t in (t1, t2, t3):
            add_bm(tb, t)
        bolts = bmesh.new()
        for bx in (-0.115, 0.115):
            disc(bolts, Vector((bx, 0.062, 0.004)), (0, 0, 1), 0.0065, dome=0.002, thick=0.002)
        tb.transform(m)
        bolts.transform(m)
        make("PlateText_" + key, tb, ["PlateText"], smooth=None)
        make("PlateBolts_" + key, bolts, ["Chrome"])
        if key == "Rear":
            pl_l = bmesh.new()
            for bx in (-0.08, 0.08):
                add_box(pl_l, Vector((bx, PLATE_RECESS_Y + 0.004, 0.548)), (0.05, 0.010, 0.006),
                        bevel=0.002)
            make("Light_Plate", pl_l, ["LensClear"], "Body")

    # ---- badges -------------------------------------------------------
    hit, nrm = ray(bvh, (0.0, -2.075, 3.0), (0, 0, -1))
    b = text_mesh("S", 0.075, 0.0025, shear=0.35)
    m = frame_matrix(hit + nrm * 0.001, Vector((1, 0, 0)), nrm)
    b.transform(m)
    make("Badge_S", b, ["Badge"], "Hood", smooth=None)
    boot_badges, red_r = bmesh.new(), bmesh.new()
    # (sizes and places measured on the reference for a mesh body)
    k_rnd, z_rnd = (1.0, 0.816) if BODY_MESH else (1.0, 0.822)
    script_at, script_wh = ((0.030, 0.738), (0.27, 0.030)) if BODY_MESH else ((0.075, 0.752), None)
    spec_at, spec_wh = ((-0.200, 0.736), (0.075, 0.022)) if BODY_MESH else ((-0.150, 0.750), None)

    def fit_box(bm, wh):
        """Scale a text mesh to a width and height (the script is wide and low)."""
        if wh:
            xs = [v.co.x for v in bm.verts]
            ys = [v.co.y for v in bm.verts]
            bm.transform(Matrix.Diagonal((wh[0] / max(1e-6, max(xs) - min(xs)),
                                          wh[1] / max(1e-6, max(ys) - min(ys)), 1.0, 1.0)))
        return bm

    # Nissan roundel above the script
    hit, nrm = ray(bvh, (0.0, 4.0, z_rnd), (0, -1, 0))
    m = frame_matrix(hit + nrm * 0.0015, Vector((-1, 0, 0)), nrm) @ Matrix.Scale(k_rnd, 4)
    rnd = bmesh.new()
    revolve([(0.026, 0.0), (0.033, 0.0), (0.033, 0.003), (0.026, 0.003)], 32, closed=True,
            bm=rnd, xform=Matrix.Rotation(math.radians(-90), 4, "Y"))
    add_box(rnd, Vector((0, 0, 0.0018)), (0.072, 0.017, 0.0035), bevel=0.0)
    t = text_mesh("NISSAN", 0.0105, 0.0006)
    t.transform(Matrix.Translation((0, -0.004, 0.0038)))
    add_bm(rnd, t)
    rnd.transform(m)
    add_bm(boot_badges, rnd)
    # "Silvia" script (left of centre seen from behind) and "Spec R"
    hit, nrm = ray(bvh, (script_at[0], 4.0, script_at[1]), (0, -1, 0))
    b = fit_box(text_mesh("Silvia", 0.040, 0.002, shear=0.32), script_wh)
    b.transform(frame_matrix(hit + nrm * 0.001, Vector((-1, 0, 0)), nrm))
    add_bm(boot_badges, b)
    hit, nrm = ray(bvh, (spec_at[0], 4.0, spec_at[1]), (0, -1, 0))
    b = fit_box(text_mesh("Spec", 0.024, 0.002, shear=0.25, align="RIGHT"), spec_wh)
    b.transform(frame_matrix(hit + nrm * 0.001, Vector((-1, 0, 0)), nrm))
    add_bm(boot_badges, b)
    k_r = 1.15 if BODY_MESH else 1.0
    b = text_mesh("R", 0.034 * k_r, 0.0022, shear=0.25, align="LEFT")
    b.transform(frame_matrix(hit + nrm * 0.001 + Vector((-0.006, 0, 0.004)), Vector((-1, 0, 0)), nrm))
    add_bm(red_r, b)
    make("Badge_Boot", boot_badges, ["Badge"], "Trunk", smooth=None)
    make("Badge_SpecR_R", red_r, ["Terminal"], "Trunk", smooth=None)
    return parts


def detail_doors(panels, ref):
    """Door handle pockets and black window frames on the doors."""
    for side, sx in (("L", 1), ("R", -1)):
        door = panels["Door_" + side]
        cut = volume("hpk_" + side, side=fillet([(0.398, 0.744), (0.532, 0.744),
                                                  (0.532, 0.780), (0.398, 0.780)],
                                                 {0: 0.015, 1: 0.015, 2: 0.015, 3: 0.015}),
                     xr=(0.818, 1.3) if sx > 0 else (-1.3, -0.818), mat="BlackMatte")
        boolean(door, cut)
        delete_object(cut)
        # window frame (sash) and the sail ahead of the glass in gloss black,
        # above the belt line. The skin is split along the belt first (10 cm
        # chords of Z_BELT): on the unshaved shoulder under the sail, black
        # by face centre followed the mesh's triangles and showed as a row
        # of yellow teeth at the front corner of the window.
        me = door.data
        if "BlackGloss" not in me.materials:
            me.materials.append(MATS["BlackGloss"])
        bi = list(me.materials).index(MATS["BlackGloss"])
        pi = list(me.materials).index(MATS["Paint"]) if MATS["Paint"].name in me.materials else 0
        by = [-0.80 + 0.10 * k for k in range(16)]
        bz = [Z_BELT(y) + 0.003 for y in by]

        def belt(y):
            k = min(max(int((y - by[0]) / 0.10), 0), len(by) - 2)
            t = (y - by[k]) / (by[k + 1] - by[k])
            return bz[k] + (bz[k + 1] - bz[k]) * t
        chords = [((0.0, by[k], bz[k]), (0.0, bz[k] - bz[k + 1], by[k + 1] - by[k]),
                   lambda c, k=k: by[k] <= c.y < by[k + 1] or
                   (k == 0 and c.y < by[0]) or (k == len(by) - 2 and c.y >= by[-1]))
                  for k in range(len(by) - 1)]
        split_faces(door, chords, lambda c, n: abs(c.z - belt(c.y)) < 0.04)
        me = door.data
        for poly in me.polygons:
            c = poly.center
            if poly.material_index == pi and c.z > belt(c.y):
                poly.material_index = bi


# --------------------------------------------------------------------------
# Interior (RHD) and SR20DET engine bay
# --------------------------------------------------------------------------

def catmull(points, samples=6):
    P = [Vector(p) for p in points]
    if len(P) < 3:
        return P
    out = []
    ext = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for k in range(samples):
            t = k / samples
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    return out


def sweep_tube(bm, points, r, segs=12, mat=0, samples=6, cap=True, smooth=True):
    """Tube of radius r along a (smoothed) polyline, parallel-transport frames."""
    path = catmull(points, samples) if smooth else [Vector(p) for p in points]
    T = []
    for i in range(len(path)):
        a = path[max(i - 1, 0)]
        b = path[min(i + 1, len(path) - 1)]
        T.append((b - a).normalized())
    ref_ = Vector((0, 0, 1)) if abs(T[0].z) < 0.9 else Vector((1, 0, 0))
    N = T[0].cross(ref_).normalized()
    secs = []
    for i, p in enumerate(path):
        if i > 0:
            axis = T[i - 1].cross(T[i])
            if axis.length > 1e-8:
                ang = T[i - 1].angle(T[i])
                N = Matrix.Rotation(ang, 3, axis.normalized()) @ N
        Bn = T[i].cross(N).normalized()
        secs.append([p + (N * math.cos(2 * math.pi * k / segs) + Bn * math.sin(2 * math.pi * k / segs)) * r
                     for k in range(segs)])
    return loft_tube(bm, None, secs, mat=mat, cap=cap)


def rbox(bm, center, size, mat=0, bevel=0.01, rx=0.0, ry=0.0, rz=0.0):
    rot = (Matrix.Rotation(math.radians(rz), 3, "Z") @ Matrix.Rotation(math.radians(ry), 3, "Y")
           @ Matrix.Rotation(math.radians(rx), 3, "X"))
    return add_box(bm, Vector(center), size, mat=mat, rot=rot, bevel=bevel)


DASH_SHIFT = -0.13
STEER_HUB = Vector((-0.37, -0.130 + DASH_SHIFT, 0.770))


def build_interior(body):
    bvh = bvh_of(body)
    parts = {}

    def make(name, bm, mats, assembly="Body", smooth=40):
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
        ob = new_object(name, bm, mats, smooth_angle=smooth)
        ob["assembly"] = assembly
        parts[name] = ob
        return ob

    DRV = -1          # right-hand drive: driver on the car's right (-X)
    # ---- dashboard ------------------------------------------------------
    dash, trim, black = bmesh.new(), bmesh.new(), bmesh.new()
    prof = fillet([(-0.652, 0.40), (-0.652, 0.842), (-0.585, 0.866), (-0.44, 0.858),
                   (-0.335, 0.836), (-0.288, 0.800), (-0.280, 0.735), (-0.300, 0.630),
                   (-0.380, 0.520), (-0.500, 0.470)], {2: 0.03, 4: 0.04, 5: 0.03, 6: 0.04,
                                                       7: 0.05, 8: 0.05})
    prism_from_poly(prof, "x", -0.715, 0.715, bm=dash)
    # binnacle hood over the gauges (driver side)
    xg = 0.37 * DRV
    rbox(dash, (xg, -0.355, 0.858), (0.40, 0.17, 0.058), bevel=0.025)
    # centre stack and console
    rbox(trim, (0.0, -0.34, 0.62), (0.26, 0.15, 0.32), bevel=0.02)
    rbox(trim, (0.0, 0.095, 0.335), (0.24, 1.05, 0.19), bevel=0.04)
    rbox(trim, (0.0, -0.24, 0.42), (0.25, 0.16, 0.16), bevel=0.03, rx=-25)
    # transmission tunnel under the console
    rbox(black, (0.0, -0.11, 0.27), (0.34, 1.34, 0.10), bevel=0.04)
    make("Dashboard", dash, ["Interior"])
    # ---- gauges ------------------------------------------------------
    g, marks, needles = bmesh.new(), bmesh.new(), bmesh.new()
    face = Vector((0, 0.96, 0.28)).normalized()
    for dx, r in ((0.0, 0.052), (0.118, 0.043), (-0.112, 0.036)):
        c = Vector((xg + dx, -0.288, 0.782))
        disc(g, c, face, r, thick=0.004)
        ring(g, c + face * 0.004, face, r, r + 0.006, 0.01)
        n_ticks = 9
        for i in range(n_ticks):
            a = math.radians(-135 + 270 * i / (n_ticks - 1))
            m = frame_matrix(c + face * 0.0015, Vector((1, 0, 0)), face)
            tick = m @ Vector((math.sin(a) * r * 0.82, math.cos(a) * r * 0.82, 0))
            add_box(marks, tick, (0.004, 0.004, 0.002),
                    rot=m.to_3x3() @ Matrix.Rotation(-a, 3, "Z"))
        m = frame_matrix(c + face * 0.003, Vector((1, 0, 0)), face)
        a = math.radians(-60)
        tip = m @ Vector((math.sin(a) * r * 0.75, math.cos(a) * r * 0.75, 0))
        add_cylinder(needles, c + face * 0.003, tip, 0.0018, 6)
    # boost gauge pod on the dash top centre
    c = Vector((0.0, -0.40, 0.862))
    pf = Vector((0, 0.8, 0.6)).normalized()
    add_cylinder(black, c - pf * 0.03, c, 0.034, 20)
    disc(g, c + pf * 0.001, pf, 0.028, thick=0.003)
    make("Gauges", g, ["Gauge"])
    make("CentreConsole", trim, ["InteriorTrim"])
    make("Tunnel", black, ["BlackMatte"])
    make("GaugeMarks", marks, ["GaugeMark"], smooth=None)
    make("Light_Dash", needles, ["Needle"])
    # head unit, climate dials, vents
    hu = bmesh.new()
    rbox(hu, (0.0, -0.262, 0.640), (0.19, 0.012, 0.055), bevel=0.003)
    for dx in (-0.065, 0.0, 0.065):
        disc(hu, Vector((dx, -0.262, 0.55)), (0, 1, 0), 0.020, dome=0.004, thick=0.01)
    for dx in (-0.06, 0.06):
        rbox(hu, (dx, -0.270, 0.735), (0.09, 0.012, 0.045), bevel=0.004)
        for k in range(4):
            rbox(hu, (dx, -0.263, 0.720 + k * 0.01), (0.085, 0.004, 0.003), bevel=0.0)
    for sx in (1, -1):
        c = Vector((0.655 * sx, -0.282, 0.765))
        ring(hu, c, (0, 1, 0), 0.026, 0.034, 0.012)
        disc(hu, c - Vector((0, 0.006, 0)), (0, 1, 0), 0.026, thick=0.004)
    make("DashTrim", hu, ["BlackGloss"])
    # ---- steering column & wheel (RHD) ---------------------------------
    hub = Vector((xg, -0.130, 0.770))
    sw_axis = Vector((0, 0.906, 0.423)).normalized()
    col = bmesh.new()
    add_cylinder(col, Vector((xg, -0.40, 0.66)), hub - sw_axis * 0.04, 0.045, 16)
    rbox(col, (xg, -0.27, 0.70), (0.16, 0.14, 0.11), bevel=0.03, rx=-25)
    make("SteeringColumn", col, ["InteriorTrim"])
    sw = bmesh.new()
    q = Vector((1, 0, 0)).rotation_difference(sw_axis)
    xf = Matrix.Translation(hub) @ q.to_matrix().to_4x4()
    rim = [(0.185 + 0.016 * math.cos(2 * math.pi * k / 10), 0.016 * math.sin(2 * math.pi * k / 10))
           for k in range(10)]
    rim = [(r, a) for r, a in rim]
    revolve(rim, 48, closed=True, bm=sw, xform=xf)
    hubp = [(0.0, 0.035), (0.050, 0.033), (0.065, 0.020), (0.068, 0.0), (0.0, 0.0)]
    revolve(hubp, 28, closed=True, bm=sw, xform=xf)
    m = frame_matrix(hub, Vector((1, 0, 0)), sw_axis)
    for ang in (90, 210, 330):
        a = math.radians(ang)
        d = Vector((math.cos(a), math.sin(a), 0))
        p0 = m @ (d * 0.05 + Vector((0, 0, 0.012)))
        p1 = m @ (d * 0.178)
        add_box(sw, (p0 + p1) / 2, ((p1 - p0).length, 0.030 if ang == 90 else 0.045, 0.014),
                rot=(p1 - p0).to_track_quat("X", "Z").to_matrix(), bevel=0.004)
    disc(sw, hub + sw_axis * 0.037, sw_axis, 0.018, thick=0.002, mat=0)
    ob = make("SteeringWheel", sw, ["Leather"])
    ob["hub"] = list(hub)
    ob["axis"] = list(sw_axis)
    # ---- pedals ------------------------------------------------------
    pd = bmesh.new()
    for dx in (-0.10, 0.0, 0.10):
        x = xg + dx
        rbox(pd, (x, -0.565, 0.33), (0.055 if dx else 0.07, 0.012, 0.075), bevel=0.004, rx=-30)
        add_cylinder(pd, Vector((x, -0.57, 0.36)), Vector((x, -0.62, 0.58)), 0.008, 8)
    make("Pedals", pd, ["Chrome"])
    # ---- shifter & handbrake ----------------------------------------
    sh, knob = bmesh.new(), bmesh.new()
    base = Vector((0.0, 0.02, 0.43))
    revolve([(0.0, 0.0), (0.065, 0.0), (0.05, 0.03), (0.025, 0.065), (0.0, 0.07)], 20, closed=True,
            bm=sh, xform=Matrix.Translation(base) @ Matrix.Rotation(math.radians(-90), 4, "Y"))
    add_cylinder(sh, base, base + Vector((0, -0.02, 0.15)), 0.008, 10)
    revolve([(0.0, -0.028), (0.020, -0.022), (0.028, 0.0), (0.020, 0.022), (0.0, 0.028)], 18,
            closed=True, bm=knob, xform=Matrix.Translation(base + Vector((0, -0.022, 0.175))))
    hb0 = Vector((0.075, 0.24, 0.43))
    hb1 = Vector((0.075, 0.47, 0.47))
    rbox(sh, (hb0 + hb1) / 2, ((hb1 - hb0).length, 0.03, 0.035), bevel=0.01,
         rx=0, rz=90, ry=-8)
    make("Shifter", sh, ["Leather"])
    make("ShiftKnob", knob, ["Chrome"])
    # ---- seats ---------------------------------------------------------
    st, ins, rails = bmesh.new(), bmesh.new(), bmesh.new()
    for sx in (1, -1):
        x = 0.37 * sx
        rbox(st, (x, 0.30, 0.385), (0.48, 0.50, 0.11), bevel=0.035)
        for bx in (-0.205, 0.205):
            rbox(st, (x + bx, 0.30, 0.43), (0.075, 0.46, 0.13), bevel=0.03)
        rbox(ins, (x, 0.30, 0.443), (0.30, 0.40, 0.012), bevel=0.004)
        back_c = Vector((x, 0.615, 0.735))
        rbox(st, back_c, (0.46, 0.11, 0.62), bevel=0.04, rx=-17)
        for bx in (-0.215, 0.215):
            rbox(st, back_c + Vector((bx, -0.05, -0.04)), (0.08, 0.14, 0.50), bevel=0.03, rx=-17)
        rbox(ins, back_c + Vector((0, -0.057, -0.02)), (0.28, 0.012, 0.48), bevel=0.004, rx=-17)
        rbox(st, (x, 0.725, 1.115), (0.25, 0.09, 0.17), bevel=0.03, rx=-10)
        for px in (-0.07, 0.07):
            add_cylinder(rails, Vector((x + px, 0.71, 1.03)), Vector((x + px, 0.69, 1.06)), 0.006, 8)
        for rx_ in (-0.17, 0.17):
            rbox(rails, (x + rx_, 0.30, 0.265), (0.03, 0.52, 0.05), bevel=0.004)
    # rear bench (2+2 coupe)
    rbox(st, (0.0, 0.81, 0.36), (0.84, 0.30, 0.13), bevel=0.04)
    rbox(st, (0.0, 0.915, 0.63), (0.84, 0.10, 0.44), bevel=0.04, rx=-22)
    for sx in (1, -1):
        rbox(ins, (0.24 * sx, 0.79, 0.428), (0.30, 0.22, 0.01), bevel=0.003)
    make("Seats", st, ["Seat"])
    make("SeatInserts", ins, ["SeatAccent"])
    make("SeatRails", rails, ["BlackMatte"])
    # ---- floor mats, parcel shelf speakers --------------------------
    fm = bmesh.new()
    for sx in (1, -1):
        rbox(fm, (0.37 * sx, -0.30, 0.245), (0.42, 0.50, 0.008), bevel=0.003)
        rbox(fm, (0.30 * sx, 0.60, 0.245), (0.30, 0.10, 0.008), bevel=0.003)
    make("FloorMats", fm, ["Carpet"])
    sp = bmesh.new()
    for sx in (1, -1):
        disc(sp, Vector((0.38 * sx, 1.25, 0.862)), (0, 0, 1), 0.06, dome=0.008, thick=0.004)
    make("ShelfSpeakers", sp, ["BlackMatte"])
    # ---- mirror, visors (placed against the real roof lining) ----------
    hit, _ = ray(bvh, (0.0, -0.02, 0.8), (0, 0, 1))
    zt = hit.z if hit else 1.21
    rm = bmesh.new()
    add_cylinder(rm, Vector((0.0, -0.035, zt)), Vector((0.0, -0.07, zt - 0.045)), 0.008, 8)
    rbox(rm, (0.0, -0.07, zt - 0.06), (0.24, 0.028, 0.068), bevel=0.012)
    make("RearViewMirror", rm, ["BlackMatte"])
    rmg = bmesh.new()
    rbox(rmg, (0.0, -0.055, zt - 0.06), (0.22, 0.002, 0.056), bevel=0.0)
    make("RearViewMirrorGlass", rmg, ["Mirror"])
    vs = bmesh.new()
    for sx in (1, -1):
        hit, _ = ray(bvh, (0.33 * sx, 0.07, 0.8), (0, 0, 1))
        z = (hit.z if hit else 1.21) - 0.016
        rbox(vs, (0.33 * sx, 0.07, z), (0.32, 0.16, 0.022), bevel=0.008)
    make("SunVisors", vs, ["Headliner"])
    # ---- door cards (ride with the doors) -------------------------------
    for side, sx in (("L", 1), ("R", -1)):
        dc, dcb = bmesh.new(), bmesh.new()
        hit, _ = ray(bvh, (0.2 * sx, 0.10, 0.57), (sx, 0, 0))
        xi = (hit.x if hit else 0.80 * sx) - 0.022 * sx
        rbox(dc, (xi, 0.08, 0.565), (0.05, 0.50, 0.045), bevel=0.012)
        rbox(dc, (xi + 0.005 * sx, -0.05, 0.70), (0.035, 0.30, 0.02), bevel=0.008)
        rbox(dcb, (xi - 0.012 * sx, 0.20, 0.592), (0.03, 0.12, 0.012), bevel=0.003)
        hit, _ = ray(bvh, (0.2 * sx, -0.38, 0.42), (sx, 0, 0))
        xs_ = (hit.x if hit else 0.79 * sx) - 0.006 * sx
        disc(dcb, Vector((xs_, -0.38, 0.42)), (-sx, 0, 0), 0.065, dome=0.008, thick=0.006)
        make("DoorCard_" + side, dc, ["InteriorTrim"], "Door_" + side)
        make("DoorSwitches_" + side, dcb, ["BlackMatte"], "Door_" + side)
    # the windscreen base moved 0.13 m forward relative to the seats
    for n in ("Dashboard", "CentreConsole", "Gauges", "GaugeMarks", "Light_Dash", "DashTrim",
              "SteeringColumn", "SteeringWheel", "Pedals"):
        ob = parts[n]
        ob.data.transform(Matrix.Translation((0.0, DASH_SHIFT, 0.0)))
    if BODY_MESH:     # the dash top's front corners stay inboard of the A-pillar
        def dx(y):    # feet (the reconstruction's pillars are not shaved)
            return screen_edge_x(y) - 0.045
        clip = volume("cut_dash", plan=[(-0.75, -0.25), (-0.75, -0.62), (-dx(-0.66), -0.66),
                                        (-dx(-0.86), -0.86), (dx(-0.86), -0.86),
                                        (dx(-0.66), -0.66), (0.75, -0.62), (0.75, -0.25)],
                      zr=(0.0, 2.0), mat="Interior")
        boolean(parts["Dashboard"], clip, "INTERSECT")
        delete_object(clip)
    # ---- boot: spare wheel ----------------------------------------------
    sp = bmesh.new()
    c = Vector((0.0, 1.78, 0.39))
    revolve([(0.20, 0.06), (0.26, 0.06), (0.27, 0.0), (0.26, -0.06), (0.20, -0.06)], 32,
            closed=True, bm=sp, xform=Matrix.Translation(c) @ Matrix.Rotation(math.radians(-90), 4, "Y"))
    make("SpareTyre", sp, ["Rubber"])
    return parts


def build_engine_bay():
    parts = {}

    def make(name, bm, mats, assembly="Body", smooth=40):
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
        ob = new_object(name, bm, mats, smooth_angle=smooth)
        ob["assembly"] = assembly
        parts[name] = ob
        return ob

    y0, y1 = -1.62, -1.04            # cylinder head front / rear
    blk, vc, coil, alu, hose, blackp, chrome, cpl = (bmesh.new() for _ in range(8))
    # block, sump, head, bellhousing, timing cover
    rbox(blk, (0.0, (y0 + y1) / 2 + 0.01, 0.39), (0.33, 0.60, 0.26), bevel=0.02)
    rbox(blk, (0.0, (y0 + y1) / 2 + 0.03, 0.245), (0.27, 0.50, 0.07), bevel=0.02)
    rbox(blk, (0.0, (y0 + y1) / 2, 0.575), (0.34, 0.60, 0.11), bevel=0.015)
    revolve([(0.0, 0.0), (0.20, 0.0), (0.21, 0.05), (0.13, 0.30), (0.0, 0.30)], 24, closed=True,
            bm=blk, xform=Matrix.Translation((0, y1 + 0.02, 0.38)) @
            Matrix.Rotation(math.radians(90), 4, "Z"))
    rbox(blackp, (0.0, y0 - 0.025, 0.47), (0.30, 0.04, 0.30), bevel=0.02)
    # pulleys + belt
    for (px, pz, pr) in ((0.0, 0.31, 0.075), (-0.12, 0.47, 0.045), (0.10, 0.52, 0.05),
                         (0.0, 0.60, 0.055)):
        c = Vector((px, y0 - 0.055, pz))
        disc(chrome, c, (0, -1, 0), pr, thick=0.022)
    add_cylinder(blk, Vector((-0.17, y0 + 0.05, 0.46)), Vector((-0.17, y0 + 0.20, 0.46)), 0.055, 18)
    # red valve cover: twin cam humps + centre coil-pack plate
    secs = []
    for t in (0.0, 0.03, 0.97, 1.0):
        yy = lerp(y0 + 0.02, y1 - 0.02, t)
        k = 0.92 if t in (0.0, 1.0) else 1.0
        sec = []
        for ang in range(0, 181, 15):
            a = math.radians(ang)
            sec.append(Vector((-0.155 * k * math.cos(a), yy, 0.630 + 0.075 * k * math.sin(a) ** 0.6)))
        sec += [Vector((0.155 * k, yy, 0.625)), Vector((-0.155 * k, yy, 0.625))]
        secs.append(sec[:-1])
    loft_tube(vc, None, secs, mat=0)
    bmesh.ops.recalc_face_normals(vc, faces=vc.faces)
    for i in range(6):
        yy = lerp(y0 + 0.07, y1 - 0.07, i / 5)
        for sx in (1, -1):
            rbox(vc, (0.11 * sx, yy, 0.69), (0.012, 0.05, 0.03), bevel=0.003)
    rbox(coil, (0.0, (y0 + y1) / 2, 0.712), (0.095, 0.50, 0.03), bevel=0.01)
    for i in range(4):
        yy = lerp(y0 + 0.11, y1 - 0.11, i / 3)
        add_cylinder(coil, Vector((0.0, yy, 0.725)), Vector((0.0, yy, 0.735)), 0.022, 12)
    t = text_mesh("TWIN CAM TURBO", 0.018, 0.001)
    t.transform(frame_matrix(Vector((0.0, (y0 + y1) / 2 + 0.17, 0.728)), Vector((0, -1, 0)),
                             Vector((0, 0, 1))) @ Matrix.Rotation(math.radians(90), 4, "Z"))
    add_bm(chrome, t)
    # intake plenum (+X) with runners and throttle body facing forward
    rbox(alu, (0.255, (y0 + y1) / 2 - 0.02, 0.625), (0.12, 0.56, 0.11), bevel=0.04)
    for i in range(4):
        yy = lerp(y0 + 0.08, y1 - 0.08, i / 3)
        sweep_tube(alu, [(0.24, yy, 0.60), (0.215, yy, 0.55), (0.17, yy, 0.55)], 0.022, 10)
    add_cylinder(alu, Vector((0.255, y0 - 0.02, 0.625)), Vector((0.255, y0 - 0.09, 0.625)), 0.038, 18)
    # exhaust side (-X): manifold heat shield, T28 turbo, downpipe
    rbox(chrome, (-0.215, (y0 + y1) / 2 - 0.02, 0.50), (0.07, 0.44, 0.12), bevel=0.02, ry=-20)
    tc = Vector((-0.30, -1.43, 0.58))
    revolve([(0.0, 0.0), (0.06, 0.0), (0.085, 0.02), (0.09, 0.045), (0.07, 0.07), (0.0, 0.07)], 24,
            closed=True, bm=alu, xform=Matrix.Translation(tc) @ Matrix.Rotation(math.radians(90), 4, "Z"))
    add_cylinder(alu, tc - Vector((0, 0.02, 0)), tc - Vector((0, 0.07, 0)), 0.042, 18)
    tb = bmesh.new()
    revolve([(0.0, 0.0), (0.075, 0.0), (0.08, 0.05), (0.06, 0.08), (0.0, 0.08)], 20, closed=True,
            bm=tb, xform=Matrix.Translation(tc + Vector((0, 0.07, -0.02))) @
            Matrix.Rotation(math.radians(90), 4, "Z"))
    sweep_tube(tb, [tc + Vector((-0.02, 0.14, -0.05)), (-0.33, -1.20, 0.38), (-0.30, -1.08, 0.22),
                    (-0.22, -0.95, 0.16)], 0.032, 12)
    # front-mount intercooler + piping (visible through the bumper)
    ic = bmesh.new()
    rbox(ic, (0.0, -2.035, 0.335), (0.66, 0.055, 0.16), bevel=0.004)
    for sx in (1, -1):
        rbox(alu, (0.355 * sx, -2.035, 0.335), (0.05, 0.07, 0.19), bevel=0.012)
    for k in range(12):
        rbox(ic, (0.0, -2.064, 0.262 + k * 0.0135), (0.64, 0.004, 0.003), bevel=0.0)
    pipe_r = [(-0.355, -2.0, 0.40), (-0.40, -1.88, 0.47), (-0.38, -1.70, 0.60),
              (-0.32, -1.52, 0.66), (-0.30, -1.47, 0.62)]
    pipe_l = [(0.355, -2.0, 0.40), (0.40, -1.88, 0.47), (0.36, -1.76, 0.60),
              (0.29, -1.70, 0.625), (0.255, -1.70, 0.625)]
    sweep_tube(alu, pipe_r, 0.030, 14)
    sweep_tube(alu, pipe_l, 0.030, 14)
    for pth in (pipe_r, pipe_l):
        a, b = Vector(pth[1]), Vector(pth[2])
        add_cylinder(cpl, a.lerp(b, 0.35), a.lerp(b, 0.65), 0.035, 14)
    # radiator, fan shroud, hoses
    rad = bmesh.new()
    rbox(rad, (0.0, -1.955, 0.465), (0.64, 0.04, 0.30), bevel=0.004)
    rbox(blackp, (0.0, -1.955, 0.635), (0.66, 0.06, 0.04), bevel=0.01)
    rbox(blackp, (0.0, -1.955, 0.300), (0.66, 0.06, 0.035), bevel=0.01)
    ring(blackp, Vector((0.0, -1.905, 0.465)), (0, 1, 0), 0.13, 0.15, 0.05)
    for k in range(7):
        a = 2 * math.pi * k / 7
        c = Vector((0.0, -1.89, 0.465))
        tip = c + Vector((math.cos(a) * 0.125, 0, math.sin(a) * 0.125))
        add_box(blackp, (c + tip) / 2, (0.125, 0.004, 0.035),
                rot=Matrix.Rotation(-a, 3, "Y") @ Matrix.Rotation(math.radians(25), 3, "X"))
    disc(blackp, Vector((0.0, -1.885, 0.465)), (0, 1, 0), 0.04, thick=0.03)
    sweep_tube(hose, [(0.25, -1.94, 0.62), (0.20, -1.80, 0.66), (0.10, -1.68, 0.65),
                      (0.05, y0 - 0.01, 0.63)], 0.022, 12)
    sweep_tube(hose, [(-0.25, -1.94, 0.33), (-0.18, -1.80, 0.32), (-0.06, y0 - 0.04, 0.38)],
               0.022, 12)
    # air box + intake to the turbo inlet
    rbox(blackp, (0.48, -1.82, 0.52), (0.18, 0.26, 0.16), bevel=0.02)
    sweep_tube(blackp, [(0.40, -1.75, 0.60), (0.20, -1.72, 0.68), (-0.10, -1.66, 0.68),
                        (-0.28, -1.55, 0.61), (-0.30, -1.50, 0.58)], 0.038, 14)
    # battery (front right), fuse box, reservoirs
    bat, term = bmesh.new(), bmesh.new()
    rbox(bat, (-0.50, -1.83, 0.47), (0.14, 0.25, 0.19), bevel=0.008)
    for dy in (-0.08, 0.08):
        add_cylinder(term, Vector((-0.50, -1.83 + dy, 0.565)), Vector((-0.50, -1.83 + dy, 0.585)), 0.012, 10)
    rbox(blackp, (0.52, -1.10, 0.56), (0.14, 0.18, 0.10), bevel=0.012)
    tr = bmesh.new()
    rbox(tr, (0.54, -1.55, 0.58), (0.10, 0.16, 0.13), bevel=0.02)
    rbox(tr, (-0.06, -1.04, 0.70), (0.08, 0.06, 0.05), bevel=0.01)
    add_cylinder(cpl, Vector((0.54, -1.55, 0.645)), Vector((0.54, -1.55, 0.665)), 0.022, 14)
    add_cylinder(blackp, Vector((-0.24, -1.70, 0.62)), Vector((-0.24, -1.70, 0.70)), 0.035, 14)
    # brake booster + master cylinder (driver side, RHD)
    bb = Vector((-0.40, -0.94, 0.60))
    add_cylinder(blackp, bb, bb - Vector((0, 0.10, 0)), 0.11, 28)
    add_cylinder(alu, bb - Vector((0, 0.10, 0)), bb - Vector((0, 0.23, 0)), 0.030, 14)
    rbox(tr, (-0.40, -1.12, 0.665), (0.07, 0.09, 0.05), bevel=0.01)
    # strut tops + strut tower bar
    sb = bmesh.new()
    for sx in (1, -1):
        c = Vector((0.60 * sx, -1.30, 0.692))
        add_cylinder(sb, c, c + Vector((0, 0, 0.018)), 0.07, 20)
        add_cylinder(chrome, c + Vector((0, 0, 0.018)), c + Vector((0, 0, 0.04)), 0.014, 6)
        rbox(sb, (0.55 * sx, -1.30, 0.735), (0.12, 0.11, 0.012), bevel=0.003)
    sweep_tube(sb, [(-0.50, -1.30, 0.742), (-0.20, -1.30, 0.748), (0.20, -1.30, 0.748),
                    (0.50, -1.30, 0.742)], 0.019, 16)
    # wiring loom along the firewall
    sweep_tube(hose, [(-0.55, -0.93, 0.70), (-0.2, -0.93, 0.72), (0.2, -0.93, 0.72),
                      (0.55, -0.93, 0.66)], 0.012, 8)
    make("Engine", blk, ["EngineBlock"])
    make("ValveCover", vc, ["ValveCover"])
    make("CoilPacks", coil, ["CoilPack"])
    make("EngineAlloy", alu, ["Aluminum"])
    make("EngineHoses", hose, ["Hose"])
    make("EnginePlastics", blackp, ["BlackMatte"])
    make("EngineChrome", chrome, ["Chrome"])
    make("Couplers", cpl, ["Coupler"])
    make("Turbo", tb, ["Turbine"])
    make("Intercooler", ic, ["Radiator"])
    make("Radiator", rad, ["Radiator"])
    make("Battery", bat, ["Battery"])
    make("BatteryTerminals", term, ["Terminal"])
    make("Reservoirs", tr, ["Translucent"])
    make("StrutBar", sb, ["Aluminum"])
    return parts


def build_underbody():
    parts = {}

    def make(name, bm, mats, assembly="Body", smooth=40):
        ob = new_object(name, bm, mats, smooth_angle=smooth)
        ob["assembly"] = assembly
        parts[name] = ob
        return ob

    ub, ex = bmesh.new(), bmesh.new()
    # exhaust: downpipe -> centre pipe -> muffler -> tip
    sweep_tube(ex, [(-0.22, -0.95, 0.16), (-0.05, -0.60, 0.14), (0.05, 0.20, 0.14),
                    (0.15, 1.10, 0.145), (0.30, 1.55, 0.18)], 0.030, 12)
    revolve([(0.0, 0.0), (0.085, 0.0), (0.09, 0.02), (0.09, 0.30), (0.085, 0.32), (0.0, 0.32)],
            24, closed=True, bm=ex, xform=Matrix.Translation((0.0, -0.05, 0.14)) @
            Matrix.Rotation(math.radians(90), 4, "Z") @ Matrix.Diagonal((1, 1, 0.6, 1)))
    revolve([(0.0, 0.0), (0.12, 0.0), (0.125, 0.02), (0.125, 0.36), (0.12, 0.38), (0.0, 0.38)],
            28, closed=True, bm=ex, xform=Matrix.Translation((0.32, 1.60, 0.205)) @
            Matrix.Rotation(math.radians(90), 4, "Z") @ Matrix.Diagonal((1, 1, 0.62, 1)))
    sweep_tube(ex, [(0.40, 1.95, 0.23), (0.43, 2.0, 0.245), (0.44, 2.06, 0.248)], 0.032, 12)
    # front: crossmember, lower arms, tie rods; rear: subframe, diff, arms
    for yy, w in ((Y_FAX + 0.05, 0.86), (Y_RAX - 0.02, 0.84)):
        rbox(ub, (0.0, yy, 0.20), (w, 0.10, 0.07), bevel=0.01)
    for sx in (1, -1):
        add_cylinder(ub, Vector((0.30 * sx, Y_FAX + 0.05, 0.19)), Vector((0.66 * sx, Y_FAX, WHEEL_Z - 0.10)), 0.018, 10)
        add_cylinder(ub, Vector((0.25 * sx, Y_FAX - 0.25, 0.20)), Vector((0.62 * sx, Y_FAX - 0.02, WHEEL_Z - 0.09)), 0.014, 8)
        add_cylinder(ub, Vector((0.20 * sx, Y_FAX + 0.12, 0.25)), Vector((0.62 * sx, Y_FAX + 0.12, WHEEL_Z - 0.02)), 0.010, 8)
        add_cylinder(ub, Vector((0.28 * sx, Y_RAX, 0.22)), Vector((0.64 * sx, Y_RAX, WHEEL_Z - 0.08)), 0.018, 10)
        add_cylinder(ub, Vector((0.25 * sx, Y_RAX + 0.18, 0.24)), Vector((0.62 * sx, Y_RAX + 0.05, WHEEL_Z)), 0.012, 8)
        # half shafts
        add_cylinder(ub, Vector((0.12 * sx, Y_RAX, 0.30)), Vector((0.60 * sx, Y_RAX, WHEEL_Z)), 0.016, 10)
        # coil-over struts visible in the arches
        add_cylinder(ub, Vector((0.60 * sx, Y_FAX - 0.03, WHEEL_Z)), Vector((0.60 * sx, Y_FAX - 0.03, 0.70)), 0.028, 12)
        add_cylinder(ub, Vector((0.60 * sx, Y_RAX + 0.03, WHEEL_Z)), Vector((0.60 * sx, Y_RAX + 0.03, 0.68)), 0.026, 12)
    # R200 diff + driveshaft + transmission + fuel tank
    revolve([(0.0, 0.0), (0.10, 0.0), (0.12, 0.06), (0.11, 0.16), (0.0, 0.18)], 20, closed=True,
            bm=ub, xform=Matrix.Translation((0.0, Y_RAX - 0.08, 0.30)) @
            Matrix.Rotation(math.radians(90), 4, "Z"))
    add_cylinder(ub, Vector((0.0, -0.75, 0.30)), Vector((0.0, -0.20, 0.17)), 0.09, 16, r1=0.06)
    add_cylinder(ub, Vector((0.0, -0.20, 0.17)), Vector((0.0, Y_RAX - 0.08, 0.30)), 0.035, 12)
    rbox(ub, (0.0, 1.05, 0.22), (0.62, 0.36, 0.10), bevel=0.03)
    make("Exhaust", ex, ["Exhaust"])
    make("Chassis", ub, ["Underbody"])
    return parts


def tag_headliner(body):
    """Roof lining in light grey, carpeted floors in the cabin and boot."""
    me = body.data
    for name in ("Headliner", "Carpet"):
        if MATS[name].name not in me.materials:
            me.materials.append(MATS[name])
    names = [m.name for m in me.materials]
    if "Interior" not in names:
        return
    ii = names.index("Interior")
    hi, ci = names.index("Headliner"), names.index("Carpet")
    for p in me.polygons:
        if p.material_index != ii:
            continue
        n, c = p.normal, p.center
        if n.z < -0.6 and c.z > 1.0:
            p.material_index = hi
        elif n.z > 0.8 and c.z < 0.40:
            p.material_index = ci


# hinge pivots and opening rotations (axis, degrees) for the opening panels
HINGES = {
    "Door_L": (Vector((0.80, -0.700, 0.55)), "Z", -62.0),
    "Door_R": (Vector((-0.80, -0.700, 0.55)), "Z", 62.0),
    "Hood": (Vector((0.0, -0.815, 0.880)), "X", -52.0),
    "Trunk": (Vector((0.0, 1.532, 0.962)), "X", 58.0),
}


def set_origin(ob, point):
    """Move the object origin to `point` (world) without moving the mesh."""
    off = Vector(point) - ob.location
    ob.data.transform(Matrix.Translation(-off))
    ob.location = Vector(point)


def open_panels(parts, amount=1.0):
    """Swing doors/hood/boot open (preview / debugging)."""
    for name, (piv, axis, deg) in HINGES.items():
        for ob in parts.values():
            if ob.get("assembly") == name:
                ob.rotation_mode = "XYZ"
                r = math.radians(deg * amount)
                rot = Matrix.Rotation(r, 4, axis)
                M = Matrix.Translation(piv) @ rot @ Matrix.Translation(-piv)
                ob.matrix_world = M @ ob.matrix_world


# --------------------------------------------------------------------------
# Roblox export
# --------------------------------------------------------------------------

def _rgb(c, gamma=True):
    """Linear (0-1) colour -> sRGB 0-255 tuple for Roblox Color3."""
    out = []
    for v in c:
        v = max(0.0, min(1.0, v))
        if gamma:
            v = 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
        out.append(int(round(v * 255)))
    return tuple(out)


# Blender material -> Roblox MeshPart look: (Material, colour, Transparency, Reflectance)
def roblox_look(mat_name):
    m = MATS[mat_name]
    col = _rgb(tuple(m.diffuse_color)[:3])
    table = {
        "Paint": ("SmoothPlastic", None, 0.0, 0.08),
        "BlackGloss": ("SmoothPlastic", None, 0.0, 0.08),
        "BlackMatte": ("Plastic", None, 0.0, 0.0),
        "Rubber": ("SmoothPlastic", (28, 28, 28), 0.0, 0.0),
        "Glass": ("Glass", (70, 82, 84), 0.55, 0.15),
        "Chrome": ("SmoothPlastic", (200, 200, 205), 0.0, 0.45),
        "Reflector": ("SmoothPlastic", (200, 200, 205), 0.0, 0.5),
        "LensClear": ("Glass", (235, 240, 245), 0.6, 0.1),
        "LensRed": ("Glass", (110, 8, 12), 0.35, 0.05),
        "BulbRed": ("SmoothPlastic", (200, 10, 10), 0.0, 0.1),
        "LensAmber": ("Glass", (255, 140, 0), 0.2, 0.05),
        "LensSmoke": ("Glass", (40, 40, 44), 0.4, 0.05),
        "TailAmber": ("Glass", (150, 70, 10), 0.35, 0.05),
        "AmberReflector": ("SmoothPlastic", (255, 150, 20), 0.0, 0.1),
        "Housing": ("SmoothPlastic", (18, 18, 20), 0.0, 0.0),
        "Interior": ("Fabric", (34, 34, 37), 0.0, 0.0),
        "InteriorTrim": ("SmoothPlastic", (45, 45, 48), 0.0, 0.0),
        "Seat": ("Fabric", (25, 25, 27), 0.0, 0.0),
        "SeatAccent": ("Fabric", (70, 70, 74), 0.0, 0.0),
        "Rim": ("Metal", (190, 192, 196), 0.0, 0.15),
        "Brake": ("Metal", (130, 130, 132), 0.0, 0.05),
        "Caliper": ("SmoothPlastic", (70, 70, 76), 0.0, 0.05),
        "Underbody": ("Plastic", (22, 22, 24), 0.0, 0.0),
        "Exhaust": ("Metal", (170, 170, 175), 0.0, 0.2),
        "PlateWhite": ("SmoothPlastic", (238, 240, 236), 0.0, 0.0),
        "PlateText": ("SmoothPlastic", (10, 90, 40), 0.0, 0.0),
        "Badge": ("SmoothPlastic", (215, 215, 220), 0.0, 0.5),
        "Mirror": ("SmoothPlastic", (210, 215, 220), 0.0, 0.85),
        "Emitter": ("SmoothPlastic", (240, 240, 230), 0.0, 0.1),
        "Headliner": ("Fabric", (150, 150, 146), 0.0, 0.0),
        "Carpet": ("Fabric", (20, 20, 22), 0.0, 0.0),
        "Gauge": ("SmoothPlastic", (8, 8, 10), 0.0, 0.1),
        "GaugeMark": ("SmoothPlastic", (235, 235, 235), 0.0, 0.0),
        "Needle": ("Neon", (255, 60, 10), 0.0, 0.0),
        "Leather": ("SmoothPlastic", (30, 30, 32), 0.0, 0.0),
        "EngineBlock": ("Metal", (150, 152, 156), 0.0, 0.05),
        "ValveCover": ("SmoothPlastic", (170, 20, 20), 0.0, 0.05),
        "Aluminum": ("Metal", (210, 212, 216), 0.0, 0.25),
        "Hose": ("Plastic", (20, 20, 22), 0.0, 0.0),
        "Coupler": ("SmoothPlastic", (30, 70, 170), 0.0, 0.0),
        "Translucent": ("Glass", (235, 235, 225), 0.3, 0.0),
        "Radiator": ("Metal", (45, 45, 48), 0.0, 0.0),
        "Battery": ("SmoothPlastic", (25, 25, 30), 0.0, 0.0),
        "Terminal": ("SmoothPlastic", (200, 20, 20), 0.0, 0.0),
        "CoilPack": ("SmoothPlastic", (25, 25, 28), 0.0, 0.0),
        "Turbine": ("Metal", (90, 80, 72), 0.0, 0.0),
    }
    mat, c, tr, rf = table.get(mat_name, ("SmoothPlastic", None, 0.0, 0.0))
    return {"Material": mat, "Color": c or col, "Transparency": tr, "Reflectance": rf}


def _select_only(obs, active):
    for o in bpy.context.scene.objects:
        if o is not None and o.select_get():
            o.select_set(False)
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active


def bake_normals(ob):
    """Freeze the current smooth/sharp shading into custom normals so it
    survives separating/joining (seams stay invisible in Roblox)."""
    me = ob.data
    if len(me.polygons) == 0:
        return
    if hasattr(me, "corner_normals"):
        nrm = [tuple(cn.vector) for cn in me.corner_normals]
    else:
        me.calc_normals_split()
        nrm = [tuple(lp.normal) for lp in me.loops]
    me.normals_split_custom_set(nrm)


def separate_materials(ob):
    used = {p.material_index for p in ob.data.polygons}
    if len(used) <= 1:
        return [ob]
    _select_only([ob], ob)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="MATERIAL")
    bpy.ops.object.mode_set(mode="OBJECT")
    return [o for o in bpy.context.selected_objects]


def main_material(ob):
    me = ob.data
    if not me.polygons:
        return None
    idx = me.polygons[0].material_index
    return me.materials[idx].name if idx < len(me.materials) else None


def split_to_limit(ob, limit=TRI_LIMIT - 500):
    """Recursively halve a mesh (along its longest axis) until every piece
    is under the Roblox triangle cap."""
    if tri_count(ob) <= limit:
        return [ob]
    me = ob.data
    dims = ob.dimensions
    axis = max(range(3), key=lambda i: dims[i])
    cs = sorted(p.center[axis] for p in me.polygons)
    med = cs[len(cs) // 2]
    for p in me.polygons:
        p.select = p.center[axis] > med
    _select_only([ob], ob)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.object.mode_set(mode="OBJECT")
    for p in me.polygons:
        p.select = p.center[axis] > med
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    pieces = [o for o in bpy.context.selected_objects]
    out = []
    for o in pieces:
        out += split_to_limit(o, limit)
    return out


def finalize_parts(parts):
    """Group everything into Roblox-ready MeshParts: one material per part,
    grouped by assembly (Body, Door_L, Hood, Wheel_FL ...), under the
    triangle cap. Light parts keep their own names for the light script."""
    from collections import defaultdict
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH" and "assembly" in o]
    for o in [o for o in objs if not o.data.polygons]:
        print(f"[warn] {o.name} came out empty, skipped")
        objs.remove(o)
        delete_object(o)
    for o in objs:
        _select_only([o], o)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bake_normals(o)
    groups = defaultdict(list)
    lights = []
    for o in objs:
        asm = o["assembly"]
        if o.name.startswith("Light_"):
            lights.append(o)
            continue
        for piece in separate_materials(o):
            piece["assembly"] = asm
            groups[(asm, main_material(piece))].append(piece)
    final = []
    for (asm, mat), obs in sorted(groups.items()):
        target = obs[0]
        if len(obs) > 1:
            _select_only(obs, target)
            bpy.ops.object.join()
        target.name = f"{asm}_{mat}"
        target.data.name = target.name
        target["assembly"] = asm
        target["material"] = mat
        pieces = split_to_limit(target)
        for i, pc in enumerate(pieces):
            if len(pieces) > 1:
                pc.name = f"{asm}_{mat}_{i + 1}"
                pc.data.name = pc.name
            pc["assembly"] = asm
            pc["material"] = mat
            final.append(pc)
    for o in lights:
        o["material"] = main_material(o)
        final.append(o)
    for o in final:
        # clean material slots down to the one used
        mat = o["material"]
        me = o.data
        if len(me.materials) > 1:
            keep = MATS[mat]
            me.materials.clear()
            me.materials.append(keep)
            for p in me.polygons:
                p.material_index = 0
        _select_only([o], o)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    return final


def car_frame_point(p):
    """Blender metres -> car-local studs (right, up, forward) from the wheel centre."""
    p = Vector(p)
    return (round(-p.x / STUD, 4), round((p.z - WHEEL_Z) / STUD, 4), round(-p.y / STUD, 4))


def export_roblox(final, outdir):
    os.makedirs(outdir, exist_ok=True)
    # 1) editable .blend at real scale
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(outdir, "S15_SpecR.blend"), compress=True)
    # 2) Roblox copy: nose to Roblox's forward (-Z), 1 unit = 1 stud
    rot = Matrix.Rotation(math.pi, 4, "Z")
    sc = Matrix.Diagonal((1 / STUD, 1 / STUD, 1 / STUD, 1.0))
    for o in final:
        o.matrix_world = sc @ rot @ o.matrix_world
    _select_only(final, final[0])
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    fbx = os.path.join(outdir, "S15_SpecR_Roblox.fbx")
    # Roblox's documented Blender recipe: unit system None (1 unit = 1 stud),
    # Apply Scalings = FBX Unit Scale, Forward = Z, Up = Y. Import in Studio
    # with Scale Unit = Stud, World Forward = Front, World Up = Top.
    us = bpy.context.scene.unit_settings
    old_units = (us.system, us.scale_length)
    us.system, us.scale_length = "NONE", 1.0
    bpy.ops.export_scene.fbx(filepath=fbx, use_selection=True, object_types={"MESH"},
                             global_scale=1.0, apply_unit_scale=True,
                             apply_scale_options="FBX_SCALE_UNITS", axis_forward="Z",
                             axis_up="Y", mesh_smooth_type="OFF", use_triangles=True,
                             add_leaf_bones=False, bake_anim=False, use_custom_props=False,
                             path_mode="COPY", embed_textures=True)
    us.system, us.scale_length = old_units
    # 3) glTF in metres (same orientation)
    for o in final:
        o.matrix_world = Matrix.Diagonal((STUD, STUD, STUD, 1.0)) @ o.matrix_world
    _select_only(final, final[0])
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    glb = os.path.join(outdir, "S15_SpecR.glb")
    try:
        bpy.ops.export_scene.gltf(filepath=glb, use_selection=True, export_format="GLB",
                                  export_apply=True, export_yup=True)
    except TypeError:
        bpy.ops.export_scene.gltf(filepath=glb, use_selection=True, export_format="GLB")
    # back to the modelling orientation so the open scene matches the .blend
    for o in final:
        o.matrix_world = rot @ o.matrix_world
    _select_only(final, final[0])
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    return fbx, glb


LUAU_TEMPLATE = r"""--[[
	Nissan Silvia S15 Spec R - Roblox setup script
	Generated by s15_spec_r.py. Run ONCE in Roblox Studio:

	  1. Import S15_SpecR_Roblox.fbx with the 3D Importer
	     (File Transform: Scale Unit = Stud, World Forward = Front, World Up = Top).
	  2. Select the imported model in the Explorer.
	  3. Paste this whole file into View > Command Bar and press Enter.

	What it does:
	  * paints every MeshPart (material / colour / transparency / reflectance)
	  * builds the A-Chassis layout: Body, Wheels/FL,FR,RL,RR (+Parts, +Fixed),
	    Misc/Door_L, Door_R, Hood, Trunk, SteeringWheel and a DriveSeat (RHD)
	  * adds hinge data + ProximityPrompts so doors, hood and boot open
	  * adds S15_Panels + S15_Lights server scripts and an A-Chassis plugin
	    ("S15 Lights Plugin") - move that plugin into "A-Chassis Tune/Plugins".
]]

local Selection = game:GetService("Selection")
local src = Selection:Get()[1]
assert(src and (src:IsA("Model") or src:IsA("Folder")), "Select the imported S15 model in the Explorer first")

local LOOKS = __LOOKS__
local PARTS = __PARTS__
local HINGES = __HINGES__
local SEATS = __SEATS__
local STEER = __STEER__
local WHEEL_SIZE = __WHEEL__
local WHEELBASE = __WHEELBASE__
local ANCHOR = "__ANCHOR__"

-- collect the imported MeshParts by name
local meshes = {}
for _, d in ipairs(src:GetDescendants()) do
	if d:IsA("MeshPart") then
		meshes[d.Name] = d
	end
end
for name in pairs(PARTS) do
	if not meshes[name] then
		warn("[S15] missing part " .. name)
	end
end

-- looks
for name, part in pairs(meshes) do
	local info = PARTS[name]
	if info then
		local look = LOOKS[info[2]]
		part.Material = Enum.Material[look[1]]
		part.Color = Color3.fromRGB(look[2][1], look[2][2], look[2][3])
		part.Transparency = look[3]
		part.Reflectance = look[4]
		part.CastShadow = look[3] < 0.3
	end
	part.Anchored = true
	part.CanCollide = false
	part.CanTouch = false
	part.CanQuery = true
end

-- the car's own frame, measured from the four tyres
local function pos(n)
	local p = meshes[n]
	assert(p, "missing " .. n)
	return p.Position
end
local FL, FR = pos("Wheel_FL_Rubber"), pos("Wheel_FR_Rubber")
local RL, RR = pos("Wheel_RL_Rubber"), pos("Wheel_RR_Rubber")
local origin = (FL + FR + RL + RR) / 4
local fwd = ((FL + FR) / 2 - (RL + RR) / 2).Unit
local right = (FR - FL).Unit
local up = right:Cross(fwd).Unit
if up.Y < 0.5 then
	warn("[S15] the import looks mirrored or upside down - check the importer's World Up / World Forward settings")
end
-- tolerate an import at another scale: measure the wheelbase
local scale = ((FL + FR) / 2 - (RL + RR) / 2).Magnitude / WHEELBASE
if math.abs(scale - 1) > 0.02 then
	warn(string.format("[S15] model is %.2fx the expected stud scale - offsets scaled to match", scale))
end
local function W(p) -- car-local studs {right, up, forward} -> world
	return origin + (right * p[1] + up * p[2] + fwd * p[3]) * scale
end
local function V(d)
	return right * d[1] + up * d[2] + fwd * d[3]
end
-- hierarchy -----------------------------------------------------------------
local car = Instance.new("Model")
car.Name = "Nissan Silvia S15 Spec R"
local body = Instance.new("Model")
body.Name = "Body"
body.Parent = car
local wheels = Instance.new("Model")
wheels.Name = "Wheels"
wheels.Parent = car
local misc = Instance.new("Model")
misc.Name = "Misc"
misc.Parent = car

local miscModels = {}
for _, n in ipairs({ "Door_L", "Door_R", "Hood", "Trunk", "SteeringWheel" }) do
	local m = Instance.new("Model")
	m.Name = n
	m.Parent = misc
	miscModels[n] = m
end

for _, k in ipairs({ "FL", "FR", "RL", "RR" }) do
	local tyre = meshes["Wheel_" .. k .. "_Rubber"]
	local w = Instance.new("Part")
	w.Name = k
	w.Shape = Enum.PartType.Cylinder
	w.Size = Vector3.new(WHEEL_SIZE[1], WHEEL_SIZE[2], WHEEL_SIZE[2]) * scale
	w.CFrame = CFrame.fromMatrix(tyre.Position, right, up)
	w.Transparency = 1
	w.Anchored = true
	w.CanCollide = false
	w.Parent = wheels
	local p = Instance.new("Model")
	p.Name = "Parts"
	p.Parent = w
	local f = Instance.new("Model")
	f.Name = "Fixed"
	f.Parent = w
	miscModels["Wheel_" .. k] = p
	miscModels["Caliper_" .. k] = f
end

for name, part in pairs(meshes) do
	local info = PARTS[name]
	local asm = info and info[1] or "Body"
	part.Parent = miscModels[asm] or body
end

local anchor = meshes[ANCHOR]
body.PrimaryPart = anchor
for n, m in pairs(miscModels) do
	if m:IsA("Model") and #m:GetChildren() > 0 and not m.PrimaryPart then
		local best
		for _, c in ipairs(m:GetChildren()) do
			if c:IsA("BasePart") and (not best or c.Size.Magnitude > best.Size.Magnitude) then
				best = c
			end
		end
		m.PrimaryPart = best
	end
end

-- seats (right-hand drive)
local function makeSeat(class, name, p)
	local s = Instance.new(class)
	s.Name = name
	s.Size = Vector3.new(1.6, 0.6, 1.6) * scale
	s.CFrame = CFrame.fromMatrix(W(p), right, up)
	s.Transparency = 1
	s.Anchored = true
	s.CanCollide = false
	s.Parent = (class == "VehicleSeat") and car or body
	return s
end
local seat = makeSeat("VehicleSeat", "DriveSeat", SEATS.driver)
makeSeat("Seat", "PassengerSeat", SEATS.passenger)

-- opening panels: hinge frame stored relative to the body anchor -------------
for n, h in pairs(HINGES) do
	local m = miscModels[n]
	local axis = V(h.axis).Unit
	local perp = math.abs(axis:Dot(up)) > 0.9 and fwd or up
	perp = (perp - axis * axis:Dot(perp)).Unit
	local hcf = CFrame.fromMatrix(W(h.pivot), axis, perp)
	m:SetAttribute("HingeOffset", anchor.CFrame:ToObjectSpace(hcf))
	m:SetAttribute("OpenAngle", h.angle)
	local prompt = Instance.new("ProximityPrompt")
	prompt.Name = "OpenPrompt"
	prompt.ActionText = "Open"
	prompt.ObjectText = h.label
	prompt.HoldDuration = 0
	prompt.MaxActivationDistance = 9
	prompt.RequiresLineOfSight = false
	prompt.KeyboardKeyCode = Enum.KeyCode.F
	prompt.Parent = m.PrimaryPart
end

-- steering wheel spin axis
do
	local sw = miscModels.SteeringWheel
	local axis = V(STEER.axis).Unit
	local x = up:Cross(axis).Unit
	local y = axis:Cross(x).Unit
	-- hub frame: Z = steering column axis
	sw:SetAttribute("HubOffset", anchor.CFrame:ToObjectSpace(CFrame.fromMatrix(W(STEER.hub), x, y)))
end

-- runtime scripts --------------------------------------------------------------
local lightEvent = Instance.new("RemoteEvent")
lightEvent.Name = "S15_LightEvent"
lightEvent.Parent = car

local panels = Instance.new("Script")
panels.Name = "S15_Panels"
panels.Source = [==[__PANELS__]==]
panels.Parent = car

local lights = Instance.new("Script")
lights.Name = "S15_Lights"
lights.Source = [==[__LIGHTS__]==]
lights.Parent = car

local pluginFolder = Instance.new("Folder")
pluginFolder.Name = "Move into A-Chassis Tune > Plugins"
pluginFolder.Parent = car
local plug = Instance.new("LocalScript")
plug.Name = "S15 Lights Plugin"
plug.Source = [==[__PLUGIN__]==]
plug.Parent = pluginFolder

car.PrimaryPart = seat
car.Parent = src.Parent or workspace
src:Destroy()
Selection:Set({ car })
print("[S15] done: " .. #car:GetDescendants() .. " instances. Drop the A-Chassis Tune into the model next.")
"""

LUAU_PANELS = r"""-- Doors / hood / boot: ProximityPrompt toggles. Works on an anchored
-- display car (tweens) and on a driving A-Chassis car (servo hinges).
local TweenService = game:GetService("TweenService")
local car = script.Parent
local body = car:WaitForChild("Body")
local anchor = body.PrimaryPart or body:FindFirstChildWhichIsA("BasePart")
local misc = car:WaitForChild("Misc")

local function setup(panel)
	local main = panel.PrimaryPart
	local off = panel:GetAttribute("HingeOffset")
	local angle = panel:GetAttribute("OpenAngle")
	if not (main and off and angle) then
		return
	end
	-- glue the panel's pieces to its main part
	for _, p in ipairs(panel:GetDescendants()) do
		if p:IsA("BasePart") then
			p.Massless = true
			if p ~= main then
				local w = Instance.new("WeldConstraint")
				w.Part0, w.Part1 = main, p
				w.Parent = main
			end
		end
	end
	-- physical hinge (used once the chassis unanchors the car)
	local a0 = Instance.new("Attachment")
	a0.Name = panel.Name .. "_Hinge0"
	a0.CFrame = off
	a0.Parent = anchor
	local a1 = Instance.new("Attachment")
	a1.Name = "Hinge1"
	a1.CFrame = main.CFrame:ToObjectSpace(anchor.CFrame * off)
	a1.Parent = main
	local h = Instance.new("HingeConstraint")
	h.Attachment0, h.Attachment1 = a0, a1
	h.ActuatorType = Enum.ActuatorType.Servo
	h.ServoMaxTorque = 1e7
	h.AngularSpeed = 2.5
	h.TargetAngle = 0
	h.LimitsEnabled = true
	h.LowerAngle = math.min(0, angle) - 1
	h.UpperAngle = math.max(0, angle) + 1
	h.Parent = main

	local open = false
	-- closed pose relative to the hinge (for the anchored / display mode)
	local closedRel = (anchor.CFrame * off):ToObjectSpace(panel:GetPivot())
	local driver = Instance.new("NumberValue")
	driver.Changed:Connect(function(t)
		if main.Anchored then
			local hcf = anchor.CFrame * off
			panel:PivotTo(hcf * CFrame.Angles(math.rad(angle * t), 0, 0) * closedRel)
		end
	end)
	-- when the chassis unanchors the car, let the panel ride on its hinge
	anchor:GetPropertyChangedSignal("Anchored"):Connect(function()
		if not anchor.Anchored then
			for _, p in ipairs(panel:GetDescendants()) do
				if p:IsA("BasePart") then
					p.Anchored = false
				end
			end
			h.TargetAngle = open and angle or 0
		end
	end)
	local prompt = main:FindFirstChild("OpenPrompt")
	prompt.Triggered:Connect(function()
		open = not open
		prompt.ActionText = open and "Close" or "Open"
		if main.Anchored then
			TweenService:Create(driver, TweenInfo.new(0.8, Enum.EasingStyle.Quad), { Value = open and 1 or 0 }):Play()
		else
			h.TargetAngle = open and angle or 0
		end
	end)
end

for _, n in ipairs({ "Door_L", "Door_R", "Hood", "Trunk" }) do
	local m = misc:FindFirstChild(n)
	if m then
		setup(m)
	end
end

-- steering wheel follows the steering input (set by the light plugin)
local sw = misc:FindFirstChild("SteeringWheel")
if sw and sw.PrimaryPart then
	local hub = sw:GetAttribute("HubOffset")
	for _, p in ipairs(sw:GetDescendants()) do
		if p:IsA("BasePart") and p ~= sw.PrimaryPart then
			local w = Instance.new("WeldConstraint")
			w.Part0, w.Part1 = sw.PrimaryPart, p
			w.Parent = p
		end
	end
	local m6 = Instance.new("Motor6D")
	m6.Name = "SteerMotor"
	m6.Part0 = anchor
	m6.Part1 = sw.PrimaryPart
	m6.C0 = hub
	m6.C1 = sw.PrimaryPart.CFrame:ToObjectSpace(anchor.CFrame * hub)
	m6.Parent = sw.PrimaryPart
	anchor:GetPropertyChangedSignal("Anchored"):Connect(function()
		if not anchor.Anchored then
			for _, p in ipairs(sw:GetDescendants()) do
				if p:IsA("BasePart") then
					p.Anchored = false
					p.Massless = true
				end
			end
		end
	end)
	car:GetAttributeChangedSignal("Steer"):Connect(function()
		local s = car:GetAttribute("Steer") or 0
		m6.Transform = CFrame.Angles(0, 0, -math.rad(270) * s)
	end)
end
"""

LUAU_LIGHTS = r"""-- Working lights. Driven by the "S15 Lights Plugin" (A-Chassis) through
-- S15_LightEvent, or by car attributes: Headlights, Brake, Reverse,
-- IndicatorLeft, IndicatorRight, Hazards (booleans).
local car = script.Parent
local event = car:WaitForChild("S15_LightEvent")

local lights = {}
for _, d in ipairs(car:GetDescendants()) do
	if d:IsA("BasePart") and d.Name:sub(1, 6) == "Light_" then
		lights[d.Name] = { part = d, Material = d.Material, Color = d.Color, Transparency = d.Transparency }
	end
end

local function set(name, on, color)
	local l = lights[name]
	if not l then
		return
	end
	if on then
		l.part.Material = Enum.Material.Neon
		l.part.Color = color
		l.part.Transparency = 0
	else
		l.part.Material, l.part.Color, l.part.Transparency = l.Material, l.Color, l.Transparency
	end
end

-- headlight beams
local beams = {}
for _, n in ipairs({ "Light_Head_L", "Light_Head_R" }) do
	local l = lights[n]
	if l then
		local body = car:FindFirstChild("Wheels")
		local fl, rl = body and body:FindFirstChild("FL"), body and body:FindFirstChild("RL")
		local fwd = (fl and rl) and (fl.Position - rl.Position).Unit or l.part.CFrame.LookVector
		local a = Instance.new("Attachment")
		a.CFrame = CFrame.lookAt(Vector3.zero, l.part.CFrame:VectorToObjectSpace(fwd) + Vector3.new(0, -0.06, 0))
		a.Parent = l.part
		local s = Instance.new("SpotLight")
		s.Brightness = 3
		s.Range = 60
		s.Angle = 60
		s.Face = Enum.NormalId.Front
		s.Shadows = true
		s.Enabled = false
		s.Parent = a
		table.insert(beams, s)
	end
end

local WHITE = Color3.fromRGB(255, 250, 235)
local TAIL = Color3.fromRGB(120, 0, 0)
local BRAKE = Color3.fromRGB(255, 20, 20)
local AMBER = Color3.fromRGB(255, 150, 0)
local blink = false

local function refresh()
	local head = car:GetAttribute("Headlights") == true
	local brake = car:GetAttribute("Brake") == true
	local rev = car:GetAttribute("Reverse") == true
	local haz = car:GetAttribute("Hazards") == true
	local il = car:GetAttribute("IndicatorLeft") == true or haz
	local ir = car:GetAttribute("IndicatorRight") == true or haz
	for _, n in ipairs({ "Light_Head_L", "Light_Head_R", "Light_Plate", "Light_Dash" }) do
		set(n, head, n == "Light_Dash" and Color3.fromRGB(255, 70, 20) or WHITE)
	end
	for _, b in ipairs(beams) do
		b.Enabled = head
	end
	for _, n in ipairs({ "Light_Brake_L", "Light_Brake_R" }) do
		set(n, brake or head, brake and BRAKE or TAIL)
	end
	set("Light_Brake_C", brake, BRAKE)
	set("Light_Reverse_L", rev, WHITE)
	set("Light_Reverse_R", rev, WHITE)
	for _, n in ipairs({ "Light_Indicator_FL", "Light_Indicator_SL", "Light_Indicator_RL" }) do
		set(n, il and blink, AMBER)
	end
	for _, n in ipairs({ "Light_Indicator_FR", "Light_Indicator_SR", "Light_Indicator_RR" }) do
		set(n, ir and blink, AMBER)
	end
end

event.OnServerEvent:Connect(function(player, state)
	local seat = car:FindFirstChild("DriveSeat")
	if not (seat and seat.Occupant and seat.Occupant.Parent == player.Character) then
		return
	end
	for k, v in pairs(state) do
		if k == "Steer" then
			car:SetAttribute("Steer", math.clamp(tonumber(v) or 0, -1, 1))
		elseif k == "Headlights" or k == "Brake" or k == "Reverse" or k == "IndicatorLeft"
			or k == "IndicatorRight" or k == "Hazards" then
			car:SetAttribute(k, v == true)
		end
	end
end)

car.AttributeChanged:Connect(refresh)
refresh()
while true do
	task.wait(0.4)
	blink = not blink
	refresh()
end
"""

LUAU_PLUGIN = r"""-- A-Chassis plugin: put this LocalScript in "A-Chassis Tune > Plugins".
-- L = headlights, Z / C = left / right indicator, X = hazards.
local UIS = game:GetService("UserInputService")
local iface = script.Parent
local car = iface:WaitForChild("Car").Value
local values = iface:WaitForChild("Values")
local event = car:WaitForChild("S15_LightEvent")

local state = { Headlights = false, IndicatorLeft = false, IndicatorRight = false, Hazards = false }
local last = { Brake = false, Reverse = false, Steer = 0 }

local function send(t)
	event:FireServer(t)
end

UIS.InputBegan:Connect(function(input, gp)
	if gp then
		return
	end
	local k = input.KeyCode
	if k == Enum.KeyCode.L then
		state.Headlights = not state.Headlights
	elseif k == Enum.KeyCode.Z then
		state.IndicatorLeft = not state.IndicatorLeft
		state.IndicatorRight = false
	elseif k == Enum.KeyCode.C then
		state.IndicatorRight = not state.IndicatorRight
		state.IndicatorLeft = false
	elseif k == Enum.KeyCode.X then
		state.Hazards = not state.Hazards
	else
		return
	end
	send(state)
end)

local function num(name)
	local v = values:FindFirstChild(name)
	return v and tonumber(v.Value) or 0
end

while task.wait(0.1) do
	local gear = num("Gear")
	local brake = num("Brake") > 0.05
	local steer = values:FindFirstChild("SteerC") and num("SteerC") or num("SteerT")
	local t = { Brake = brake, Reverse = gear < 0, Steer = math.floor(steer * 50 + 0.5) / 50 }
	if t.Brake ~= last.Brake or t.Reverse ~= last.Reverse or t.Steer ~= last.Steer then
		last = t
		send(t)
	end
end
"""


def _lua(v, indent=1):
    pad = "\t" * indent
    if isinstance(v, dict):
        items = []
        for k in sorted(v):
            key = k if str(k).isidentifier() else f'["{k}"]'
            items.append(f"{pad}{key} = {_lua(v[k], indent + 1)},")
        return "{\n" + "\n".join(items) + "\n" + "\t" * (indent - 1) + "}"
    if isinstance(v, (list, tuple)):
        return "{ " + ", ".join(_lua(x, indent + 1) for x in v) + " }"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def write_luau(final, outdir):
    looks, parts = {}, {}
    for o in final:
        mat = o["material"]
        lk = roblox_look(mat)
        looks[mat] = [lk["Material"], list(lk["Color"]), lk["Transparency"], lk["Reflectance"]]
        parts[o.name] = [o["assembly"], mat]
    body_parts = [o for o in final if o["assembly"] == "Body" and o["material"] == "Paint"]
    anchor = max(body_parts, key=tri_count).name
    labels = {"Door_L": "Left door", "Door_R": "Right door", "Hood": "Bonnet", "Trunk": "Boot"}
    hinges = {}
    for name, (piv, axis, deg) in HINGES.items():
        # Blender X = car's left; express the axis in (right, up, forward)
        ax = {"X": [-1.0, 0.0, 0.0], "Z": [0.0, 1.0, 0.0]}[axis]
        ang = -deg if axis == "X" else deg
        hinges[name] = {"pivot": list(car_frame_point(piv)), "axis": ax, "angle": ang,
                        "label": labels[name]}
        if axis == "X":
            hinges[name]["axis"] = [1.0, 0.0, 0.0]
    seats = {"driver": list(car_frame_point((-0.37, 0.32, 0.50))),
             "passenger": list(car_frame_point((0.37, 0.32, 0.50)))}
    hub = STEER_HUB
    ax = Vector((0, 0.906, 0.423)).normalized()
    steer = {"hub": list(car_frame_point(hub)), "axis": [round(-ax.x, 4), round(ax.z, 4), round(-ax.y, 4)]}
    lua = LUAU_TEMPLATE
    lua = lua.replace("__LOOKS__", _lua(looks))
    lua = lua.replace("__PARTS__", _lua(parts))
    lua = lua.replace("__HINGES__", _lua(hinges))
    lua = lua.replace("__SEATS__", _lua(seats))
    lua = lua.replace("__STEER__", _lua(steer))
    lua = lua.replace("__WHEEL__", _lua([round(TYRE_W / STUD, 4), round(2 * TYRE_R / STUD, 4)]))
    lua = lua.replace("__ANCHOR__", anchor)
    lua = lua.replace("__WHEELBASE__", f"{WHEELBASE / STUD:.4f}")
    lua = lua.replace("__PANELS__", LUAU_PANELS)
    lua = lua.replace("__LIGHTS__", LUAU_LIGHTS)
    lua = lua.replace("__PLUGIN__", LUAU_PLUGIN)
    path = os.path.join(outdir, "S15_Setup.lua")
    with open(path, "w") as f:
        f.write(lua)
    return path


def write_part_list(final, outdir):
    rows = sorted(((o.name, o["assembly"], o["material"], tri_count(o)) for o in final),
                  key=lambda r: (r[1], r[0]))
    total = sum(r[3] for r in rows)
    with open(os.path.join(outdir, "parts.txt"), "w") as f:
        f.write(f"{'part':34s} {'assembly':16s} {'material':14s} tris\n")
        for r in rows:
            f.write(f"{r[0]:34s} {r[1]:16s} {r[2]:14s} {r[3]}\n")
        f.write(f"\n{len(rows)} parts, {total} triangles total, "
                f"max {max(r[3] for r in rows)} per part (Roblox cap {TRI_LIMIT})\n")
    return rows, total


# --------------------------------------------------------------------------
# Preview rendering
# --------------------------------------------------------------------------

def _reflection_card(name, size, loc, aim, strength, edge, see_through=False):
    """Emissive panel seen only in reflections (glossy rays): white in the
    middle, fading to black over `edge` of its half-width (smootherstep), so
    clear coat and glass mirror a soft studio panel instead of a hard,
    clipped light rectangle. Named Studio_* so the export step removes it.
    see_through: rays pass on behind it (it only adds its light), so it can
    hang in front of another card that other receivers see."""
    sc = bpy.context.scene
    me = bpy.data.meshes.new(name)
    sx, sy = size[0] / 2, size[1] / 2
    me.from_pydata([(-sx, -sy, 0), (sx, -sy, 0), (sx, sy, 0), (-sx, sy, 0)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name="UVMap")
    for i, c in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        uv.data[i].uv = c
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector(aim) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    for attr in ("visible_camera", "visible_diffuse", "visible_shadow",
                 "visible_transmission", "visible_volume_scatter"):
        setattr(ob, attr, False)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["UV"], sep.inputs[0])
    fac = []
    for ax in ("X", "Y"):
        lin = nt.nodes.new("ShaderNodeMath")          # |2u - 1|: 0 centre, 1 edge
        lin.operation = "MULTIPLY_ADD"
        lin.inputs[1].default_value, lin.inputs[2].default_value = 2.0, -1.0
        nt.links.new(sep.outputs[ax], lin.inputs[0])
        ab = nt.nodes.new("ShaderNodeMath")
        ab.operation = "ABSOLUTE"
        nt.links.new(lin.outputs[0], ab.inputs[0])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.interpolation_type = "SMOOTHERSTEP"
        mr.inputs["From Min"].default_value = 1.0
        mr.inputs["From Max"].default_value = 1.0 - edge
        nt.links.new(ab.outputs[0], mr.inputs["Value"])
        fac.append(mr.outputs["Result"])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(fac[0], mul.inputs[0])
    nt.links.new(fac[1], mul.inputs[1])
    amp = nt.nodes.new("ShaderNodeMath")
    amp.operation = "MULTIPLY"
    amp.inputs[1].default_value = strength
    nt.links.new(mul.outputs[0], amp.inputs[0])
    nt.links.new(amp.outputs[0], em.inputs["Strength"])
    shader = em.outputs[0]
    if see_through:
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(shader, add.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], add.inputs[1])
        shader = add.outputs[0]
    nt.links.new(shader, out.inputs["Surface"])
    me.materials.append(m)
    return ob


def _bare_metal(ob):
    """Chrome, alloy, badges, lamp reflectors...: every material is metallic,
    apart from rough ones (>= 0.7) that show no highlight anyway (the black
    hub face of a brake disc). Painted or glazed parts are never bare metal."""
    if ob.type != "MESH":
        return False
    vals = []
    for m in ob.data.materials:
        p = m.node_tree.nodes.get("Principled BSDF") if m and m.use_nodes else None
        if p is None:
            return False
        vals.append((p.inputs["Metallic"].default_value, p.inputs["Roughness"].default_value))
    return (any(mt >= 0.8 for mt, r in vals)
            and all(mt >= 0.8 or r >= 0.7 for mt, r in vals))


def _link_set(name, obs, exclude=False):
    """A light-linking receiver collection (not linked to the scene; named
    Studio_* so the export step removes it). exclude: every object except
    these receives the light."""
    col = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    for o in list(col.objects):
        col.objects.unlink(o)
    for o in obs:
        col.objects.link(o)
    for co in col.collection_objects:
        co.light_linking.link_state = "EXCLUDE" if exclude else "INCLUDE"
    return col


def highlight_shoulder(sc, knee=0.75, white=0.98):
    """Standard has no highlight roll-off, so the brightest paint goes flat
    (one channel pinned at 1.0). This compositor shoulder scales each pixel's
    RGB so that max(R, G, B) stays linear up to `knee` and then eases towards
    `white` (display-linear, after exposure) without reaching it. Hue and
    saturation are kept, so the yellow stays vivid. The roll-off is
    rational, over * span / (over + span): its long tail lets a close-up
    hood keep its shading instead of piling up just under `white`."""
    gain = 2.0 ** sc.view_settings.exposure
    k, span = knee / gain, (white - knee) / gain
    sc.use_nodes = True
    cn = sc.node_tree
    cn.nodes.clear()
    rl = cn.nodes.new("CompositorNodeRLayers")
    sep = cn.nodes.new("CompositorNodeSeparateColor")
    cn.links.new(rl.outputs["Image"], sep.inputs[0])

    def op(kind, a, b=None):
        n = cn.nodes.new("CompositorNodeMath")
        n.operation = kind
        for i, v in enumerate((a, b)):
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            elif v is not None:
                cn.links.new(v, n.inputs[i])
        return n.outputs[0]
    peak = op("MAXIMUM", op("MAXIMUM", sep.outputs[0], sep.outputs[1]), sep.outputs[2])
    over = op("MAXIMUM", op("SUBTRACT", peak, k), 0.0)
    roll = op("DIVIDE", op("MULTIPLY", over, span), op("ADD", over, span))
    ratio = op("DIVIDE", op("ADD", op("MINIMUM", peak, k), roll), op("MAXIMUM", peak, 1e-6))
    mul = cn.nodes.new("CompositorNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    cn.links.new(rl.outputs["Image"], mul.inputs[1])
    cn.links.new(ratio, mul.inputs[2])
    cn.links.new(mul.outputs[0], cn.nodes.new("CompositorNodeComposite").inputs["Image"])


# glossy-only twins of the studio lights that only bare metal sees:
# (x the light's energy, x its size). Stronger side strips for the rims, a
# broad front panel so the headlamp bowls and rings catch a reflection.
METAL_TWINS = {"Studio_Top": (1.0, 1.0), "Studio_SideL": (1.5, 1.0), "Studio_SideR": (1.5, 1.0),
               "Studio_Front": (6.0, 2.4), "Studio_Rear": (1.0, 1.0)}
GLASS_CARD = 1.0       # strength of the top card as the windows see it


def setup_render_scene(res=(1280, 720), samples=48):
    """Photo studio: grey seamless floor, a big overhead softbox and long
    side strips. The area lights only light the car (they are hidden from
    glossy rays); what the clear coat and the glass mirror are dim
    soft-edged reflection cards, so under the Standard view transform the
    highlights stay soft instead of clipping to ragged white streaks. Bare
    metal alone (light linking) also mirrors glossy-only twins of the lights,
    so chrome and alloy still sparkle. Works on a freshly built scene and
    on a saved one; calling it again is harmless."""
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    try:
        sc.cycles.use_denoising = True
    except AttributeError:
        pass
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"   # AgX/Filmic wash the yellow out
    sc.view_settings.look = "None"
    sc.view_settings.exposure = -0.6
    highlight_shoulder(sc)
    world = bpy.data.worlds.new("Studio") if sc.world is None else sc.world
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.20, 0.20, 0.21, 1)
    bg.inputs["Strength"].default_value = 1.3      # ambient fill; backdrop ~ reference grey
    nt.links.new(bg.outputs[0], out.inputs[0])
    if "Ground" not in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, 0))
        g = bpy.context.object
        g.name = "Ground"
        gm = bpy.data.materials.new("StudioFloor")
        gm.use_nodes = True
        gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.11, 0.11, 0.115, 1)
        gm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.55
        g.data.materials.append(gm)
    # diffuse-only key/fill lights. The top is kept moderate so a close-up
    # hood stays under the shoulder (it took ~2/3 of the hood's light); the
    # sides, front and rear lift the flanks and the ends to the reference
    boxes = (("Studio_Top", (3.2, 7.0), (0, 0, 5.2), (0, 0, 0), 450),
             ("Studio_SideL", (7.5, 1.2), (5.0, 0, 2.0), (0, 0, 0.6), 400),
             ("Studio_SideR", (7.5, 1.2), (-5.0, 0, 2.0), (0, 0, 0.6), 400),
             ("Studio_Front", (3.0, 1.5), (0, -7.5, 2.5), (0, 0, 0.5), 175),
             ("Studio_Rear", (3.0, 1.5), (0, 7.5, 2.5), (0, 0, 0.5), 260))
    for name, size, loc, aim, energy in boxes:
        lo = bpy.data.objects.get(name)
        if lo is None:
            ld = bpy.data.lights.new(name, "AREA")
            ld.shape = "RECTANGLE"
            ld.size, ld.size_y = size
            ld.energy = energy
            lo = bpy.data.objects.new(name, ld)
            sc.collection.objects.link(lo)
            lo.location = loc
            lo.rotation_euler = (Vector(aim) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        lo.visible_glossy = False       # no hard mirrored light boxes in paint/glass
    # what the paint and glass reflect instead: radiance 1.5-2.5 stays below
    # white even at grazing Fresnel (box radiance was ~8-10). The side strips
    # are dim and wide-edged: in a side view the far one grazes along the
    # hood crown and the roofline and laid a cream band there.
    cards = (("Studio_CardTop", (4.5, 9.0), (0, 0, 5.15), (0, 0, 0), 2.5, 0.6),
             ("Studio_CardSideL", (9.0, 2.0), (5.05, 0, 2.0), (0, 0, 0.6), 1.0, 0.8),
             ("Studio_CardSideR", (9.0, 2.0), (-5.05, 0, 2.0), (0, 0, 0.6), 1.0, 0.8),
             ("Studio_CardFront", (4.0, 2.0), (0, -7.55, 2.5), (0, 0, 0.5), 2.0, 0.5),
             ("Studio_CardRear", (4.0, 2.0), (0, 7.55, 2.5), (0, 0, 0.5), 1.5, 0.5))
    for name, size, loc, aim, strength, edge in cards:
        if name not in bpy.data.objects:
            _reflection_card(name, size, loc, aim, strength, edge)
    # bare metal (rims, chrome, badges, lamp reflectors, exhaust) would look
    # dull grey with only the dim cards to mirror: it gets glossy-only twins
    # of the lights through light linking, which paint and glass never see
    metal = _link_set("Studio_MetalReceivers", [o for o in sc.objects if _bare_metal(o)])
    for name, size, loc, aim, energy in boxes:
        lo = bpy.data.objects[name]
        go = bpy.data.objects.get(name + "Gloss")
        if go is None:
            go = bpy.data.objects.new(name + "Gloss", lo.data.copy())
            sc.collection.objects.link(go)
        # 15 cm towards the car: in front of the cards (the top card hangs
        # under the top light), so metal's own rays reach the twin
        go.location = Vector(loc) + (Vector(aim) - Vector(loc)).normalized() * 0.15
        go.rotation_euler = lo.rotation_euler
        k_energy, k_size = METAL_TWINS[name]
        go.data.energy = energy * k_energy
        go.data.size, go.data.size_y = size[0] * k_size, size[1] * k_size
        for attr in ("visible_camera", "visible_diffuse", "visible_transmission",
                     "visible_volume_scatter"):
            setattr(go, attr, False)
        go.visible_glossy = True
        go.light_linking.receiver_collection = metal
    # the windows mirror their own, dimmer copy of the top card (at full
    # strength it laid a flat grey veil over the cabin). It hangs just below
    # the top card and is see-through, so the paint still sees the top card.
    glass = [o for o in sc.objects if o.type == "MESH"
             and any(m and m.name.split(".")[0] == "Glass" for m in o.data.materials)]
    top = bpy.data.objects["Studio_CardTop"]
    if glass:
        top.light_linking.receiver_collection = _link_set("Studio_NoGlass", glass, exclude=True)
        if "Studio_CardGlass" not in bpy.data.objects:
            _reflection_card("Studio_CardGlass", (4.5, 9.0), (0, 0, 5.12), (0, 0, 0),
                             GLASS_CARD, 1.0, see_through=True)
        bpy.data.objects["Studio_CardGlass"].light_linking.receiver_collection = \
            _link_set("Studio_GlassOnly", glass)


def render_view(path, loc, target=(0, 0, 0.55), lens=50, ortho=None):
    sc = bpy.context.scene
    cam = bpy.data.objects.get("PreviewCam")
    if cam is None:
        cd = bpy.data.cameras.new("PreviewCam")
        cam = bpy.data.objects.new("PreviewCam", cd)
        sc.collection.objects.link(cam)
    cam.location = loc
    d = Vector(target) - Vector(loc)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    if ortho:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = ortho
    else:
        cam.data.type = "PERSP"
        cam.data.lens = lens
    sc.camera = cam
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


PREVIEW_VIEWS = {
    "front34": ((-8.9, -11.2, 2.9), (0, -0.15, 0.58), 85, None),
    "rear34": ((8.2, 11.6, 3.0), (0, 0.2, 0.6), 85, None),
    "side": ((13.0, 0.0, 1.0), (0, 0, 0.62), 85, None),
    "front": ((0.0, -13.0, 1.1), (0, 0, 0.62), 85, None),
    "rear": ((0.0, 13.0, 1.2), (0, 0, 0.66), 85, None),
    "top": ((0.0, 0.0, 9.0), (0, 0.0001, 0), 50, 5.0),
    "nose": ((-2.6, -4.6, 1.3), (0.0, -1.9, 0.5), 50, None),
    "hoodtop": ((-1.6, -3.9, 1.6), (0.0, -1.85, 0.6), 50, None),
    "open": ((-4.2, -5.2, 2.6), (0, -0.4, 0.6), 40, None),
    "tail": ((2.4, 4.6, 1.5), (0.0, 1.9, 0.6), 50, None),
    "tlight": ((1.3, 3.4, 1.05), (0.55, 2.1, 0.85), 50, None),
    "cabin": ((2.3, 0.75, 1.20), (-0.15, -0.20, 0.72), 26, None),
    "bay": ((-1.2, -3.1, 1.9), (0.0, -1.35, 0.55), 32, None),
    "under": ((3.2, -4.0, -0.25), (0.0, 0.0, 0.25), 28, None),
}


def render_previews(outdir, views=None, samples=48, res=(1280, 720)):
    setup_render_scene(res, samples)
    os.makedirs(outdir, exist_ok=True)
    for name, (loc, tgt, lens, ortho) in PREVIEW_VIEWS.items():
        if views and name not in views:
            continue
        render_view(os.path.join(outdir, f"preview_{name}.png"), loc, tgt, lens, ortho)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def reset_scene():
    MATS.clear()
    if bpy.app.background:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        return
    # inside the Blender UI: build into a fresh scene, leave the user's data alone
    sc = bpy.data.scenes.new("S15_SpecR")
    bpy.context.window.scene = sc


def build():
    reset_scene()
    setup_materials()
    resolve_body_mesh()
    B = build_body()
    body = B["body"]
    parts = {"Body": body}
    parts.update(cut_windows(body, B["ref"]))
    parts.update(cut_lamps_and_openings(body, B["ref"]))
    cut_seams(body, B["layer"])
    panels = build_panels(body, B["layer"])
    detail_doors(panels, B["ref"])
    parts.update(panels)
    for ob in parts.values():
        ob["assembly"] = ob.get("assembly", "Body")
        shade_smooth(ob, 55 if BODY_MESH else 35)      # mesh bodies: no stray creases
    for name in panels:
        panels[name]["assembly"] = name
    parts["Glass_Door_L"]["assembly"] = "Door_L"
    parts["Glass_Door_R"]["assembly"] = "Door_R"
    parts.update(build_exterior_details(B["ref"]))
    tag_headliner(body)
    parts.update(build_interior(body))
    parts.update(build_engine_bay())
    parts.update(build_underbody())
    parts.update(build_wheels())
    parts["SteeringWheel"]["assembly"] = "SteeringWheel"
    delete_object(B["ref"])
    delete_object(B["layer"])
    return parts


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    opts = {"out": os.path.join(here, "export"), "render": False, "views": None, "open": False,
            "export": True,
            "samples": 48}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--out":
            opts["out"] = argv[i + 1]
            i += 1
        elif a == "--clay":
            global CLAY
            CLAY = True
        elif a == "--body":
            global BODY_MESH
            BODY_MESH = argv[i + 1] if argv[i + 1].lower() != "loft" else None
            i += 1
        elif a == "--prepare-body":
            opts["prepare"] = (argv[i + 1], argv[i + 2])
            i += 2
        elif a == "--export":
            opts["export"] = True
        elif a == "--no-export":
            opts["export"] = False
        elif a == "--open":
            opts["open"] = True
        elif a == "--render":
            opts["render"] = True
        elif a == "--views":
            opts["views"] = argv[i + 1].split(",")
            i += 1
        elif a == "--samples":
            opts["samples"] = int(argv[i + 1])
            i += 1
        i += 1
    return opts


def main():
    opts = parse_args()
    if opts.get("prepare"):
        prepare_body(*opts["prepare"])
        return
    parts = build()
    if opts["open"]:
        open_panels(parts)
    if opts["render"]:
        render_previews(opts["out"], opts["views"], opts["samples"])
    if opts["export"]:
        for o in [o for o in bpy.data.objects
                  if o.name in ("Ground", "PreviewCam") or o.name.startswith("Studio_")]:
            bpy.data.objects.remove(o, do_unlink=True)
        for c in [c for c in bpy.data.collections if c.name.startswith("Studio_")]:
            bpy.data.collections.remove(c)        # light-linking sets
        os.makedirs(opts["out"], exist_ok=True)
        final = finalize_parts(parts)
        rows, total = write_part_list(final, opts["out"])
        write_luau(final, opts["out"])
        export_roblox(final, opts["out"])
        print(f"exported {len(rows)} parts, {total} triangles -> {opts['out']}")


if __name__ == "__main__":
    main()
