"""Week 5 - a frame generated from its dimensions, not typed out.

    python code/w05/frame.py

A two-bay, three-storey frame has 12 nodes and 15 members. Typed out by hand
that is fifty-odd lines in which every tag has to be right, and adding a storey
means writing them again. Generated from a loop it is the twenty lines below,
and adding a storey means changing a 3 to a 4.

The structural content this week is small - it is the same linear static
analysis as week 4, with more of it. The programming content is the whole
point, and it comes down to one idea: a TAG FORMULA. Decide once how a node's
tag follows from where it is, and the loops write themselves.

Units: N, mm, s.
"""

import sys
from pathlib import Path

import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --8<-- [start:params]
FRAME = {
    "n_storeys": 3,
    "n_bays": 2,
    "bay": 6000.0,           # mm     bay width
    "storey": 3500.0,        # mm     storey height
    "E": 200000.0,           # MPa    steel
    # columns: HEB 300 about the strong axis
    "A_col": 14910.0,        # mm^2
    "I_col": 2.517e8,        # mm^4
    # beams: IPE 400
    "A_beam": 8446.0,        # mm^2
    "I_beam": 2.313e8,       # mm^4
    "w": 25.0,               # N/mm   gravity load on every beam
    "H": 40000.0,            # N      lateral load at every floor
}
# --8<-- [end:params]


# --8<-- [start:tags]
def node_tag(line, storey):
    """The tag of the node on a given column line at a given floor.

    line   0, 1, 2 ...   from the left
    storey 0 at the base, 1 at the first floor, and so on

    The formula is the design decision of this whole script. It has to be
    unique, and it should be READABLE: node 203 is line 2, storey 3, and a
    number in an error message tells you where to look. A running counter
    would also be unique and would tell you nothing.

    The 100 caps the frame at 99 storeys, which is a limit worth stating rather
    than discovering.
    """
    return 100 * line + storey


def beam_tag(line, storey):
    """Beam from `line` to `line + 1`, at `storey`. Tags start at 1000."""
    return 1000 + 100 * line + storey


def column_tag(line, storey):
    """Column from `storey - 1` up to `storey`. Tags start at 2000."""
    return 2000 + 100 * line + storey
# --8<-- [end:tags]


# --8<-- [start:build_model]
def build_model(p):
    """Generate the whole frame from its dimensions.

    Every loop below is the same shape: walk over the lines and storeys, and
    let the tag formulas turn a position into a name. Nothing here changes when
    the frame gets bigger.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    lines = range(p["n_bays"] + 1)
    storeys = range(1, p["n_storeys"] + 1)

    # Nodes, including the base.
    for line in lines:
        for storey in range(p["n_storeys"] + 1):
            ops.node(node_tag(line, storey),
                     line * p["bay"], storey * p["storey"])

    # Fixed bases. A moment frame needs them: on pins the whole frame is a
    # mechanism until the beams pick the sway up, and the drift is enormous.
    for line in lines:
        ops.fix(node_tag(line, 0), 1, 1, 1)

    ops.geomTransf("Linear", 1)

    # Columns: one per line, per storey.
    for line in lines:
        for storey in storeys:
            ops.element("elasticBeamColumn", column_tag(line, storey),
                        node_tag(line, storey - 1), node_tag(line, storey),
                        p["A_col"], p["E"], p["I_col"], 1)

    # Beams: one per bay, per floor.
    for line in range(p["n_bays"]):
        for storey in storeys:
            ops.element("elasticBeamColumn", beam_tag(line, storey),
                        node_tag(line, storey), node_tag(line + 1, storey),
                        p["A_beam"], p["E"], p["I_beam"], 1)
# --8<-- [end:build_model]


# --8<-- [start:loads]
def apply_gravity(p, factor=1.0):
    """Uniform load on every beam, in pattern 1."""
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for line in range(p["n_bays"]):
        for storey in range(1, p["n_storeys"] + 1):
            ops.eleLoad("-ele", beam_tag(line, storey),
                        "-type", "-beamUniform", -factor * p["w"])


def apply_lateral(p, factor=1.0):
    """A horizontal force at the left-hand node of every floor, in pattern 2.

    Two patterns, not one. Keeping gravity and wind in separate patterns is
    what makes a load COMBINATION possible: the factors below are applied when
    the pattern is created, so 1.2G + 1.6W is two calls with two factors rather
    than a rebuilt model.
    """
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    for storey in range(1, p["n_storeys"] + 1):
        ops.load(node_tag(0, storey), factor * p["H"], 0.0, 0.0)
# --8<-- [end:loads]


# --8<-- [start:run]
def run():
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    ok = ops.analyze(1)
    if ok != 0:
        raise RuntimeError(f"analysis failed, analyze() returned {ok}")
    return ok
# --8<-- [end:run]


# --8<-- [start:drift]
def floor_sway(p, storey):
    """The sideways movement of a floor: the MEAN of its nodes, not one of them.

    Reading one column line looks equivalent and is not. Under gravity alone
    this frame spreads: the left roof node moves +0.094 mm, the right one
    -0.094 mm, and the middle one not at all. Report the left node and you will
    publish 0.094 mm of "sway" caused by vertical load, which is not sway at
    all -- it is the two halves of a symmetric frame leaning apart.

    The mean is zero for any symmetric spreading, and equals the sway when the
    floor really does translate. That is the quantity drift is defined on.
    """
    lines = range(p["n_bays"] + 1)
    ux = [ops.nodeDisp(node_tag(line, storey), 1) for line in lines]
    return sum(ux) / len(ux)


def storey_drifts(p):
    """Inter-storey drift ratio for every storey.

    Drift ratio is the movement of a floor RELATIVE TO THE ONE BELOW, divided
    by the storey height. Using the absolute displacement instead is the
    commonest mistake in this calculation, and it flatters the upper storeys
    badly: they move the most in total while often deforming the least.
    """
    return [(floor_sway(p, s) - floor_sway(p, s - 1)) / p["storey"]
            for s in range(1, p["n_storeys"] + 1)]
# --8<-- [end:drift]


# --8<-- [start:solve]
def solve(p=FRAME, gravity=1.0, lateral=1.0):
    """Build, load with the given combination factors, run, and report.

    Because the factors are arguments, a load combination is one call.
    """
    build_model(p)
    if gravity:
        apply_gravity(p, gravity)
    if lateral:
        apply_lateral(p, lateral)
    run()
    ops.reactions()

    lines = range(p["n_bays"] + 1)
    base_shear = sum(ops.nodeReaction(node_tag(line, 0), 1) for line in lines)
    base_axial = sum(ops.nodeReaction(node_tag(line, 0), 2) for line in lines)
    return {
        "roof_sway": floor_sway(p, p["n_storeys"]),
        "roof_ux": [ops.nodeDisp(node_tag(line, p["n_storeys"]), 1)
                    for line in lines],
        "drifts": storey_drifts(p),
        "base_shear": base_shear,
        "base_axial": base_axial,
    }
# --8<-- [end:solve]


def counts(p=FRAME):
    """How big the model is, from the dimensions alone."""
    nodes = (p["n_bays"] + 1) * (p["n_storeys"] + 1)
    columns = (p["n_bays"] + 1) * p["n_storeys"]
    beams = p["n_bays"] * p["n_storeys"]
    return {"nodes": nodes, "columns": columns, "beams": beams,
            "elements": columns + beams}


if __name__ == "__main__":
    size = counts()
    print(f"Frame: {FRAME['n_bays']} bays x {FRAME['n_storeys']} storeys"
          f"  ->  {size['nodes']} nodes, {size['elements']} elements")

    print("\nLoad combinations")
    print(f"  {'combination':<20}{'roof sway [mm]':>16}{'base shear [N]':>16}")
    for name, g, l in (("1.0 G", 1.0, 0.0),
                       ("1.0 W", 0.0, 1.0),
                       ("1.0 G + 1.0 W", 1.0, 1.0),
                       ("1.2 G + 1.6 W", 1.2, 1.6)):
        r = solve(FRAME, gravity=g, lateral=l)
        print(f"  {name:<20}{r['roof_sway']:16.4f}{r['base_shear']:16.1f}")

    # Why sway is the MEAN and not one node's displacement.
    g_only = solve(FRAME, gravity=1.0, lateral=0.0)
    print("\nGravity alone: the roof nodes, left to right")
    print("  " + "  ".join(f"{u:+.4f}" for u in g_only["roof_ux"]) + " mm")
    print(f"  mean = {g_only['roof_sway']:.4f} mm")
    print("  The frame spreads, it does not sway. Report the left node on its")
    print("  own and you publish 0.09 mm of sway caused by vertical load.")

    print("\nInter-storey drift under 1.0 G + 1.0 W")
    r = solve(FRAME, 1.0, 1.0)
    for i, d in enumerate(r["drifts"], start=1):
        print(f"  storey {i}: {d * 100:7.4f} %   (limit 2.00 %)")

    print("\n  The frame is the same twenty lines whatever its size:")
    for n in (3, 6, 12):
        big = counts(dict(FRAME, n_storeys=n))
        print(f"    {n:2d} storeys -> {big['nodes']:3d} nodes,"
              f" {big['elements']:3d} elements")
