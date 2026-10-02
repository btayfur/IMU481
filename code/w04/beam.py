"""Week 4 - a simply supported beam under a uniformly distributed load.

    python code/w04/beam.py

Everything so far has been pin-jointed: bars carrying axial force, nodes with
two degrees of freedom, no bending anywhere. This is the first model with a
BEAM in it, which brings three new things at once -- a third degree of freedom
at every node, a geometric transformation, and a load applied along a member
rather than at a point.

It also brings the first modelling decision that is genuinely yours to make:
how many elements to use. The answer is not what you would guess from other
finite element work, and the reason is worth the whole chapter.

Units: N, mm, s.
"""

import sys
from pathlib import Path

import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --8<-- [start:params]
BEAM = {
    "L": 6000.0,       # mm      span
    "E": 200000.0,     # MPa     steel
    "I": 8.356e7,      # mm^4    IPE 300, strong axis
    "A": 5380.0,       # mm^2    IPE 300
    "w": 20.0,         # N/mm    = 20 kN/m, downward, over the whole span
}
# --8<-- [end:params]


# --8<-- [start:build_model]
def build_model(p, n_elements=2):
    """Simply supported beam, divided into n_elements equal pieces.

    Three things are new here and all three are consequences of one fact: a
    beam carries bending, and a bar does not.

    ndf is 3, not 2. A node now has a rotation as well as two translations,
    because a beam end can rotate and the element resists that rotation.

    A geomTransf appears. An element's stiffness is written in its own local
    axes -- along the member and across it -- and something has to rotate that
    into the global axes the nodes live in. A Truss needed no such thing
    because axial stiffness looks the same from any direction.

    The supports are no longer both pinned: fixing the rotation at both ends
    would build a fixed-ended beam, which is a different structure.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    dx = p["L"] / n_elements
    for i in range(n_elements + 1):
        ops.node(i + 1, i * dx, 0.0)

    ops.fix(1, 1, 1, 0)                     # pin:    ux, uy held, rotation free
    ops.fix(n_elements + 1, 0, 1, 0)        # roller: uy held only

    ops.geomTransf("Linear", 1)
    for e in range(n_elements):
        ops.element("elasticBeamColumn", e + 1, e + 1, e + 2,
                    p["A"], p["E"], p["I"], 1)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    # A load along the member, not at a node. Wy is transverse in the element's
    # LOCAL axes, so downward on a horizontal beam is negative. Every element
    # needs its own share -- eleLoad applies to the element you name, and a
    # forgotten one leaves a stretch of beam unloaded and nothing says so.
    for e in range(n_elements):
        ops.eleLoad("-ele", e + 1, "-type", "-beamUniform", -p["w"])
# --8<-- [end:build_model]


# --8<-- [start:run]
def run():
    """The same seven objects as week 2. Nothing about bending changes them."""
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


# --8<-- [start:results]
def results(p, n_elements=2):
    """Read back the deflected shape, the reactions and the internal moment.

    eleResponse(tag, 'forces') returns six numbers for a 2D beam element:
    axial, shear and moment at the i end, then the same three at the j end, all
    in LOCAL axes. The moment at an interior node is therefore available twice
    -- once from the element on each side -- and the two must agree. That is a
    free equilibrium check, and it is used below.
    """
    ops.reactions()

    n_nodes = n_elements + 1
    dx = p["L"] / n_elements
    shape = [(i * dx, ops.nodeDisp(i + 1, 2)) for i in range(n_nodes)]

    reaction_left = ops.nodeReaction(1, 2)
    reaction_right = ops.nodeReaction(n_nodes, 2)

    # Internal moment at every interior node, taken from the element to its left.
    moments = [ops.eleResponse(e + 1, "forces")[5] for e in range(n_elements - 1)]

    return {
        "shape": shape,
        "R_left": reaction_left,
        "R_right": reaction_right,
        "node_moments": moments,
    }
# --8<-- [end:results]


# --8<-- [start:hand]
def hand(p):
    """The three textbook results for this beam.

        reaction   R = wL/2
        moment     M = wL^2/8   at midspan
        deflection d = 5wL^4/384EI   at midspan
    """
    w, L, E, I = p["w"], p["L"], p["E"], p["I"]
    return {
        "R": w * L / 2.0,
        "M_mid": w * L ** 2 / 8.0,
        "d_mid": -5.0 * w * L ** 4 / (384.0 * E * I),
    }
# --8<-- [end:hand]


# --8<-- [start:solve]
def solve(p=BEAM, n_elements=2):
    """Build, run, read. Returns the results dict."""
    build_model(p, n_elements)
    run()
    return results(p, n_elements)
# --8<-- [end:solve]


def midspan_deflection(p=BEAM, n_elements=2):
    """The deflection at mid-span -- if the mesh has a node there.

    Returns None when it does not, which is the point of exercise 4.2: an odd
    number of elements puts no node at the middle, and the largest deflection
    in the beam is then simply not among the numbers the model can report.
    """
    got = solve(p, n_elements)
    for x, uy in got["shape"]:
        if abs(x - p["L"] / 2.0) < 1e-9:
            return uy
    return None


if __name__ == "__main__":
    want = hand(BEAM)
    got = solve(BEAM, n_elements=2)

    print("Simply supported beam, 6 m span, 20 kN/m, IPE 300")
    print(f"  {'':<22}{'model':>14}{'hand':>14}")
    print(f"  {'reaction, left [N]':<22}{got['R_left']:14.2f}{want['R']:14.2f}")
    print(f"  {'reaction, right [N]':<22}{got['R_right']:14.2f}{want['R']:14.2f}")
    print(f"  {'midspan moment [Nmm]':<22}{got['node_moments'][0]:14.4g}"
          f"{want['M_mid']:14.4g}")
    print(f"  {'midspan deflection [mm]':<22}"
          f"{midspan_deflection(BEAM, 2):13.4f}{want['d_mid']:14.4f}")

    print("\n  Deflection reported, against number of elements:")
    for n in (1, 2, 3, 4, 8):
        d = midspan_deflection(BEAM, n)
        shown = "  no node at midspan" if d is None else f"{d:14.4f} mm"
        print(f"    {n} element(s): {shown}")

    print("\n  Refining the mesh does not make the answer more accurate --")
    print("  it was already exact. It makes more of the beam VISIBLE.")
