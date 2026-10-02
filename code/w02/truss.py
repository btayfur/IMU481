"""Week 2 - the smallest complete OpenSees model, built one layer at a time.

    python code/w02/truss.py

A two-bar pin-jointed truss carrying a vertical load at its apex. It is chosen
because every quantity in it can be checked by hand: the bar forces follow from
one joint equilibrium, and the apex displacement from the bar elongations.

Running this teaches the ORDER of the six steps and why each one has to come
where it does. Read build_model() top to bottom and notice that no step could
be moved: an element needs its nodes, a load needs its pattern, and a pattern
needs its time series.

Units: N, mm, s.
"""

import math

# --8<-- [start:import]
import openseespy.opensees as ops
# --8<-- [end:import]


# --8<-- [start:params]
# Geometry:      apex at mid-span, height H above the two supports
# Both bars are the same steel section.
TRUSS = {
    "span": 4000.0,      # mm     distance between the two supports
    "H": 1500.0,         # mm     rise of the apex above the supports
    "A": 1200.0,         # mm^2   cross-sectional area of each bar
    "E": 200000.0,       # MPa    steel
    "P": 50000.0,        # N      = 50 kN, downward at the apex
}
# --8<-- [end:params]


# --8<-- [start:build_model]
def build_model(p):
    """Build the two-bar truss.

    The six steps below are the whole of model building in OpenSees, and they
    are in the only order that works. Each one names things by an integer TAG,
    and those tags are how the later steps refer back to the earlier ones.
    """
    # 1. Start clean, and say what kind of model this is.
    #    ndm = spatial dimensions, ndf = degrees of freedom per node.
    #    A pin-jointed truss carries no moments, so 2 DOF per node is enough.
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    # 2. Nodes: a tag and its coordinates. Nothing else.
    ops.node(1, 0.0, 0.0)                       # left support
    ops.node(2, p["span"], 0.0)                 # right support
    ops.node(3, p["span"] / 2.0, p["H"])        # apex

    # 3. Restraints: 1 means "held", 0 means "free", one flag per DOF.
    ops.fix(1, 1, 1)                            # pinned
    ops.fix(2, 1, 1)                            # pinned

    # 4. Material, then elements that use it. The material must exist first --
    #    the element refers to it by tag, and OpenSees resolves that at once.
    ops.uniaxialMaterial("Elastic", 1, p["E"])
    ops.element("Truss", 1, 1, 3, p["A"], 1)    # ele 1: node 1 to node 3
    ops.element("Truss", 2, 2, 3, p["A"], 1)    # ele 2: node 2 to node 3

    # 5. How the load grows with pseudo-time, then the pattern that uses it.
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    # 6. The load itself: one value per DOF of the node it acts on.
    ops.load(3, 0.0, -p["P"])
# --8<-- [end:build_model]


# --8<-- [start:run]
def run():
    """Apply the load in a single step.

    These seven objects are the analysis. Week 3 gives them names and says what
    the alternatives are; for a linear problem this combination always works.
    """
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandSPD")
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
def results(p):
    """Read the answers back out of the model.

    Two different questions, two different commands. nodeDisp asks a node what
    it did; eleResponse asks an element what it carries. Neither reads a file --
    the model is still in memory and can simply be interrogated.
    """
    ux, uy = ops.nodeDisp(3, 1), ops.nodeDisp(3, 2)
    n1 = ops.eleResponse(1, "axialForce")[0]
    n2 = ops.eleResponse(2, "axialForce")[0]
    return {"ux": ux, "uy": uy, "N1": n1, "N2": n2}
# --8<-- [end:results]


# --8<-- [start:hand_check]
def hand_check(p):
    """The same truss solved by joint equilibrium and bar elongation.

    Vertical equilibrium at the apex: each bar carries P/2 vertically, and the
    bar is inclined at theta to the horizontal, so its axial force is
    N = (P/2)/sin(theta) in compression. The apex settles by the vertical
    component of the two equal bar shortenings, dL/sin(theta).
    """
    half = p["span"] / 2.0
    length = math.hypot(half, p["H"])            # mm, bar length
    sin_t = p["H"] / length

    n_bar = -(p["P"] / 2.0) / sin_t              # N, negative = compression
    dl = n_bar * length / (p["E"] * p["A"])      # mm, bar elongation
    uy = dl / sin_t                              # mm, apex vertical movement

    return {"length": length, "sin_t": sin_t, "N": n_bar, "dL": dl, "uy": uy}
# --8<-- [end:hand_check]


def solve(p=TRUSS):
    """Build, run and read -- the shape of every script from here on."""
    build_model(p)
    run()
    return results(p)


if __name__ == "__main__":
    got = solve()
    want = hand_check(TRUSS)

    print("Two-bar truss, units N-mm-s")
    print(f"  bar length            {want['length']:12.2f} mm")
    print(f"  apex settlement  uy   {got['uy']:12.4f} mm"
          f"   (by hand {want['uy']:.4f})")
    print(f"  apex sway        ux   {got['ux']:12.4e} mm   (symmetry: expect 0)")
    print(f"  bar 1 axial      N1   {got['N1']:12.2f} N "
          f"    (by hand {want['N']:.2f})")
    print(f"  bar 2 axial      N2   {got['N2']:12.2f} N ")
    print()
    print("  Both bars carry the same force because the truss is symmetric,")
    print("  and both are negative because they are in compression.")
