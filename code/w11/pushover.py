"""Week 11 - pushing a three-dimensional frame until it gives way.

    python code/w11/pushover.py

Everything in the course so far arrives here at once. Week 5 generated a frame
from its dimensions; week 8 built a fibre section and drove it with
displacement control; week 10 wrote a step that refuses to give up. A pushover
is those three things pointed at the same model.

Three-dimensional modelling brings two genuinely new obligations, and both of
them fail loudly rather than quietly, which is a mercy:

    ndf = 6         every node now has three rotations as well as three
                    translations, so a geomTransf needs a vector telling it
                    which way is up for the member;
    torsion         a fibre section has NO torsional stiffness - it only knows
                    about axial strain and two curvatures - so a 3D frame made
                    of bare fibre sections has a singular stiffness matrix.

The second one catches almost everybody once.

Units: N, mm, s.
"""

import math
import sys
from pathlib import Path

import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w08.section import COLUMN, define_section                        # noqa: E402
from w10.robust import Effort, robust_step                            # noqa: E402


# --8<-- [start:params]
FRAME = {
    "n_storeys": 3,
    "bay_x": 6000.0,      # mm
    "bay_y": 5000.0,      # mm
    "storey": 3500.0,     # mm
    # beams stay elastic: see the note in build_model
    "E": 200000.0,        # MPa
    "G": 11500.0,         # MPa   concrete shear modulus
    "A_beam": 1.8e5,      # mm^2  300 x 600
    "I_beam": 5.4e9,      # mm^4
    "J_col": 8.6e9,       # mm^4  torsion constant of the 400 x 600 column
    "mass_per_floor": 90000.0,   # N per node, gravity load
    "n_fibres": 12,       # fewer than week 8: this runs 12 columns at a time
}

COL_TRANSF, BEAM_X_TRANSF, BEAM_Y_TRANSF = 1, 2, 3
FIBRE_SEC = 1                 # week 8 builds section tag 1
COL_INTEG = 1                 # the beam integration scheme for the columns

# --8<-- [end:params]


# --8<-- [start:tags]
def node_tag(ix, iy, storey):
    """Column line (ix, iy) at a given floor. 1000s digit is the storey."""
    return 1000 * storey + 10 * ix + iy


def column_tag(ix, iy, storey):
    return 20000 + 1000 * storey + 10 * ix + iy


def beam_tag(direction, ix, iy, storey):
    """direction 0 = spanning in x, 1 = spanning in y."""
    return 30000 + 10000 * direction + 1000 * storey + 10 * ix + iy
# --8<-- [end:tags]


# --8<-- [start:section3d]
def define_column_section(p):
    """The week 8 fibre section, made usable in three dimensions.

    Exactly the same call as week 8, with one extra argument. A fibre section
    models one axial strain and two curvatures; it has no concept of twist, so
    its torsional stiffness is precisely zero. A 3D frame assembled from such
    sections has a singular stiffness matrix.

    OpenSees will not let you find that out the hard way. Omit -GJ in three
    dimensions and the section is refused outright:

        WARNING torsion not specified for FiberSection
        Use either -GJ $GJ or -torsion $matTag

    which is the most useful error message in this course, because it names
    both the problem and the two ways out of it.

    The elastic GJ is a patch rather than a model: the torsion stays linear
    for ever and never interacts with the bending. For a building frame that
    is entirely reasonable, and it is worth knowing that it is a choice.
    """
    define_section(COLUMN, p["n_fibres"], gj=p["G"] * p["J_col"])
# --8<-- [end:section3d]


# --8<-- [start:build_model]
def build_model(p):
    """A one-bay by one-bay, n-storey reinforced concrete frame.

    The beams are elasticBeamColumn and stay elastic, so all the yielding
    happens in the columns. That is not how a frame should be designed --
    strong-column weak-beam is the whole point of capacity design -- and it is
    exactly why this model is worth running: it shows what the mechanism looks
    like when the rule is broken. Exercise 11.3 puts it right.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    lines = ((0, 0), (1, 0), (0, 1), (1, 1))
    for storey in range(p["n_storeys"] + 1):
        for ix, iy in lines:
            ops.node(node_tag(ix, iy, storey),
                     ix * p["bay_x"], iy * p["bay_y"], storey * p["storey"])

    for ix, iy in lines:
        ops.fix(node_tag(ix, iy, 0), 1, 1, 1, 1, 1, 1)

    # A transformation needs a vector that is NOT parallel to the member.
    # For a vertical column that means anything horizontal; for a horizontal
    # beam, anything vertical. Getting this wrong is an immediate error rather
    # than a wrong answer, which is the one kindness 3D offers.
    ops.geomTransf("PDelta", COL_TRANSF, 1.0, 0.0, 0.0)
    ops.geomTransf("Linear", BEAM_X_TRANSF, 0.0, 0.0, 1.0)
    ops.geomTransf("Linear", BEAM_Y_TRANSF, 0.0, 0.0, 1.0)

    define_column_section(p)

    # How many points along the member the section response is sampled at, and
    # where they sit. Five Lobatto points put one at each end, which is where
    # the moment peaks and therefore where the plasticity will be -- and the
    # weight given to that end point acts as an implicit plastic hinge length.
    # This is a modelling decision disguised as a number, as week 8's last
    # exercise warned.
    ops.beamIntegration("Lobatto", COL_INTEG, FIBRE_SEC, 5)

    for storey in range(1, p["n_storeys"] + 1):
        for ix, iy in lines:
            ops.element("forceBeamColumn", column_tag(ix, iy, storey),
                        node_tag(ix, iy, storey - 1), node_tag(ix, iy, storey),
                        COL_TRANSF, COL_INTEG)

    # Elastic beams, both directions.
    for storey in range(1, p["n_storeys"] + 1):
        for iy in (0, 1):
            ops.element("elasticBeamColumn", beam_tag(0, 0, iy, storey),
                        node_tag(0, iy, storey), node_tag(1, iy, storey),
                        p["A_beam"], p["E"], p["G"], p["J_col"],
                        p["I_beam"], p["I_beam"], BEAM_X_TRANSF)
        for ix in (0, 1):
            ops.element("elasticBeamColumn", beam_tag(1, ix, 0, storey),
                        node_tag(ix, 0, storey), node_tag(ix, 1, storey),
                        p["A_beam"], p["E"], p["G"], p["J_col"],
                        p["I_beam"], p["I_beam"], BEAM_Y_TRANSF)
# --8<-- [end:build_model]


# --8<-- [start:gravity]
def apply_gravity(p):
    """Put the building's own weight on, then freeze it.

    loadConst('-time', 0.0) is the important line. It tells OpenSees to keep
    the gravity loads at their current value and reset pseudo-time to zero, so
    the pushover that follows starts from a loaded, already-deformed structure
    rather than from an unloaded one. Forget it and you push a weightless
    frame.
    """
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    for storey in range(1, p["n_storeys"] + 1):
        for ix, iy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            ops.load(node_tag(ix, iy, storey),
                     0.0, 0.0, -p["mass_per_floor"], 0.0, 0.0, 0.0)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1.0e-8, 20)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    if ops.analyze(10) != 0:
        raise RuntimeError("the gravity analysis did not converge")
    ops.loadConst("-time", 0.0)
# --8<-- [end:gravity]


# --8<-- [start:lateral]
def apply_lateral_pattern(p):
    """An inverted triangular load pattern, growing with height.

    The shape is a stand-in for the first mode. Its absolute size does not
    matter -- displacement control scales whatever pattern it is given -- but
    its SHAPE decides which storey is pushed hardest, and therefore where the
    mechanism forms. A pushover answers the question you asked it, and the
    pattern is most of the question.
    """
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    total = sum(range(1, p["n_storeys"] + 1))
    for storey in range(1, p["n_storeys"] + 1):
        share = storey / total
        for ix, iy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            ops.load(node_tag(ix, iy, storey),
                     share / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0)
# --8<-- [end:lateral]


# --8<-- [start:push]
def pushover(p, target, n_steps=120):
    """Push the roof to `target` in x, recording the capacity curve.

    The step is week 10's robust_step, unchanged. A pushover is exactly where
    it earns its keep: the interesting part of a capacity curve is the part
    after the structure has started to fail, which is also the part a plain
    Newton loop refuses to enter.
    """
    control = node_tag(0, 0, p["n_storeys"])
    increment = target / n_steps
    effort = Effort()

    roof, shear, history = [0.0], [0.0], [[0.0] * p["n_storeys"]]
    for _ in range(n_steps):
        if not robust_step(control, 1, increment, effort):
            break
        ops.reactions()
        base = sum(ops.nodeReaction(node_tag(ix, iy, 0), 1)
                   for ix, iy in ((0, 0), (1, 0), (0, 1), (1, 1)))
        roof.append(ops.nodeDisp(control, 1))
        shear.append(-base)
        # The drift at every step, not only at the end. A snapshot of the
        # final state cannot show WHEN a mechanism formed, and that is most
        # of what a pushover has to say.
        history.append(drift_ratios(p))
    return roof, shear, effort, history
# --8<-- [end:push]


# --8<-- [start:solve]
def capacity_curve(p=FRAME, target=250.0, n_steps=120):
    """Build, load, push.

    Returns roof displacement [mm], base shear [N], the effort counter, and
    the storey-drift history -- one row of drift ratios per step.
    """
    build_model(p)
    apply_gravity(p)
    apply_lateral_pattern(p)
    return pushover(p, target, n_steps)
# --8<-- [end:solve]


# --8<-- [start:demand]
def base_section_demand(p):
    """Curvature and extreme-fibre strains at the base of a corner column.

    Week 10's lesson, applied to a structure: the analysis finishing tells you
    nothing about whether the answer is inside the range the materials were
    calibrated for. A capacity curve is worth exactly as much as the strains
    underneath it.

    eleResponse(tag, 'section', 1, 'deformation') asks integration point 1 --
    the base of the column -- for its axial strain and two curvatures.
    """
    deformation = ops.eleResponse(column_tag(0, 0, 1), "section", 1,
                                  "deformation")
    # For a 3D fibre section the vector is [axial, kappa_z, kappa_y, twist].
    # Taking the RESULTANT of the two curvatures rather than one component is
    # not fussiness: in three dimensions a column can bend about both axes at
    # once, and reading only the one you pushed along understates the strain.
    axial = deformation[0]
    curvature = math.hypot(deformation[1], deformation[2])
    y = COLUMN["h"] / 2.0
    return {
        "curvature": curvature,
        "concrete": axial - curvature * y,
        "steel": axial + curvature * (y - COLUMN["cover"]),
    }
# --8<-- [end:demand]


def drift_ratios(p):
    """Inter-storey drift at the end of the push, storey by storey."""
    out = []
    for storey in range(1, p["n_storeys"] + 1):
        above = ops.nodeDisp(node_tag(0, 0, storey), 1)
        below = ops.nodeDisp(node_tag(0, 0, storey - 1), 1)
        out.append((above - below) / p["storey"])
    return out


if __name__ == "__main__":
    print(f"3D frame: {FRAME['n_storeys']} storeys, one bay each way,"
          f" {FRAME['n_fibres']} fibres per column section")

    roof, shear, effort, history = capacity_curve(FRAME, target=250.0, n_steps=120)
    peak = max(shear)
    i_peak = shear.index(peak)

    print(f"\n  roof pushed to        {roof[-1]:8.1f} mm")
    print(f"  peak base shear       {peak / 1000:8.1f} kN"
          f"   at {roof[i_peak]:.1f} mm")
    print(f"  shear at the end      {shear[-1] / 1000:8.1f} kN")
    print(f"  effort: {effort}")

    print("\n  Inter-storey drift at the end of the push")
    for i, d in enumerate(drift_ratios(FRAME), start=1):
        print(f"    storey {i}: {d * 100:6.3f} %")

    print("\n  The drifts are not equal. Where they concentrate is the")
    print("  mechanism, and it is the answer a pushover exists to give.")

    demand = base_section_demand(FRAME)
    print("\n  At the base of a corner column, at the end of the push")
    print(f"    curvature            {demand['curvature']:10.3e} 1/mm")
    print(f"    concrete strain      {demand['concrete']:10.4f}"
          f"   (crushes at {-COLUMN['eps_ccu']:.3f})")
    print(f"    steel strain         {demand['steel']:10.4f}")
    print("\n  Week 10's question, asked of a building: how much of this")
    print("  capacity curve is inside the range the materials describe?")
