"""Week 8 - a reinforced concrete section, built out of fibres.

    python code/w08/section.py

Weeks 6 and 7 tested materials one at a time. A section is what you get when
you put several of them side by side and insist they all obey the same plane
section: divide the cross-section into fibres, give each one a material and a
position, and the section's moment follows from summing what every fibre
carries about the centroid.

That single idea replaces the whole apparatus of cracked transformed sections
and interaction diagrams. It also introduces the first genuine DISCRETISATION
in this course. Week 4's beam mesh was exact at the nodes however coarse it
was; a fibre section is a finite sum approximating an integral, and too few
fibres gives a wrong answer that no amount of solver care will fix.

The rig is the week 6 material rig, one level up. There it was
    impose a strain      -> read a stress.
Here it is
    impose a curvature   -> read a moment.

Units: N, mm, s. Stress in MPa.
"""

import sys
from pathlib import Path

import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --8<-- [start:params]
COLUMN = {
    "b": 400.0,          # mm     section width
    "h": 600.0,          # mm     section depth
    "cover": 40.0,       # mm     to the face of the bar
    # unconfined cover concrete
    "fc": 30.0,          # MPa    positive magnitude
    "eps_c0": 0.002,
    "fcu": 0.0,          # MPa    cover spalls: nothing left
    "eps_cu": 0.005,
    # confined core: the ties raise both the strength and the usable strain
    "fcc": 38.0,         # MPa
    "eps_cc0": 0.005,
    "fccu": 8.0,         # MPa
    "eps_ccu": 0.020,
    # reinforcement
    "fy": 420.0,         # MPa
    "Es": 200000.0,      # MPa
    "b_steel": 0.01,     # -      strain-hardening ratio
    "bar_area": 314.16,  # mm^2   one 20 mm bar
    "n_bars": 3,         # per face
    "axial": -1200000.0,  # N     1200 kN compression (negative)
}
# --8<-- [end:params]

CORE, COVER, STEEL_MAT, SECTION = 1, 2, 3, 1


# --8<-- [start:section]
def define_section(p, n_fibres=20, gj=None):
    """Build the fibre section: two concretes, one steel, three patches.

    A `patch` fills a rectangle with a grid of fibres of one material. A
    `layer` puts a row of discrete bars along a line. Coordinates are the
    section's own local (y, z); for bending about the strong axis only y
    matters, so the fibres are subdivided in y and left as one strip in z.

    `n_fibres` is the number of concrete fibres across the depth, and it is the
    parameter the convergence study varies.

    `gj` is torsional stiffness, and it is None here because a 2D section has
    nowhere to twist. In three dimensions it becomes compulsory -- a fibre
    section models one axial strain and two curvatures and has no concept of
    torsion at all, so OpenSees refuses to build one without being told.
    Week 11 passes G*J.
    """
    ops.uniaxialMaterial("Concrete01", CORE,
                         -p["fcc"], -p["eps_cc0"], -p["fccu"], -p["eps_ccu"])
    ops.uniaxialMaterial("Concrete01", COVER,
                         -p["fc"], -p["eps_c0"], -p["fcu"], -p["eps_cu"])
    ops.uniaxialMaterial("Steel02", STEEL_MAT,
                         p["fy"], p["Es"], p["b_steel"], 18.0, 0.925, 0.15)

    y_out, z_out = p["h"] / 2.0, p["b"] / 2.0
    y_in, z_in = y_out - p["cover"], z_out - p["cover"]

    if gj is None:
        ops.section("Fiber", SECTION)
    else:
        ops.section("Fiber", SECTION, "-GJ", gj)
    # confined core
    ops.patch("rect", CORE, n_fibres, 1, -y_in, -z_in, y_in, z_in)
    # cover: the two faces that matter in bending, top and bottom
    ops.patch("rect", COVER, 2, 1, y_in, -z_out, y_out, z_out)
    ops.patch("rect", COVER, 2, 1, -y_out, -z_out, -y_in, z_out)
    # reinforcement, one layer near each face
    ops.layer("straight", STEEL_MAT, p["n_bars"], p["bar_area"],
              y_in, z_in, y_in, -z_in)
    ops.layer("straight", STEEL_MAT, p["n_bars"], p["bar_area"],
              -y_in, z_in, -y_in, -z_in)
# --8<-- [end:section]


# --8<-- [start:rig]
def build_rig(p, n_fibres=20):
    """Two coincident nodes joined by a zeroLengthSection element.

    The element has no length, so its deformations ARE the section's: the
    relative rotation between the two nodes is the curvature, and the moment
    the element carries is the section's moment. That is why this rig measures
    a section property rather than a member response.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 0, 1, 0)          # free to extend and to rotate

    define_section(p, n_fibres)
    ops.element("zeroLengthSection", 1, 1, 2, SECTION)
# --8<-- [end:rig]


# --8<-- [start:axial]
def apply_axial(p):
    """Put the column's axial load on first, and hold it there.

    A 'Constant' time series rather than 'Linear': once the axial load is on it
    must stay at its full value while the curvature is increased. A Linear
    series would scale it along with everything else, and the section would be
    bent while its axial load grew -- which is not the test anyone means.
    """
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, p["axial"], 0.0, 0.0)

    ops.system("BandGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.test("NormDispIncr", 1.0e-9, 10)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    if ops.analyze(1) != 0:
        raise RuntimeError("the axial load step did not converge")
    ops.loadConst("-time", 0.0)
# --8<-- [end:axial]


# --8<-- [start:curvature]
def push_curvature(max_curv, n_steps=200):
    """Impose curvature in steps and record the moment at each one.

    DisplacementControl drives a chosen degree of freedom to a chosen value
    instead of applying a chosen force. That is the only way to follow a
    response past its peak: after the section starts to lose strength there is
    no load that produces the next point, but there is always a curvature.

    Week 10 generalises this to whole structures. Here it does exactly what
    setStrain did for a material in week 6, one level up.
    """
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(2, 0.0, 0.0, 1.0)          # a unit moment, to be scaled

    d_curv = max_curv / n_steps
    ops.integrator("DisplacementControl", 2, 3, d_curv)
    ops.analysis("Static")

    curvature, moment = [0.0], [0.0]
    for _ in range(n_steps):
        if ops.analyze(1) != 0:
            break                        # the section has failed; stop cleanly
        curvature.append(ops.nodeDisp(2, 3))
        moment.append(-ops.eleResponse(1, "force")[2])
    return curvature, moment
# --8<-- [end:curvature]


# --8<-- [start:solve]
def moment_curvature(p=COLUMN, n_fibres=20, max_curv=9.0e-5, n_steps=180):
    """Build, apply the axial load, then bend it. Returns (curvature, moment).

    The default curvature stops at 9e-5 for a reason worth knowing about.
    Push this section further and the analysis stops converging at about 1e-4,
    and it is not a step-size problem either: cutting the step by a factor of
    eight buys about a quarter more curvature and no more.

    Do not conclude from that that the section has failed. Week 10 pushes this
    same section, unchanged, to 5e-4 by altering nothing but the algorithm and
    the tolerance. What stops here is the analysis, not the column.
    """
    build_rig(p, n_fibres)
    apply_axial(p)
    return push_curvature(max_curv, n_steps)
# --8<-- [end:solve]


def peak_moment(p=COLUMN, n_fibres=20, **kw):
    """The largest moment the section reaches, in kN m."""
    _, moment = moment_curvature(p, n_fibres, **kw)
    return max(moment) / 1.0e6


def yield_point(curvature, moment, frac=0.95):
    """The curvature at which the moment first reaches `frac` of its peak.

    A section has no single yield point -- the bars yield at different strains
    and the concrete softens gradually -- so any definition is a convention.
    Stating which convention you used is not optional.
    """
    target = frac * max(moment)
    for k, m in zip(curvature, moment):
        if m >= target:
            return k
    return curvature[-1]


if __name__ == "__main__":
    print(f"RC column {COLUMN['b']:.0f} x {COLUMN['h']:.0f} mm,"
          f" axial {COLUMN['axial'] / 1000:.0f} kN")

    N_STEPS = 180
    curv, mom = moment_curvature(COLUMN, n_fibres=20, n_steps=N_STEPS)
    peak = max(mom) / 1.0e6
    print(f"  peak moment      {peak:10.1f} kN m")
    print(f"  at curvature     {curv[mom.index(max(mom))]:10.3e} 1/mm")
    print(f"  95 % curvature   {yield_point(curv, mom):10.3e} 1/mm")
    print(f"  steps completed  {len(curv) - 1:10d} of {N_STEPS}")

    print("\nConvergence with the number of concrete fibres")
    print(f"  {'fibres':>8}{'peak moment [kN m]':>22}{'change':>12}")
    previous = None
    for n in (2, 4, 8, 16, 32, 64):
        m = peak_moment(COLUMN, n_fibres=n)
        change = "" if previous is None else f"{(m - previous) / previous:11.2%}"
        print(f"  {n:8d}{m:22.2f}{change:>12}")
        previous = m

    print("\n  This is a real discretisation, unlike the beam mesh in week 4.")
    print("  Two fibres cannot represent a stress block at all; sixty-four")
    print("  cost sixty-four material evaluations at every iteration.")
