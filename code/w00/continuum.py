"""Week 0 - a finite element model where the elements are genuinely approximate.

    python code/w00/continuum.py

A deep cantilever -- a wall, 3 m long and 600 mm deep -- modelled not as a
beam but as a two-dimensional solid, divided into four-node quadrilateral
elements. Inside each element the displacement is only allowed to vary
bilinearly, and a bilinear field cannot bend without also shearing. So, unlike
the beam elements of week 4, these elements do NOT reproduce the exact answer,
and the mesh decides how close the model gets.

Running it teaches the two things every finite element model of a continuum
shares:

* refining the mesh changes the answer, and it converges;
* a coarse model is too STIFF, not too flexible. Every mesh deflects less
  than the finer one after it. An assumed shape can only hold the structure
  back from deforming the way it wants to; it can never let it deform more.

You will not be able to read every line of this script until week 5. It is
here so that the numbers in week 0 come from a real analysis, and so that you
can come back to it once the commands are familiar.

Units: N, mm, s.
"""

import openseespy.opensees as ops

# --8<-- [start:params]
WALL = {
    "L": 3000.0,        # mm      length of the cantilever
    "h": 600.0,         # mm      depth
    "t": 200.0,         # mm      thickness
    "E": 30000.0,       # MPa     concrete
    "nu": 0.2,          # -       Poisson's ratio
    "P": 100000.0,      # N       = 100 kN, downward, spread over the free end
}
MESHES = [(5, 1), (10, 2), (20, 4), (40, 8), (80, 16)]   # (along, through depth)
# --8<-- [end:params]


def node_tag(i, j, ny):
    """Node in column i (along the wall) and row j (through the depth)."""
    return i * (ny + 1) + j + 1


# --8<-- [start:build_model]
def build_model(p, nx, ny):
    """Generate an nx-by-ny grid of quadrilaterals over the wall."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    for i in range(nx + 1):
        for j in range(ny + 1):
            ops.node(node_tag(i, j, ny), i * p["L"] / nx, j * p["h"] / ny)
    for j in range(ny + 1):
        ops.fix(node_tag(0, j, ny), 1, 1)             # the whole end is built in

    ops.nDMaterial("ElasticIsotropic", 1, p["E"], p["nu"])
    for i in range(nx):
        for j in range(ny):
            corners = (node_tag(i, j, ny), node_tag(i + 1, j, ny),
                       node_tag(i + 1, j + 1, ny), node_tag(i, j + 1, ny))
            ops.element("quad", i * ny + j + 1, *corners, p["t"],
                        "PlaneStress", 1)

    # The tip load as a uniform shear over the free end: each interior node
    # takes one share, the two corner nodes half a share each.
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    share = p["P"] / ny
    for j in range(ny + 1):
        f = share / 2 if j in (0, ny) else share
        ops.load(node_tag(nx, j, ny), 0.0, -f)
# --8<-- [end:build_model]


def run():
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


def tip_deflection(nx, ny):
    """Mean vertical movement of the free end, in mm (negative = down)."""
    return sum(ops.nodeDisp(node_tag(nx, j, ny), 2) for j in range(ny + 1)) / (ny + 1)


def solve(p, nx, ny):
    """Build, run, and return the tip deflection plus everything a figure
    needs: node positions and displacements, and the bending stress sxx at
    each element's four integration points."""
    build_model(p, nx, ny)
    run()
    nodes = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            tag = node_tag(i, j, ny)
            nodes[tag] = (ops.nodeCoord(tag), ops.nodeDisp(tag))
    cells = []
    for i in range(nx):
        for j in range(ny):
            tag = i * ny + j + 1
            # (sxx, syy, sxy) at each of four points; point k lies in the
            # quarter of the element nearest its k-th node.
            s = ops.eleResponse(tag, "stresses")
            cells.append((ops.eleNodes(tag), list(s[0::3])))
    return {"tip": tip_deflection(nx, ny), "nodes": nodes, "cells": cells,
            "n_elements": nx * ny}


# --8<-- [start:beam_theory]
def beam_theory(p):
    """Two independent estimates. Euler-Bernoulli ignores shear deformation;
    Timoshenko adds it, with the shear area of a rectangle, 5/6 of the whole."""
    I = p["t"] * p["h"]**3 / 12.0
    A = p["t"] * p["h"]
    G = p["E"] / (2.0 * (1.0 + p["nu"]))
    bending = p["P"] * p["L"]**3 / (3.0 * p["E"] * I)
    shear = p["P"] * p["L"] / (5.0 / 6.0 * G * A)
    return {"euler_bernoulli": bending, "timoshenko": bending + shear,
            "bending": bending, "shear": shear}
# --8<-- [end:beam_theory]


# --8<-- [start:study]
def study(p=WALL, meshes=MESHES):
    """Tip deflection on each mesh, as a fraction of the finest."""
    tips = [abs(solve(p, nx, ny)["tip"]) for nx, ny in meshes]
    return [(nx * ny, tip, tip / tips[-1]) for (nx, ny), tip in zip(meshes, tips)]
# --8<-- [end:study]


if __name__ == "__main__":
    ref = beam_theory(WALL)
    print("Cantilever wall 3000 x 600 x 200 mm, E = 30000 MPa, P = 100 kN")
    print("   elements   tip deflection [mm]   / finest mesh")
    for n, tip, frac in study():
        print(f"   {n:8d}   {tip:14.4f}        {frac:10.4f}")
    print(f"   Euler-Bernoulli beam   {ref['euler_bernoulli']:9.4f}")
    print(f"   Timoshenko beam        {ref['timoshenko']:9.4f}")
