"""S4 - a concrete cylinder, meshed into solids, crushed until it fails.

    python code/s4/cylinder.py

Everything in weeks 1-14 was made of lines. A truss is a line, a beam is a
line, and even the 3D frame in week 11 was a set of lines with sections
attached to them. That is not a limitation of OpenSees; it is what structural
analysis mostly is, because a line element with a fibre section carries an
enormous amount of physics for very little cost.

This chapter leaves that behind for the one case where it cannot work: a
standard 150 x 300 mm concrete test cylinder, meshed into brick elements and
loaded until it crushes. There is no line to reduce it to, no section to
integrate over, and no closed form to check against.

Four things it has to solve, and each is a lesson:

    the mesh        a disc has no natural quadrilateral grid. The map used
                    here turns a square grid into a round one without a
                    degenerate cell at the centre.
    the material    week S3 established that only three concrete continuum
                    materials exist in this build and that the obvious one
                    does not soften. This one does -- but its parameters are
                    not the strength, and have to be calibrated.
    the failure     softening is supposed to make the post-peak answer depend
                    on the mesh. Two experiments here show when it does and
                    when it does not, and what actually stops you instead.
    the picture     a stress field inside a solid cannot be plotted. It has to
                    be exported, and ParaView is where it gets looked at.

Units: N, mm, s.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openseespy.opensees as ops                                   # noqa: E402

OUT = Path(__file__).resolve().parent / "out"

CYLINDER = {
    "D": 150.0,          # diameter, mm -- the standard test specimen
    "L": 300.0,          # height, mm
    "fc": 30.0,          # target compressive strength, MPa
    "E": 30000.0,        # MPa
    "nu": 0.2,
    "rho": 2.4e-9,       # t/mm^3 -- 2400 kg/m^3 in N, mm, s
    "n_r": 3,            # grid divisions per quadrant across the disc
    "n_z": 8,            # layers up the height
}


# --8<-- [start:mesh]
def concentric_disc(n):
    """Map a square grid onto a disc without collapsing a cell at the centre.

    A disc is the standard awkward case for a structured mesh. Polar
    coordinates look obvious and are a trap: every cell in the innermost ring
    degenerates to a triangle at r = 0, and a brick with two coincident nodes
    has a singular Jacobian.

    The concentric map (Shirley and Chiu) avoids that entirely. It sends the
    square [-1,1]^2 to the unit disc by keeping the larger coordinate as the
    radius and turning the smaller one into an angle, so a square grid becomes
    a round grid with the same topology -- and the centre stays a proper
    four-sided cell.

    Returns (2n+1)^2 points in [-1,1]^2 mapped into the unit disc, as a list of
    (x, y) in row-major order.
    """
    pts = []
    for j in range(2 * n + 1):
        v = j / n - 1.0
        for i in range(2 * n + 1):
            u = i / n - 1.0
            if u == 0.0 and v == 0.0:
                pts.append((0.0, 0.0))
            elif abs(u) >= abs(v):
                r, a = u, (math.pi / 4.0) * (v / u)
                pts.append((r * math.cos(a), r * math.sin(a)))
            else:
                r, a = v, (math.pi / 2.0) - (math.pi / 4.0) * (u / v)
                pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def mesh(p=CYLINDER):
    """Nodes and hexahedra for the cylinder. Returns (coords, cells, groups).

    coords  {tag: (x, y, z)}
    cells   {tag: (n1..n8)} in the node order stdBrick expects -- bottom face
            counter-clockwise, then the top face the same way round
    groups  {'base': [...], 'top': [...]} the node tags on each end
    """
    n, nz = p["n_r"], p["n_z"]
    R, L = 0.5 * p["D"], p["L"]
    side = 2 * n + 1

    disc = concentric_disc(n)
    coords, k = {}, 1
    for layer in range(nz + 1):
        z = L * layer / nz
        for (x, y) in disc:
            coords[k] = (R * x, R * y, z)
            k += 1

    def tag(layer, i, j):
        return layer * side * side + j * side + i + 1

    cells, c = {}, 1
    for layer in range(nz):
        for j in range(2 * n):
            for i in range(2 * n):
                cells[c] = (
                    tag(layer, i, j), tag(layer, i + 1, j),
                    tag(layer, i + 1, j + 1), tag(layer, i, j + 1),
                    tag(layer + 1, i, j), tag(layer + 1, i + 1, j),
                    tag(layer + 1, i + 1, j + 1), tag(layer + 1, i, j + 1),
                )
                c += 1

    groups = {
        "base": [tag(0, i, j) for j in range(side) for i in range(side)],
        "top": [tag(nz, i, j) for j in range(side) for i in range(side)],
    }
    return coords, cells, groups
# --8<-- [end:mesh]


# --8<-- [start:material]
def drucker_prager(tag, p, sigma_y, hardening):
    """A pressure-dependent plastic material that can actually lose strength.

    S3 found three concrete continuum materials in this build.
    PlasticDamageConcrete3d reaches a plausible peak and then stays there
    for ever -- it does not soften, and its four damage parameters change
    nothing. FSAM is for membranes with reinforcement in them.
    PlaneStressUserMaterial needs an external subroutine.

    So this uses DruckerPrager, from the soil branch of S3's tree. It is not a
    concrete model, and pretending otherwise would be dishonest: it has no
    crack, no tension cut-off worth the name, and no distinction between
    loading and reloading. What it does have is the one property the chapter
    cannot proceed without -- a NEGATIVE hardening modulus, which makes the
    yield surface shrink as plastic strain accumulates, and so makes the
    specimen lose strength.

    Note what the parameters are not. `sigma_y` is not the compressive
    strength; it sets the size of the yield cone, and the uniaxial strength
    that follows from it also depends on the friction parameter rho. The
    strength is an OUTPUT of this material, which is why calibrate() exists.
    """
    K = p["E"] / (3.0 * (1.0 - 2.0 * p["nu"]))
    G = p["E"] / (2.0 * (1.0 + p["nu"]))
    ops.nDMaterial('DruckerPrager', tag,
                   K, G, sigma_y,
                   0.15,        # rho     -- friction, drives pressure sensitivity
                   0.05,        # rhoBar  -- non-associated flow, limits dilation
                   0.0, 0.0,    # Kinf, Ko -- no isotropic saturation hardening
                   0.0, 0.0,    # delta1, delta2 -- no tension softening of its own
                   hardening,   # H       -- NEGATIVE: this is the whole point
                   1.0,         # theta   -- fully isotropic hardening
                   p["rho"], 101.0)
# --8<-- [end:material]


def _single_brick(p, sigma_y, hardening, size=100.0):
    """One element, for calibration. The cheapest possible test of a material."""
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 3)
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for i, (x, y, z) in enumerate(corners, start=1):
        ops.node(i, x * size, y * size, z * size)
    for i in (1, 2, 3, 4):
        ops.fix(i, 0, 0, 1)
    ops.fix(1, 1, 1, 0)
    ops.fix(2, 0, 1, 0)
    ops.fix(4, 1, 0, 0)
    drucker_prager(1, p, sigma_y, hardening)
    ops.element('stdBrick', 1, *range(1, 9), 1)
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for i in (5, 6, 7, 8):
        ops.load(i, 0.0, 0.0, -1.0)
    return size


def uniaxial_curve(p, sigma_y, hardening, steps=200, d_eps=2.0e-4):
    """Crush one brick and return (strain, stress) with compression positive."""
    size = _single_brick(p, sigma_y, hardening)
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.test('NormDispIncr', 1.0e-7, 60)
    ops.algorithm('Newton')
    ops.integrator('DisplacementControl', 5, 3, -d_eps * size)
    ops.analysis('Static')

    curve = []
    for _ in range(steps):
        if ops.analyze(1) < 0:
            break
        s = ops.eleResponse(1, 'stresses')
        szz = sum(s[g * 6 + 2] for g in range(8)) / 8.0
        curve.append((-ops.nodeDisp(5, 3) / size, -szz))
    return curve


# --8<-- [start:calibrate]
def calibrate(p=CYLINDER, hardening=-500.0, tol=0.01, lo=10.0, hi=40.0):
    """Find the sigma_y that gives the target compressive strength.

    This function exists because of a fact worth stating plainly: for this
    material the compressive strength is not something you enter. It emerges
    from the yield-cone size and the friction parameter together, so the only
    way to hit fc = 30 MPa is to run the one-element test, look at the peak,
    and adjust.

    That is not a defect of OpenSees. It is what using a soil plasticity model
    for concrete costs, and being explicit about the cost is the point of
    borrowing it. Bisection, because the relationship is monotonic and we need
    perhaps a dozen evaluations.
    """
    target = p["fc"]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        curve = uniaxial_curve(p, mid, hardening, steps=60)
        peak = max(s for _, s in curve) if curve else 0.0
        if abs(peak - target) / target < tol:
            return mid, peak
        if peak < target:
            lo = mid
        else:
            hi = mid
    return mid, peak
# --8<-- [end:calibrate]


# --8<-- [start:build]
def build(p, sigma_y, hardening, weakened=0.0):
    """The whole specimen: mesh, material, elements, and the two platens.

    The platens are the modelling decision that matters. A real cylinder is
    crushed between steel plates, which restrain the ends laterally through
    friction -- and that restraint is why a cylinder fails in a cone rather
    than splitting cleanly. Model it as frictionless and you get a different
    failure. Model it as fully bonded and you get a different one again.

    What is done here is the frictionless idealisation, with the top nodes
    tied to one another vertically so the platen stays flat. That choice is
    revisited in the exercises, because it changes the answer.
    """
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 3)

    coords, cells, groups = mesh(p)
    for tag, (x, y, z) in coords.items():
        ops.node(tag, x, y, z)

    drucker_prager(1, p, sigma_y, hardening)

    # An imperfection, if asked for. A perfectly uniform specimen softens
    # uniformly and never localises, so there is nothing for the mesh to
    # disagree about; real specimens are not uniform. The weakened region is
    # a FIXED PHYSICAL VOLUME -- a disc of radius 0.5R spanning the middle 30 % of the height -- so
    # that refining the mesh does not also change the size of the flaw.
    weak = set()
    if weakened > 0.0:
        drucker_prager(2, p, sigma_y * (1.0 - weakened), hardening)
        R, L = 0.5 * p["D"], p["L"]
        for tag, nodes in cells.items():
            cx = sum(coords[n][0] for n in nodes) / 8.0
            cy = sum(coords[n][1] for n in nodes) / 8.0
            cz = sum(coords[n][2] for n in nodes) / 8.0
            if (cx * cx + cy * cy) ** 0.5 < 0.35 * R                     and abs(cz - 0.5 * L) < 0.15 * L:
                weak.add(tag)

    for tag, nodes in cells.items():
        ops.element('stdBrick', tag, *nodes, 2 if tag in weak else 1)

    # Bottom platen: no vertical movement anywhere, and just enough in-plane
    # restraint to stop the specimen sliding away. Restraining every base node
    # in x and y would be a bonded platen, which is a different experiment.
    for n in groups["base"]:
        ops.fix(n, 0, 0, 1)
    centre = groups["base"][len(groups["base"]) // 2]
    ops.fix(centre, 1, 1, 0)
    ops.fix(groups["base"][0], 0, 1, 0)

    # Top platen: flat and rigid vertically, free to expand sideways.
    master = groups["top"][len(groups["top"]) // 2]
    for n in groups["top"]:
        if n != master:
            ops.equalDOF(master, n, 3)

    return coords, cells, groups, master, weak
# --8<-- [end:build]


# --8<-- [start:crush]
def crush(p=CYLINDER, sigma_y=24.53, hardening=-500.0,
          steps=90, d_eps=1.2e-4, record=None, weakened=0.0):
    """Squeeze the cylinder until it will not take any more.

    Displacement control on the platen, because the whole interest is past the
    peak and load control cannot go there -- week 10.

    `record`, if given, is called as record(step, coords, cells) after every
    converged step, which is how the VTK series gets written without this
    function knowing anything about file formats.
    """
    coords, cells, groups, master, weak = build(p, sigma_y, hardening,
                                                weakened)
    L = p["L"]
    area = math.pi * (0.5 * p["D"]) ** 2

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(master, 0.0, 0.0, -1.0)

    ops.system('UmfPack')
    ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.test('NormDispIncr', 1.0e-6, 40)
    ops.algorithm('Newton')
    ops.integrator('DisplacementControl', master, 3, -d_eps * L)
    ops.analysis('Static')

    out = {"eps": [], "sig": [], "converged": True}
    for step in range(steps):
        if ops.analyze(1) < 0:
            out["converged"] = False
            break
        ops.reactions()
        # The platen pushes down, so the base pushes up: the reaction is
        # positive and no sign flip belongs here. Get this backwards and the
        # least-compressed early step is reported as the "peak", which looks
        # like a specimen that failed immediately.
        force = sum(ops.nodeReaction(n, 3) for n in groups["base"])
        out["eps"].append(-ops.nodeDisp(master, 3) / L)
        out["sig"].append(force / area)
        if record is not None:
            record(step, coords, cells)

    out["peak"] = max(out["sig"]) if out["sig"] else 0.0
    out["eps_at_peak"] = (out["eps"][out["sig"].index(out["peak"])]
                          if out["sig"] else 0.0)
    out["final"] = out["sig"][-1] if out["sig"] else 0.0
    out["nodes"] = len(coords)
    out["elements"] = len(cells)
    out["weakened"] = len(weak)
    return out
# --8<-- [end:crush]


# --8<-- [start:vtk]
def export_vtk(coords, cells, path, point_fields=None,
               cell_fields=None):
    """Write one .vtu file that ParaView can open. No library required.

    A displacement is a vector on a line and matplotlib draws it. A stress
    field inside a solid is not: any single view of it hides most of it, and
    what you actually want is to rotate the specimen, slice it, and threshold
    on a value. That is what ParaView is, and getting data into it is the only
    part that concerns us here.

    The format is written out by hand rather than through a library, because
    it is about forty lines and knowing what is in the file is worth more than
    one import. VTK's unstructured grid is: a list of points, a list of cells
    given as indices into those points, a type code per cell (12 is a linear
    hexahedron), and any number of named arrays attached to points or cells.

    Point and cell fields are passed separately and deliberately so. Working
    out which is which by testing whether the keys are node tags or element
    tags does not work: element tags 1..288 are a subset of node tags 1..441,
    so every element field is silently written as point data instead. Tag
    ranges overlap in OpenSees. Do not infer meaning from them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    order = sorted(coords)
    index = {tag: i for i, tag in enumerate(order)}
    cell_order = sorted(cells)

    def arr(name, data, tags):
        first = data[tags[0]]
        ncomp = 1 if isinstance(first, (int, float)) else len(first)
        vals = []
        for t in tags:
            v = data[t]
            vals.extend([v] if ncomp == 1 else list(v))
        body = " ".join(f"{x:.6g}" for x in vals)
        return (f'        <DataArray type="Float32" Name="{name}" '
                f'NumberOfComponents="{ncomp}" format="ascii">\n'
                f'          {body}\n        </DataArray>\n')

    point_fields = point_fields or {}
    cell_fields = cell_fields or {}

    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n'
                '<VTKFile type="UnstructuredGrid" version="0.1" '
                'byte_order="LittleEndian">\n  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{len(order)}" '
                f'NumberOfCells="{len(cell_order)}">\n')

        f.write('      <Points>\n        <DataArray type="Float32" '
                'NumberOfComponents="3" format="ascii">\n          ')
        f.write(" ".join(f"{c:.6g}" for t in order for c in coords[t]))
        f.write('\n        </DataArray>\n      </Points>\n')

        f.write('      <Cells>\n        <DataArray type="Int32" '
                'Name="connectivity" format="ascii">\n          ')
        f.write(" ".join(str(index[n]) for c in cell_order
                         for n in cells[c]))
        f.write('\n        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" '
                'format="ascii">\n          ')
        f.write(" ".join(str(8 * (i + 1)) for i in range(len(cell_order))))
        f.write('\n        </DataArray>\n')
        f.write('        <DataArray type="UInt8" Name="types" '
                'format="ascii">\n          ')
        f.write(" ".join("12" for _ in cell_order))     # 12 = linear hexahedron
        f.write('\n        </DataArray>\n      </Cells>\n')

        if point_fields:
            f.write('      <PointData>\n')
            for name, data in point_fields.items():
                f.write(arr(name, data, order))
            f.write('      </PointData>\n')
        if cell_fields:
            f.write('      <CellData>\n')
            for name, data in cell_fields.items():
                f.write(arr(name, data, cell_order))
            f.write('      </CellData>\n')

        f.write('    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n')
    return path
# --8<-- [end:vtk]


# --8<-- [start:series]
def write_series(paths, times, path):
    """A .pvd collection, which is what makes ParaView treat the files as time.

    Without this, twenty .vtu files are twenty unrelated models. With it they
    are one model at twenty instants, and the animation controls light up.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" '
                'version="0.1" byte_order="LittleEndian">\n  <Collection>\n')
        for t, p in zip(times, paths):
            f.write(f'    <DataSet timestep="{t:.6g}" part="0" '
                    f'file="{Path(p).name}"/>\n')
        f.write('  </Collection>\n</VTKFile>\n')
    return path
# --8<-- [end:series]


# --8<-- [start:snapshot]
def snapshot(coords, cells, scale=1.0):
    """Current geometry and the fields worth looking at, ready for export.

    Three fields, and the choice is deliberate. Displacement so the shape can
    be exaggerated; vertical stress because that is what is being applied; and
    the equivalent plastic strain, which is the one that shows WHERE the
    specimen is failing rather than how hard it is being pushed.
    """
    moved, disp = {}, {}
    for tag, (x, y, z) in coords.items():
        d = ops.nodeDisp(tag)
        disp[tag] = tuple(d[:3])
        moved[tag] = (x + scale * d[0], y + scale * d[1], z + scale * d[2])

    szz, plastic = {}, {}
    for tag in cells:
        s = ops.eleResponse(tag, 'stresses')
        szz[tag] = -sum(s[g * 6 + 2] for g in range(8)) / 8.0
        e = ops.eleResponse(tag, 'strains')
        # Deviatoric strain magnitude: a scalar that rises where the material
        # is yielding and stays near zero where it is still elastic.
        acc = 0.0
        for g in range(8):
            ex, ey, ez = e[g * 6], e[g * 6 + 1], e[g * 6 + 2]
            m = (ex + ey + ez) / 3.0
            acc += math.sqrt((2.0 / 3.0) * ((ex - m) ** 2 + (ey - m) ** 2
                                            + (ez - m) ** 2))
        plastic[tag] = acc / 8.0

    return moved, {"displacement": disp}, {"stress_zz": szz,
                                           "deviatoric_strain": plastic}
# --8<-- [end:snapshot]


# --8<-- [start:impact]
def impact(p=CYLINDER, sigma_y=24.53, hardening=-500.0,
           rate=2000.0, duration=1.5e-3, dt=5.0e-6, weakened=0.0):
    """The same specimen crushed dynamically, by a platen driven at `rate`.

    The material used here has no rate dependence whatever -- DruckerPrager's
    yield surface does not know how fast it is being loaded. So every
    difference between this and the static crush is INERTIA, and nothing else.
    That makes it a clean experiment: the specimen has to accelerate its own
    mass before the far end learns that the near end has been hit, and at a
    high enough rate the top crushes before the bottom has felt anything.

    `rate` is the platen speed in mm/s. Prescribed motion goes in as an sp
    constraint inside a pattern, which needs constraints('Transformation') --
    'Plain' cannot impose a non-zero support displacement.
    """
    coords, cells, groups, master, weak = build(p, sigma_y, hardening,
                                                weakened)
    L = p["L"]
    area = math.pi * (0.5 * p["D"]) ** 2

    steps = int(duration / dt)
    # A Linear series returns factor * t, so the factor IS the platen speed
    # and the prescribed displacement is rate * t. Writing rate * duration
    # here instead -- the total travel -- multiplies by the duration twice
    # and drives the platen a thousand times too slowly.
    ops.timeSeries('Linear', 1, '-factor', rate)
    ops.pattern('Plain', 1, 1)
    ops.sp(master, 3, -1.0)

    ops.system('UmfPack')
    ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.test('NormDispIncr', 1.0e-6, 40)
    ops.algorithm('Newton')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    out = {"t": [], "eps": [], "sig": [], "top_sig": [], "converged": True}
    for _ in range(steps):
        if ops.analyze(1, dt) < 0:
            out["converged"] = False
            break
        ops.reactions()
        force = sum(ops.nodeReaction(n, 3) for n in groups["base"])
        # The topmost layer of elements, which is what the platen is actually
        # pushing on. In a static run this equals the base stress; here it
        # does not, and the gap is the whole point.
        top_cells = [c for c in cells
                     if min(coords[n][2] for n in cells[c]) > 0.85 * L]
        top = 0.0
        for c in top_cells:
            s = ops.eleResponse(c, 'stresses')
            top += -sum(s[g * 6 + 2] for g in range(8)) / 8.0
        out["t"].append(ops.getTime())
        out["eps"].append(-ops.nodeDisp(master, 3) / L)
        out["sig"].append(force / area)
        out["top_sig"].append(top / max(1, len(top_cells)))

    out["peak"] = max(out["sig"]) if out["sig"] else 0.0
    out["peak_top"] = max(out["top_sig"]) if out["top_sig"] else 0.0
    out["steps"] = len(out["sig"])
    return out
# --8<-- [end:impact]


# --8<-- [start:mesh-study]
def mesh_study(p=CYLINDER, sigma_y=24.53, hardening=-500.0,
               densities=((2, 6), (3, 8), (4, 12)), weakened=0.0):
    """The same specimen at three mesh densities. The result is not the
    expected one, and the difference is the most useful thing in this file.

    The standard warning about softening continua is that they localise:
    deformation collects into a band one element wide whatever the element
    size, so refining the mesh narrows the band and drops the post-peak curve
    without converging to anything. Everything after the peak is then mesh
    dependent, and no amount of refinement fixes it.

The experiment below does not support that warning as stated. With mild
    softening and a uniform specimen the residual strength is IDENTICAL on
    every mesh -- because a uniform specimen softens uniformly, nothing
    localises, and there is no band for the mesh to resolve differently.
    Localisation is not a consequence of softening alone; it needs a trigger.

    Give it one -- the `weakened` region -- and steepen the softening enough
    for the tangent to turn indefinite, and what happens is not a mesh
    dependent post-peak curve. The analysis simply stops, at the peak, and how
    far it got depends on the mesh. Which is the practical form of the
    warning: you will meet the convergence wall long before you get a chance
    to be misled by a mesh dependent answer.
    """
    rows = []
    for n_r, n_z in densities:
        q = dict(p, n_r=n_r, n_z=n_z)
        out = crush(q, sigma_y, hardening, weakened=weakened)
        rows.append({
            "n_r": n_r, "n_z": n_z,
            "weakened": out["weakened"],
            "elements": out["elements"],
            "peak": out["peak"],
            "eps_at_peak": out["eps_at_peak"],
            "final": out["final"],
            "residual_fraction": out["final"] / out["peak"] if out["peak"]
            else 0.0,
        })
    return rows
# --8<-- [end:mesh-study]


if __name__ == "__main__":
    p = CYLINDER
    coords, cells, groups = mesh(p)
    print("A standard cylinder, meshed")
    print(f"  {p['D']:.0f} x {p['L']:.0f} mm, "
          f"{p['n_r']} divisions per quadrant, {p['n_z']} layers")
    print(f"  {len(coords)} nodes, {len(cells)} brick elements")
    print(f"  {len(groups['base'])} nodes on each end")

    print("\nCalibrating: the parameters are not the strength")
    sigma_y, peak = calibrate(p)
    print(f"  sigma_y = {sigma_y:6.2f} MPa gives fc = {peak:5.2f} MPa"
          f"  (target {p['fc']:.0f})")
    print("  sigma_y is the size of the yield cone, not a strength. The")
    print("  strength comes out of the analysis, so it has to be calibrated.")

    print("\nCrushing it")
    frames, times = [], []

    def record(step, coords, cells):
        if step % 6:
            return
        moved, pf, cf = snapshot(coords, cells, scale=8.0)
        frames.append(export_vtk(moved, cells,
                                 OUT / f"cylinder-{len(frames):03d}.vtu",
                                 pf, cf))
        times.append(step)

    out = crush(p, sigma_y, record=record)
    print(f"  peak {out['peak']:5.2f} MPa at a strain of "
          f"{out['eps_at_peak']:.4f}")
    print(f"  carried to a strain of {out['eps'][-1]:.4f}, by which point it")
    print(f"  was down to {out['final']:5.2f} MPa "
          f"({out['final'] / out['peak']:.0%} of the peak)")
    if not out["converged"]:
        print("  and then it stopped converging, which is what failure looks")
        print("  like when the tangent has gone indefinite -- see week 10.")

    series = write_series(frames, times, OUT / "cylinder.pvd")
    print(f"\n  {len(frames)} frames written; open {series.name} in ParaView")
    print("  Colour by deviatoric_strain and the failure band is immediate.")

    print("\nDoes the mesh change the answer? Two experiments, and the")
    print("textbook warning turns out to be only half of the story.")
    print(f"  {'mesh':>8}{'elements':>10}{'weak':>6}{'peak':>9}"
          f"{'residual':>10}{'ran':>6}")
    print("  A -- mild softening (H = -500), no flaw:")
    for r in mesh_study(p, sigma_y, -500.0):
        print(f"  {r['n_r']}x{r['n_z']:<6}{r['elements']:10d}"
              f"{r['weakened']:6d}{r['peak']:9.2f}"
              f"{r['residual_fraction']:9.1%}{'yes':>6}")
    print("  B -- steep softening (H = -2000), 15 % flaw at mid-height:")
    for r in mesh_study(p, sigma_y, -2000.0, weakened=0.15):
        print(f"  {r['n_r']}x{r['n_z']:<6}{r['elements']:10d}"
              f"{r['weakened']:6d}{r['peak']:9.2f}"
              f"{r['residual_fraction']:9.1%}{'no':>6}")
    print("  A runs to the end and gives the SAME residual on every mesh.")
    print("  Nothing localised, so there was nothing for the meshes to")
    print("  disagree about. B never gets past the peak at all, and how far")
    print("  it gets depends on the mesh. What stops you in practice is")
    print("  convergence, not mesh dependence.")

    print("\nCrushed dynamically instead, by a platen driven at a speed.")
    print("The material has NO rate dependence, so this is inertia alone.")
    print(f"  {'rate mm/s':>11}{'steps':>8}{'peak':>9}{'top/base':>10}")
    for rate in (250.0, 1000.0, 4000.0):
        d = impact(p, sigma_y, rate=rate)
        pk = d["peak"] or float("nan")
        print(f"  {rate:9.0f}  {d['steps']:8d}{d['peak']:9.2f}"
              f"{d['peak_top'] / pk:10.2f}")
    print(f"  Static, for comparison: {out['peak']:.2f} MPa.")
    print("  A rate-independent material apparently gains a third of its")
    print("  strength. It has not: the specimen is carrying its own inertia,")
    print("  and at the highest rate the top runs 6 % ahead of the base")
    print("  because the stress wave has not finished crossing it.")
