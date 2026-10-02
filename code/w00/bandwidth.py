"""Week 0 - most of a stiffness matrix is zero, and numbering decides where.

    python code/w00/bandwidth.py

A plane frame, ten storeys and three bays, assembled with the same
frame_element() and assemble() as the rest of week 0. The matrix is then
examined rather than solved.

Running it teaches the fact that makes finite element programs fast enough to
use: a degree of freedom is coupled only to the degrees of freedom of the
nodes it shares an element with. However large the structure, each row of K
holds a handful of nonzero numbers, so the storage and the work grow roughly
in proportion to the size of the model, not with its square.

Where those nonzeros sit depends on nothing but the order in which the
degrees of freedom are numbered. Number the frame floor by floor and they
crowd into a narrow band along the diagonal; number it at random and they are
scattered across the whole matrix. The structure is the same, the answer is
the same, and the cost of solving it is not. This is what ops.numberer is for
(week 3).

Units: N, mm, s.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w00.stiffness import assemble, frame_element                # noqa: E402
from w01.environment_check import CANTILEVER                     # noqa: E402

# --8<-- [start:params]
FRAME = {
    "storeys": 10,
    "bays": 3,
    "h": 3000.0,              # mm      storey height
    "b": 6000.0,              # mm      bay width
    "E": CANTILEVER["E"],     # MPa     steel
    "A": CANTILEVER["A"],     # mm^2    IPE 300, for every member
    "I": CANTILEVER["I"],     # mm^4
}
# --8<-- [end:params]


def free_nodes(p):
    """Every node above the base, as (column line, level)."""
    return [(c, lev) for lev in range(1, p["storeys"] + 1)
            for c in range(p["bays"] + 1)]


# --8<-- [start:numbering]
def numbering(p, order="floor", seed=0):
    """Give each free node its three DOF numbers, in one of three orders.

    'floor'  : level by level, left to right -- the order you would write down
    'column' : column line by column line, bottom to top
    'random' : a shuffled order, seeded so that it is repeatable
    """
    nodes = free_nodes(p)
    if order == "column":
        nodes = sorted(nodes)
    elif order == "random":
        nodes = [nodes[i] for i in np.random.default_rng(seed).permutation(len(nodes))]
    return {node: [3*k, 3*k + 1, 3*k + 2] for k, node in enumerate(nodes)}
# --8<-- [end:numbering]


def members(p):
    """Columns then beams, as pairs of (column line, level)."""
    cols = [((c, lev), (c, lev + 1)) for lev in range(p["storeys"])
            for c in range(p["bays"] + 1)]
    beams = [((c, lev), (c + 1, lev)) for lev in range(1, p["storeys"] + 1)
             for c in range(p["bays"])]
    return cols + beams


def free_stiffness(p=FRAME, order="floor", seed=0):
    """K for the free DOFs only. Base nodes are fixed, so their DOFs are simply
    never numbered: a member reaching the base contributes to its top node
    alone."""
    dof = numbering(p, order, seed)
    elements = []
    for (ni, nj) in members(p):
        xi, yi = ni[0] * p["b"], ni[1] * p["h"]
        xj, yj = nj[0] * p["b"], nj[1] * p["h"]
        k = frame_element(xi, yi, xj, yj, p["E"], p["A"], p["I"])
        rows = [(r, d) for r, d in enumerate(dof.get(ni, [None]*3) + dof.get(nj, [None]*3))
                if d is not None]
        keep = [r for r, _ in rows]
        elements.append((k[np.ix_(keep, keep)], [d for _, d in rows]))
    return assemble(3 * len(dof), elements)


# --8<-- [start:pattern]
def pattern(K):
    """How many entries are nonzero, and how far from the diagonal they reach."""
    rows, cols = np.nonzero(np.abs(K) > 1e-9 * np.abs(K).max())
    n = K.shape[0]
    return {"n": n, "entries": n * n, "nonzero": len(rows),
            "fraction": len(rows) / (n * n),
            "per_row": len(rows) / n,
            "half_bandwidth": int(np.max(np.abs(rows - cols)))}
# --8<-- [end:pattern]


if __name__ == "__main__":
    print("Plane frame, 10 storeys x 3 bays, free DOFs only")
    print("   numbering   DOFs   entries   nonzero   fraction   per row   half-band")
    for order in ("floor", "column", "random"):
        s = pattern(free_stiffness(FRAME, order))
        print(f"   {order:9s} {s['n']:5d} {s['entries']:9d} {s['nonzero']:9d}"
              f"   {100*s['fraction']:6.2f} %  {s['per_row']:7.2f}   {s['half_bandwidth']:7d}")
    tall = pattern(free_stiffness(dict(FRAME, storeys=40, bays=10)))
    print()
    print(f"   40 x 10 frame: {tall['n']} DOFs, {tall['per_row']:.2f} nonzero per row,"
          f" {100*tall['fraction']:.2f} % of the matrix")
