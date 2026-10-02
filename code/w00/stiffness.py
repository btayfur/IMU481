"""Week 0 - the direct stiffness method, in a few dozen lines of numpy.

    python code/w00/stiffness.py

This is what OpenSees does when you call analyze() on a linear model, written
out so that nothing is hidden: one stiffness matrix per element, all of them
added into one matrix for the structure, the supports applied by setting aside
the rows and columns they hold, one call to a linear solver, and the member
forces recovered from the displacements.

It solves two structures you will meet again: the two-bar truss of week 2 and
the cantilever of week 1, loaded uniformly. When you reach week 2, compare the
numbers this script prints with the ones OpenSees prints. They agree to every
digit shown, and seeing that is the point of running it.

Nothing here knows what kind of element it is assembling. assemble() takes a
list of matrices and the degrees of freedom each one belongs to, and that is
all. A truss bar, a beam and a plate element are the same thing to it.

Sign convention: global x to the right, y UPWARD, rotation anticlockwise --
the same as OpenSees. A downward load is negative.

Units: N, mm, s.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w01.environment_check import CANTILEVER                     # noqa: E402
from w02.truss import TRUSS                                      # noqa: E402
from w00.ritz import BEAM                                        # noqa: E402


# --8<-- [start:truss_element]
def truss_element(xi, yi, xj, yj, E, A):
    """Stiffness of a pin-jointed bar, in GLOBAL axes: a 4x4 matrix acting on
    (ux_i, uy_i, ux_j, uy_j).

    Along its own axis a bar is a spring of stiffness EA/L. The cosines c and s
    turn that one spring into the four global directions.
    """
    L = math.hypot(xj - xi, yj - yi)
    c, s = (xj - xi) / L, (yj - yi) / L
    t = np.array([[c*c, c*s], [c*s, s*s]])
    return (E * A / L) * np.block([[t, -t], [-t, t]])
# --8<-- [end:truss_element]


# --8<-- [start:beam_element]
def beam_element(E, I, L):
    """Stiffness of a prismatic beam lying along x: a 4x4 matrix acting on
    (v_i, theta_i, v_j, theta_j).

    Column k is the set of end forces and moments that holds the beam in the
    shape where DOF k equals one and the other three are zero.
    """
    return (E * I / L**3) * np.array([
        [ 12.0,    6*L,  -12.0,    6*L],
        [  6*L, 4*L*L,   -6*L,  2*L*L],
        [-12.0,   -6*L,   12.0,   -6*L],
        [  6*L, 2*L*L,   -6*L,  4*L*L],
    ])
# --8<-- [end:beam_element]


def frame_element(xi, yi, xj, yj, E, A, I):
    """A 2D frame member: a bar and a beam in one, rotated into global axes.
    A 6x6 matrix acting on (ux_i, uy_i, rz_i, ux_j, uy_j, rz_j).

    This is what a geomTransf does for you in week 4: the element is written
    in its own axes, and T rotates it into the axes the nodes live in.
    """
    L = math.hypot(xj - xi, yj - yi)
    c, s = (xj - xi) / L, (yj - yi) / L
    k = np.zeros((6, 6))
    a = E * A / L
    k[np.ix_([0, 3], [0, 3])] = a * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = beam_element(E, I, L)
    r = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.block([[r, np.zeros((3, 3))], [np.zeros((3, 3)), r]])
    return T.T @ k @ T


# --8<-- [start:assemble]
def assemble(n_dof, elements):
    """Add every element matrix into the structure's matrix.

    `elements` is a list of (k, dofs): an element's matrix and the global DOF
    numbers its rows and columns belong to. Where two elements share a node
    they share DOFs, and their stiffnesses ADD -- two members meeting at a
    joint are two springs holding it.
    """
    K = np.zeros((n_dof, n_dof))
    for k, dofs in elements:
        K[np.ix_(dofs, dofs)] += k
    return K
# --8<-- [end:assemble]


# --8<-- [start:solve]
def solve(K, F, fixed):
    """Apply the supports, solve, and recover the reactions.

    A support says a displacement is zero, so its column multiplies nothing and
    its row is an equation for a reaction rather than for a displacement. Both
    are set aside, the remaining 'free' block is solved, and the reactions come
    back from the full matrix afterwards.
    """
    n = len(F)
    free = [d for d in range(n) if d not in fixed]
    u = np.zeros(n)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    R = K @ u - F                       # nonzero only where a support is
    return u, R, free
# --8<-- [end:solve]


# --8<-- [start:truss]
def truss(p=TRUSS):
    """The week 2 truss. Node n owns DOFs 2n and 2n+1 (x then y), counting
    nodes from 0 as Python does."""
    xy = [(0.0, 0.0), (p["span"], 0.0), (p["span"] / 2, p["H"])]
    bars = [(0, 2), (1, 2)]                       # node pairs
    elements = []
    for i, j in bars:
        k = truss_element(*xy[i], *xy[j], p["E"], p["A"])
        elements.append((k, [2*i, 2*i + 1, 2*j, 2*j + 1]))

    K = assemble(2 * len(xy), elements)
    F = np.zeros(2 * len(xy))
    F[5] = -p["P"]                                # apex, downward
    u, R, free = solve(K, F, fixed=[0, 1, 2, 3])  # both supports pinned

    N = []
    for i, j in bars:                             # axial force from elongation
        (xi, yi), (xj, yj) = xy[i], xy[j]
        L = math.hypot(xj - xi, yj - yi)
        c, s = (xj - xi) / L, (yj - yi) / L
        elong = c * (u[2*j] - u[2*i]) + s * (u[2*j + 1] - u[2*i + 1])
        N.append(p["E"] * p["A"] / L * elong)
    return {"K": K, "K_ff": K[np.ix_(free, free)], "u": u, "R": R, "N": N}
# --8<-- [end:truss]


# --8<-- [start:cantilever]
def cantilever(p=BEAM, n_elements=1, P_tip=0.0):
    """The week 1 cantilever, fixed at x = 0, in n equal beam elements.

    A load BETWEEN nodes has to be turned into nodal forces before K u = F can
    carry it. The uniform load q becomes, on each element, the 'equivalent
    nodal loads' qL/2 at each end and a pair of moments qL^2/12.
    """
    n_dof = 2 * (n_elements + 1)                  # v and theta at every node
    Le = p["L"] / n_elements
    elements, F = [], np.zeros(n_dof)
    for e in range(n_elements):
        dofs = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
        elements.append((beam_element(p["E"], p["I"], Le), dofs))
        q = -p["q"]                               # downward
        F[dofs] += [q*Le/2, q*Le**2/12, q*Le/2, -q*Le**2/12]
    F[-2] += -P_tip                               # optional point load at the tip

    K = assemble(n_dof, elements)
    u, R, free = solve(K, F, fixed=[0, 1])        # wall: v = 0, theta = 0
    return {"v_tip": u[-2], "theta_tip": u[-1], "R_wall": R[0], "M_wall": R[1],
            "v": u[0::2], "theta": u[1::2], "K_ff": K[np.ix_(free, free)]}
# --8<-- [end:cantilever]


def report():
    t = truss()
    print("Two-bar truss of week 2, by the direct stiffness method")
    print(f"  structure matrix {t['K'].shape[0]} x {t['K'].shape[1]},"
          f" free block {t['K_ff'].shape[0]} x {t['K_ff'].shape[1]}:")
    for row in t["K_ff"]:
        print("     " + "".join(f"{v:12.1f}" for v in row) + "   N/mm")
    print(f"  apex             ux = {t['u'][4]:10.4f} mm    uy = {t['u'][5]:10.4f} mm")
    print(f"  bar forces       N1 = {t['N'][0]:10.2f} N     N2 = {t['N'][1]:10.2f} N")
    print(f"  reactions, node 1   ({t['R'][0]:9.2f}, {t['R'][1]:9.2f}) N")
    print(f"  reactions, node 2   ({t['R'][2]:9.2f}, {t['R'][3]:9.2f}) N")
    print()

    c1 = cantilever(dict(BEAM, q=0.0), 1, P_tip=CANTILEVER["P"])
    print("Cantilever of week 1, 10 kN at the tip, one element")
    print(f"  tip deflection {c1['v_tip']:10.4f} mm")
    print()

    print("The same cantilever under q = 10 N/mm, n elements")
    print("  elements   tip deflection   tip rotation   wall moment")
    print("                  [mm]           [rad]          [kNm]")
    for n in (1, 2, 4):
        c = cantilever(BEAM, n)
        print(f"  {n:5d}      {c['v_tip']:10.4f}     {c['theta_tip']:10.6f}"
              f"     {c['M_wall']/1e6:9.3f}")


if __name__ == "__main__":
    report()
