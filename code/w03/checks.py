"""Week 3 - four ways to find out whether a model is right.

    python code/w03/checks.py

A closed-form comparison is the check everyone thinks of, and it is the one you
can least often make: real structures do not have textbook solutions, which is
why you are modelling them. The useful checks are the ones that need no
external answer, because those are available on every model you will ever
build.

The four here are ordered by how widely they apply, weakest last:

    1. equilibrium  - do the reactions add up to the applied load?
    2. symmetry     - does a symmetric structure respond symmetrically?
    3. scaling      - does doubling the load double the response, when theory
                      says it must?
    4. closed form  - does it match the textbook, when there is one?

The first three test the model you actually built. None of them proves it
right; each of them can prove it wrong, which is all any test does.

Units: N, mm, s.
"""

import sys
from pathlib import Path

import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w02.truss import TRUSS, build_model, hand_check, run, solve   # noqa: E402


# --8<-- [start:equilibrium]
def check_equilibrium(p=TRUSS):
    """Sum the reactions and compare them with what was applied.

    ops.reactions() computes them; ops.nodeReaction() then reads them node by
    node. Forgetting the first call is a classic: nodeReaction returns zeros,
    which looks like a support that is not working rather than a query that was
    never answered.

    This check needs no theory and no reference solution, so it works on any
    model, however complicated -- which is exactly what makes it the one to
    reach for first.
    """
    build_model(p)
    run()
    ops.reactions()

    rx = sum(ops.nodeReaction(n, 1) for n in (1, 2))
    ry = sum(ops.nodeReaction(n, 2) for n in (1, 2))

    # The only applied load is P downward at the apex.
    return {"sum_rx": rx, "sum_ry": ry, "applied": -p["P"],
            "residual": ry - p["P"]}
# --8<-- [end:equilibrium]


# --8<-- [start:symmetry]
def check_symmetry(p=TRUSS):
    """A structure symmetric in geometry, supports AND load must respond so.

    All three have to hold. It is easy to build a symmetric frame and then
    restrain one support differently from the other; the model runs, and this
    check is what notices.
    """
    got = solve(p)
    return {"ux": got["ux"], "N1": got["N1"], "N2": got["N2"],
            "force_gap": abs(got["N1"] - got["N2"])}
# --8<-- [end:symmetry]


# --8<-- [start:scaling]
def check_scaling(p=TRUSS, factor=2.0):
    """In a linear elastic model, response is proportional to load. Exactly.

    Worth running even though it looks trivial, because it stops being true the
    moment anything in the model is nonlinear -- and from week 10 onward this
    same check becomes the way you DETECT that something has yielded.
    """
    base = solve(p)
    scaled = solve(dict(p, P=p["P"] * factor))
    return {"uy_base": base["uy"], "uy_scaled": scaled["uy"],
            "ratio": scaled["uy"] / base["uy"]}
# --8<-- [end:scaling]


# --8<-- [start:closed_form]
def check_closed_form(p=TRUSS):
    """Compare with the hand solution -- when one exists.

    Note what this does and does not establish. Both routes assume pin-jointed
    bars carrying axial force only, so agreement confirms the geometry, the
    areas, the modulus and the arithmetic. It says nothing about whether
    pin-jointed bars are the right idealisation of the real connection.
    """
    got = solve(p)
    want = hand_check(p)
    return {"uy_model": got["uy"], "uy_hand": want["uy"],
            "N_model": got["N1"], "N_hand": want["N"],
            "rel_error": abs(got["uy"] - want["uy"]) / abs(want["uy"])}
# --8<-- [end:closed_form]


if __name__ == "__main__":
    print("Four checks on the week 2 truss\n")

    eq = check_equilibrium()
    print("1. Equilibrium")
    print(f"   sum of vertical reactions {eq['sum_ry']:12.4f} N")
    print(f"   applied load              {-eq['applied']:12.4f} N")
    print(f"   residual                  {eq['residual']:12.4e} N")

    sy = check_symmetry()
    print("\n2. Symmetry")
    print(f"   apex horizontal movement  {sy['ux']:12.4e} mm")
    print(f"   difference in bar forces  {sy['force_gap']:12.4e} N")

    sc = check_scaling()
    print("\n3. Scaling (load doubled)")
    print(f"   uy at P                   {sc['uy_base']:12.4f} mm")
    print(f"   uy at 2P                  {sc['uy_scaled']:12.4f} mm")
    print(f"   ratio                     {sc['ratio']:12.6f}")

    cf = check_closed_form()
    print("\n4. Closed form")
    print(f"   uy from the model         {cf['uy_model']:12.4f} mm")
    print(f"   uy by hand                {cf['uy_hand']:12.4f} mm")
    print(f"   relative error            {cf['rel_error']:12.2e}")

    print("\n   Checks 1-3 needed no reference solution. Check 4 did, and on a")
    print("   real structure it is the one you will usually have to do without.")
