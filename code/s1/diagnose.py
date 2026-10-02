"""S1 - what to do when OpenSees will not run, stops early, or quietly lies.

    python code/s1/diagnose.py

Three kinds of failure, in increasing order of danger:

    it refuses      an exception, or a WARNING followed by nothing. Loud, and
                    therefore the cheapest kind. The message names the command.
    it stops early  analyze() returns a negative number and the loop ends where
                    the structure has not. Week 10 is about this one.
    it lies         the script runs to completion, prints numbers, draws a
                    figure, and every one of them is wrong. Nothing at all is
                    printed to tell you.

Almost every hour a student loses goes to the third kind, and it is the only
one no error message will ever help with. So this file is mostly about that:
four failures that produce no warning whatever, each run twice - once broken,
once correct - so the damage is a number rather than a caution.

Units: N, mm, s.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openseespy.opensees as ops                                   # noqa: E402

G = 9810.0                              # mm/s^2


# --8<-- [start:audit]
def audit(dynamic=False):
    """A health check on whatever is in the domain RIGHT NOW.

    Call it after building and before analysing. Every check here corresponds
    to a mistake that OpenSees itself will not report -- it will either fail
    with a message that names the wrong culprit, or succeed and be wrong.

    Returns a list of complaints; an empty list is the good case.
    """
    problems = []
    nodes = ops.getNodeTags()
    eles = ops.getEleTags()

    if not nodes:
        return ["the domain is empty -- did the model get built at all?"]
    if not eles:
        problems.append("nodes exist but no elements: nothing connects them")

    # A node no element refers to. Usually a typo in an element's node list,
    # and the giveaway is that the model looks right in a plot.
    used = set()
    for e in eles:
        used.update(ops.eleNodes(e))
    orphans = [n for n in nodes if n not in used]
    if orphans:
        problems.append(f"nodes attached to no element: {orphans}")

    # Duplicate coordinates: two nodes at the same point are not connected to
    # each other, and the structure has a hidden hinge that no plot will show.
    seen = {}
    for n in nodes:
        key = tuple(round(c, 6) for c in ops.nodeCoord(n))
        seen.setdefault(key, []).append(n)
    doubled = [v for v in seen.values() if len(v) > 1]
    if doubled:
        problems.append(f"nodes sharing a location: {doubled} -- intentional "
                        "only if you meant a zeroLength element there")

    # For a transient run, a free DOF with no mass makes the mass matrix
    # singular. OpenSees will report a solver failure and blame the solver.
    if dynamic:
        massless = []
        for n in nodes:
            m = ops.nodeMass(n)
            free = [i for i in range(len(m))
                    if not _is_fixed(n, i + 1)]
            if free and all(m[i] == 0.0 for i in free):
                massless.append(n)
        if massless:
            problems.append(f"free but massless nodes: {massless} -- a "
                            "transient analysis needs mass on every free DOF")

    return problems


def _is_fixed(node, dof):
    """True if `dof` (1-based) of `node` carries a single-point constraint."""
    try:
        return dof in ops.getFixedDOFs(node)
    except Exception:                          # older builds lack the query
        return False
# --8<-- [end:audit]


# --8<-- [start:mechanism]
def mechanism_check(n=3):
    """Is the structure a mechanism? Ask for eigenvalues and read the smallest.

    A properly restrained structure has a positive-definite stiffness matrix,
    so every eigenvalue is comfortably positive. Leave one restraint off and
    the structure can move without straining: that motion has zero stiffness,
    and the eigenvalue collapses towards zero.

    This is the cheapest possible test for the single most common modelling
    error, and it costs one line. It is also far more informative than the
    solver's own message, which typically says the matrix is singular without
    saying which direction is free.
    """
    try:
        vals = ops.eigen('-fullGenLapack', n)
    except Exception as exc:                   # a hard singularity throws
        return None, f"eigen failed outright: {exc}"

    smallest = min(vals)
    if smallest < 1.0e-8:
        return smallest, ("a zero eigenvalue: the structure is a mechanism. "
                          "Something is unrestrained, or two elements meet at "
                          "coincident but unconnected nodes")
    return smallest, "restrained: the smallest eigenvalue is positive"
# --8<-- [end:mechanism]


# --8<-- [start:codes]
# What analyze() means when it does not return 0. It returns a code; it does
# not raise. `if ops.analyze(1) < 0` is therefore not optional -- without it a
# failed step is indistinguishable from a converged one, and the loop happily
# carries on integrating a structure that never came into equilibrium.
ANALYZE_CODES = {
    0: "converged",
    -1: "the algorithm failed -- residual never fell below the test tolerance",
    -2: "the integrator failed -- often a step size the state cannot support",
    -3: "the system of equations failed -- singular or ill-conditioned matrix",
    -4: "the constraint handler failed",
    -5: "the analysis could not be created or the domain changed under it",
}


def explain(code):
    """Turn an analyze() return value into something actionable."""
    return ANALYZE_CODES.get(code, f"undocumented return code {code}")
# --8<-- [end:codes]


# ---------------------------------------------------------------------------
# Four failures that print nothing at all. Each is run twice.
# ---------------------------------------------------------------------------

def _cantilever(mass_unit="tonne", restrain=True):
    """A 3 m steel cantilever with a mass on top. The rig for the demos below."""
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, 3000.0)
    if restrain:
        ops.fix(1, 1, 1, 1)
    else:
        ops.fix(1, 1, 1, 0)                    # base free to rotate: a mechanism
    weight = 50.0e3                            # 50 kN
    m = weight / G if mass_unit == "tonne" else weight / 9.81
    ops.mass(2, m, m, 0.0)
    ops.geomTransf('Linear', 1)
    ops.element('elasticBeamColumn', 1, 1, 2, 8.0e3, 200000.0, 1.6e8, 1)
    return weight


# --8<-- [start:reactions]
def reactions_forgotten():
    """The classic silent zero: reactions are not computed until you ask.

    ops.reactions() runs the restoring-force recovery. Skip it and every
    nodeReaction call returns exactly 0.0 -- not an error, not a warning, just
    a beautifully formatted table of zeros that fails equilibrium by 100 %.
    """
    weight = _cantilever()
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(2, 0.0, -weight, 0.0)
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    ops.analyze(1)

    without = ops.nodeReaction(1, 2)           # before reactions() -- zero
    ops.reactions()
    with_call = ops.nodeReaction(1, 2)         # after -- the real value
    return without, with_call, weight
# --8<-- [end:reactions]


# --8<-- [start:mass]
def mass_in_kilograms():
    """Mass in kg inside an N-mm-s model: every period wrong by the same factor.

    In N, mm, s the mass unit is N*s^2/mm, which is a tonne. Dividing a weight
    in newtons by 9.81 instead of 9810 gives a number a thousand times too
    large, and since T goes as the square root of mass, every period comes out
    sqrt(1000) = 31.6 times too long.

    Nothing warns. The model runs, the mode shapes are perfect, and a building
    that should have T = 0.5 s reports 15.8 s -- which, if you do not know the
    answer already, looks like a very flexible building rather than a bug.
    """
    def period(unit):
        _cantilever(mass_unit=unit)
        w2 = ops.eigen('-fullGenLapack', 1)[0]
        return 2.0 * math.pi / math.sqrt(w2)

    right = period("tonne")
    wrong = period("kg")
    return right, wrong, wrong / right
# --8<-- [end:mass]


# --8<-- [start:contamination]
def domain_contamination():
    """No wipe(): the second model is built on top of the first one.

    OpenSees keeps ONE global domain. Building a second model without wiping
    does not replace the first -- it adds to it. In a notebook, where cells are
    re-run out of order, this is the single most common source of results that
    change between runs of identical code.

    The tell is the node count. The symptom is a period that moves every time
    you press run.
    """
    _cantilever()
    first = len(ops.getNodeTags())

    # A second cantilever, deliberately without wipe(), with fresh tags so
    # nothing collides and nothing complains.
    ops.node(3, 5000.0, 0.0)
    ops.node(4, 5000.0, 3000.0)
    ops.fix(3, 1, 1, 1)
    ops.mass(4, 5.0, 5.0, 0.0)
    ops.element('elasticBeamColumn', 2, 3, 4, 8.0e3, 200000.0, 1.6e8, 1)
    after = len(ops.getNodeTags())

    return first, after
# --8<-- [end:contamination]


# --8<-- [start:tolerance]
def _pull_truss(tol, steps=20):
    """A bar pulled well past yield in `steps` load increments. Returns the path.

    Steel02 rather than Steel01 so the tangent varies smoothly through yield;
    with a bilinear material the transition occupies a single step and there is
    much less for a bad tolerance to get wrong.
    """
    ops.wipe()
    ops.model('basic', '-ndm', 1, '-ndf', 1)
    ops.node(1, 0.0)
    ops.node(2, 1000.0)
    ops.fix(1, 1)
    ops.uniaxialMaterial('Steel02', 1, 420.0, 200000.0, 0.01, 18.0, 0.925, 0.15)
    ops.element('truss', 1, 1, 2, 500.0, 1)

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(2, 400.0e3)                       # well past the 210 kN yield load
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.test('NormDispIncr', tol, 20)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 1.0 / steps)
    ops.analysis('Static')

    path, failures = [], 0
    for _ in range(steps):
        if ops.analyze(1) < 0:
            failures += 1
        path.append(ops.nodeDisp(2, 1))
    return path, failures


# --8<-- [start:tolerance]
def tolerance_too_loose(loose=200.0, tight=1.0e-10, steps=20):
    """A tolerance so loose that "converged" stops meaning anything -- and what
    that actually costs, which is not what you would guess.

    test('NormDispIncr', tol, iters) accepts the step when the displacement
    increment between two iterations falls below `tol`. Set `tol` large enough
    and the first iteration always passes: every step reports success and the
    answer is whatever one tangent step happened to give.

    The obvious expectation is that the final answer is then wrong. It is not.
    Pulled monotonically past yield, the loose run finishes on exactly the same
    displacement as the tight one, to every printed digit.

    The reason is worth understanding, because it is what makes this failure
    hard to catch: the force left unbalanced by a lazy step is not discarded.
    It stays in the residual and turns up in the NEXT step, so under a
    monotonic load with a hardening material the errors are repaid and the
    endpoint recovers.

    What does not recover is the PATH. Through the two steps where the bar
    actually yields, the loose run is out by more than eighty per cent -- it
    simply has not noticed yet. So anything path-dependent is wrong: which step
    yielded, the energy under the curve, the peak of a cycle, the state a
    transient analysis carries into its inertia terms, where the unbalance has
    nowhere to be repaid from.

    Which is the real lesson. Loosening the tolerance until the complaints stop
    does not fix the failure and does not always corrupt the headline number.
    It relocates the damage to the part of the answer nobody checks.
    """
    tight_path, _ = _pull_truss(tight, steps)
    loose_path, failures = _pull_truss(loose, steps)

    err = [abs(a - b) / max(abs(a), 1.0e-9)
           for a, b in zip(tight_path, loose_path)]
    worst = max(range(len(err)), key=err.__getitem__)
    return {
        "tight": tight_path,
        "loose": loose_path,
        "final_error": err[-1],
        "worst_error": err[worst],
        "worst_step": worst + 1,
        "reported_failures": failures,
    }
# --8<-- [end:tolerance]
# --8<-- [end:tolerance]


if __name__ == "__main__":
    print("S1 -- four failures that produce no error message\n")

    without, with_call, weight = reactions_forgotten()
    print("1. reactions() not called")
    print(f"     before the call {without:12.1f} N")
    print(f"     after the call  {with_call:12.1f} N   (applied {-weight:.1f} N)")
    print("     No warning. The first number is what an unwary script reports.")

    right, wrong, ratio = mass_in_kilograms()
    print("\n2. mass entered in kg instead of tonnes")
    print(f"     correct  T = {right:8.4f} s")
    print(f"     in kg    T = {wrong:8.4f} s     ({ratio:.2f} times too long)")
    print(f"     sqrt(1000) = {math.sqrt(1000.0):.2f}, and there it is.")

    first, after = domain_contamination()
    print("\n3. a second model built without wipe()")
    print(f"     nodes after the first model  {first}")
    print(f"     nodes after the second       {after}   (not {first})")
    print("     The domain is global. It accumulated instead of resetting.")

    t = tolerance_too_loose()
    print("\n4. a convergence tolerance chosen to stop the complaints")
    print(f"     steps reported as failed  {t['reported_failures']}")
    print(f"     error in the final answer {t['final_error']:8.2%}")
    print(f"     worst error along the way {t['worst_error']:8.2%}"
          f"  (step {t['worst_step']}, where the bar yields)")
    print("     The endpoint is exact because the unbalanced force is carried")
    print("     into the next step and repaid. The PATH is not, and anything")
    print("     path-dependent -- energy, a peak, a transient state -- is wrong.")

    print("\nAnd one that is loud, for contrast:")
    _cantilever(restrain=False)
    val, message = mechanism_check(1)
    print(f"     base released:  smallest eigenvalue {val:.3e}")
    print(f"     {message}")
    _cantilever(restrain=True)
    val, message = mechanism_check(1)
    print(f"     base fixed:     smallest eigenvalue {val:.3e}")
    print(f"     {message}")
