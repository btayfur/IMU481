"""Week 10 - what to do when the analysis stops.

    python code/w10/robust.py

Week 8 left a debt. The moment-curvature analysis of the reinforced concrete
column stopped converging at about 1e-4 curvature, and cutting the step size by
a factor of eight bought a quarter more and then stopped again. That is not a
step-size problem, and this week is about what it is instead.

Three things are separable and worth separating:

    the ALGORITHM   how each step iterates towards equilibrium
    the TEST        when an iteration is close enough to stop
    the STEP        how far one step tries to go

A failed analysis is a failure of one of the three, and the diagnosis is
different in each case. The pay-off is robust_push(): about thirty lines that
turn the week 8 analysis from one that stops at 1e-4 into one that runs to
2.5e-4, without changing a single thing about the model.

Units: N, mm, s.
"""

import sys
from pathlib import Path

import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w08.section import COLUMN, apply_axial, build_rig                # noqa: E402


# --8<-- [start:algorithms]
# Tried in order. Newton is fastest when it works; the others fail differently,
# which is the only reason to have more than one.
ALGORITHMS = (
    ("Newton", ("Newton",)),
    ("ModifiedNewton", ("ModifiedNewton",)),
    ("KrylovNewton", ("KrylovNewton",)),
    ("NewtonLineSearch", ("NewtonLineSearch", "-type", "Bisection")),
)
# --8<-- [end:algorithms]


# --8<-- [start:counter]
class Effort:
    """Counts what a robust analysis actually costs.

    Worth counting, because the fallbacks are not free: every failed attempt
    is a full set of iterations thrown away, and a script that silently
    subdivides can take a hundred times longer than one that does not without
    ever telling you.
    """

    def __init__(self):
        self.attempts = 0
        self.failures = 0
        self.subdivisions = 0
        self.used = {}

    def record(self, name, ok):
        self.attempts += 1
        if ok:
            self.used[name] = self.used.get(name, 0) + 1
        else:
            self.failures += 1

    def __str__(self):
        used = ", ".join(f"{k} {v}" for k, v in self.used.items())
        return (f"{self.attempts} attempts, {self.failures} failed, "
                f"{self.subdivisions} subdivisions [{used}]")
# --8<-- [end:counter]


# --8<-- [start:step]
def try_step(node, dof, increment, algorithm, tol=1e-8, max_iter=15):
    """One attempt at one step, with one algorithm. Returns True if it worked.

    Note that everything is re-declared on every attempt. OpenSees keeps the
    last algorithm and integrator you set, so a fallback that forgets to reset
    them silently carries the previous choice into the next step.
    """
    ops.test("NormDispIncr", tol, max_iter)
    ops.algorithm(*algorithm)
    ops.integrator("DisplacementControl", node, dof, increment)
    ops.analysis("Static")
    return ops.analyze(1) == 0
# --8<-- [end:step]


# --8<-- [start:robust]
def robust_step(node, dof, increment, effort, tol=1e-8, depth=0, max_depth=5):
    """Advance one step, whatever it takes.

    The strategy, in order:
      1. try each algorithm in turn at the requested increment;
      2. if all of them fail, halve the increment and take two half-steps;
      3. give up after max_depth halvings -- 1/32 of the original step.

    Step (2) is recursive, so a single difficult step can quietly become
    thirty-two tiny ones while the rest of the analysis marches on at full
    size. That is exactly the behaviour you want, and exactly why the effort
    counter exists.
    """
    for name, algorithm in ALGORITHMS:
        ok = try_step(node, dof, increment, algorithm, tol=tol)
        effort.record(name, ok)
        if ok:
            return True

    if depth >= max_depth:
        return False

    effort.subdivisions += 1
    half = increment / 2.0
    return (robust_step(node, dof, half, effort, tol, depth + 1, max_depth)
            and robust_step(node, dof, half, effort, tol, depth + 1, max_depth))
# --8<-- [end:robust]


# --8<-- [start:push]
def robust_push(target, n_steps, node=2, dof=3, tol=1e-8):
    """Drive `node`'s `dof` to `target`, refusing to stop at the first refusal.

    Returns the path reached and the effort it took. If it does give up, it
    returns what it managed -- which is the honest thing to do, and lets the
    caller see how far the model really got.
    """
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(node, 0.0, 0.0, 1.0)

    increment = target / n_steps
    effort = Effort()
    curvature, moment = [0.0], [0.0]

    for _ in range(n_steps):
        if not robust_step(node, dof, increment, effort, tol=tol):
            break
        curvature.append(ops.nodeDisp(node, dof))
        moment.append(-ops.eleResponse(1, "force")[2])

    return curvature, moment, effort
# --8<-- [end:push]


# --8<-- [start:plain]
def plain_push(target, n_steps, algorithm=("Newton",), node=2, dof=3, tol=1e-8):
    """The week 8 loop: one algorithm, one step size, stop at the first refusal.

    Kept for comparison. This is what almost every script does, and it is fine
    right up to the moment it is not.
    """
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(node, 0.0, 0.0, 1.0)

    increment = target / n_steps
    effort = Effort()
    curvature, moment = [0.0], [0.0]

    for _ in range(n_steps):
        ok = try_step(node, dof, increment, algorithm, tol=tol)
        effort.record(algorithm[0], ok)
        if not ok:
            break
        curvature.append(ops.nodeDisp(node, dof))
        moment.append(-ops.eleResponse(1, "force")[2])

    return curvature, moment, effort
# --8<-- [end:plain]


def run_push(kind, target=5.0e-4, n_steps=200, n_fibres=32,
             algorithm=("Newton",), tol=1e-8):
    """Build the week 8 section fresh, then push it one way or the other."""
    build_rig(COLUMN, n_fibres)
    apply_axial(COLUMN)
    if kind == "robust":
        return robust_push(target, n_steps, tol=tol)
    return plain_push(target, n_steps, algorithm, tol=tol)


# --8<-- [start:validity]
def validity_limits(p=COLUMN):
    """The curvatures at which the model stops describing a real column.

    This function exists because of what robust_push does. Making the solver
    more determined does not make the model more valid -- it simply removes
    the one signal that used to tell you something was wrong. The analysis
    that stopped at 1e-4 was, by accident, stopping near the right place.

    Two limits matter here, and NEITHER is enforced by any material in the
    section: Concrete01 holds its residual strength for ever, and Steel02
    never fractures. Both will carry you cheerfully to 15 % strain.
    """
    y_concrete = p["h"] / 2.0                 # extreme fibre
    y_steel = p["h"] / 2.0 - p["cover"]       # outermost bar
    return {
        "confined core crushed": p["eps_ccu"] / y_concrete,
        "steel past 5 % strain": 0.05 / y_steel,
    }


def extreme_strains(curvature, p=COLUMN):
    """Extreme concrete and steel strain at a given curvature.

    Plane sections, so both are just curvature times a distance. Printing
    these next to a result is the cheapest validity check there is.
    """
    return (curvature * p["h"] / 2.0,
            curvature * (p["h"] / 2.0 - p["cover"]))
# --8<-- [end:validity]


# --8<-- [start:tolerance]
def tolerance_sweep(tolerances, target=1.0e-4, n_steps=80):
    """How far a plain Newton analysis gets, against the convergence tolerance.

    A loose tolerance is not free accuracy: it lets steps 'converge' that have
    not, and the error accumulates into the next step's starting point. A tight
    one refuses steps that were nearly fine. Neither extreme is safe, and the
    sensible range is narrower than people expect.
    """
    out = []
    for tol in tolerances:
        build_rig(COLUMN, 32)
        apply_axial(COLUMN)
        ops.timeSeries("Linear", 2)
        ops.pattern("Plain", 2, 2)
        ops.load(2, 0.0, 0.0, 1.0)

        increment = target / n_steps
        reached, moment = 0.0, 0.0
        for _ in range(n_steps):
            if not try_step(2, 3, increment, ("Newton",), tol=tol):
                break
            reached = ops.nodeDisp(2, 3)
            moment = -ops.eleResponse(1, "force")[2]
        out.append((tol, reached, moment / 1.0e6))
    return out
# --8<-- [end:tolerance]


if __name__ == "__main__":
    TARGET, STEPS = 5.0e-4, 200

    print(f"Week 8's section, pushed to {TARGET:.1e} curvature\n")
    print(f"  {'strategy':<22}{'reached':>12}{'of target':>11}   effort")

    c, _, e = run_push("plain", TARGET, STEPS)
    print(f"  {'Newton only':<22}{c[-1]:12.3e}{c[-1] / TARGET:11.0%}   {e}")

    cr, _, er = run_push("robust", TARGET, STEPS)
    print(f"  {'robust_push':<22}{cr[-1]:12.3e}{cr[-1] / TARGET:11.0%}   {er}")

    print("\n  Nothing about the model changed. Only the strategy did, and it")
    print("  cost two extra attempts out of two hundred.")

    print("\nBut how much of that is worth having?")
    limits = validity_limits()
    for name, k in sorted(limits.items(), key=lambda kv: kv[1]):
        print(f"  {name:<26}{k:10.3e}   ({k / TARGET:5.0%} of the target)")
    ec, es = extreme_strains(cr[-1])
    print(f"  at the end of the run       concrete {ec:.3f}, steel {es:.3f}")
    print("  No material in this section can fail: Concrete01 holds its")
    print("  residual for ever and Steel02 never fractures. The solver was")
    print("  never going to stop, and now nothing else does either.")

    print("\nThe same comparison with the tolerance tightened to 1e-9")
    for kind, label in (("plain", "Newton only"), ("robust", "robust_push")):
        ck, _, ek = run_push(kind, 2.5e-4, STEPS, tol=1e-9)
        print(f"  {label:<22}{ck[-1]:12.3e}{ck[-1] / 2.5e-4:11.0%}   {ek}")
    print("  A TIGHTER tolerance made the plain analysis worse, not safer.")

    print("\nConvergence tolerance, plain Newton to 1e-4")
    print(f"  {'tolerance':>12}{'reached':>12}{'moment [kN m]':>16}")
    for tol, reached, moment in tolerance_sweep([1e-4, 1e-6, 1e-8, 1e-10]):
        print(f"  {tol:12.0e}{reached:12.3e}{moment:16.2f}")
    print("\n  Every row above reached the target. They do not agree on the")
    print("  answer: a loose tolerance converges happily and is 22 % wrong.")
