"""Week 1 - prove that your environment works, and meet the habit of the course.

    python code/w01/environment_check.py

Running this teaches three things the printed text cannot.

First, whether your installation is actually usable. Every import below is one
the rest of the course depends on, and an import that fails here fails loudly,
now, instead of in week 6 in the middle of a fiber section.

Second, the unit system. OpenSees carries no units -- it multiplies whatever
numbers you hand it. This course works in NEWTON, MILLIMETRE, SECOND, so a
modulus is written 200000 and reads as MPa. Nothing warns you if you mix
systems; the analysis succeeds and the answer is wrong.

Third, and most important, the habit: a result you have not checked against
something independent is not a result. The cantilever below has a closed-form
answer that has been in textbooks for two centuries. We compute it both ways
and compare. Every worked example in these notes ends this way.
"""

# --8<-- [start:imports]
import openseespy.opensees as ops
# --8<-- [end:imports]


# --8<-- [start:params]
# Units: N, mm, s.  Every number below obeys them and nothing else does.
CANTILEVER = {
    "L": 3000.0,      # mm      span
    "E": 200000.0,    # MPa     = N/mm^2, structural steel
    "I": 8.356e7,     # mm^4    IPE 300, strong axis
    "A": 5380.0,      # mm^2    IPE 300
    "P": 10000.0,     # N       = 10 kN, downward at the tip
}
# --8<-- [end:params]


# --8<-- [start:build_model]
def build_model(p):
    """Build a 2D cantilever: fixed at node 1, loaded downward at node 2.

    The first line is ops.wipe(). OpenSees keeps ONE global model in memory,
    so without it you silently extend whatever was built last -- and the
    symptom appears later, somewhere else, as a wrong answer rather than an
    error. Start every model function this way, for the whole course.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    ops.node(1, 0.0, 0.0)
    ops.node(2, p["L"], 0.0)
    ops.fix(1, 1, 1, 1)                       # fully fixed: ux, uy, rotation

    ops.geomTransf("Linear", 1)
    ops.element("elasticBeamColumn", 1, 1, 2, p["A"], p["E"], p["I"], 1)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0.0, -p["P"], 0.0)            # negative = downward
# --8<-- [end:build_model]


# --8<-- [start:run_analysis]
def run_analysis():
    """Apply the load in one step and return the tip displacement and rotation.

    This is the linear static recipe. Week 3 explains what each of these seven
    objects is for; week 10 explains what to do when this recipe is not enough.
    For now, take it as the shortest thing that works.
    """
    ops.system("BandGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")

    if ops.analyze(1) != 0:
        raise RuntimeError("the analysis did not converge")

    return ops.nodeDisp(2, 2), ops.nodeDisp(2, 3)
# --8<-- [end:run_analysis]


# --8<-- [start:closed_form]
def closed_form(p):
    """Tip deflection and rotation of a cantilever under a point load.

    delta = P L^3 / (3 E I)      theta = P L^2 / (2 E I)

    Negative because the load acts downward, matching the model's sign
    convention. Getting a sign convention to agree is half of any verification.
    """
    delta = -p["P"] * p["L"] ** 3 / (3.0 * p["E"] * p["I"])
    theta = -p["P"] * p["L"] ** 2 / (2.0 * p["E"] * p["I"])
    return delta, theta
# --8<-- [end:closed_form]


# --8<-- [start:verify]
def check(p=CANTILEVER):
    """Run both routes and return them side by side, with relative errors."""
    build_model(p)
    num_delta, num_theta = run_analysis()
    exact_delta, exact_theta = closed_form(p)

    return {
        "delta": (num_delta, exact_delta, abs(num_delta - exact_delta) / abs(exact_delta)),
        "theta": (num_theta, exact_theta, abs(num_theta - exact_theta) / abs(exact_theta)),
    }
# --8<-- [end:verify]


def _report_imports():
    """Name every package the course needs, and say which one is missing."""
    import importlib

    for name in ("openseespy", "opsvis", "opstool", "numpy", "matplotlib"):
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "installed")
            print(f"  {name:<12} {version}")
        except ImportError as exc:
            print(f"  {name:<12} *** NOT AVAILABLE *** ({exc})")


if __name__ == "__main__":
    print("Packages")
    _report_imports()

    print("\nCantilever, units N-mm-s")
    result = check()
    print(f"  {'':<12}{'OpenSees':>16}{'closed form':>16}{'rel. error':>14}")
    for name, unit in (("delta", "mm"), ("theta", "rad")):
        got, want, err = result[name]
        print(f"  {name + ' [' + unit + ']':<12}{got:16.4f}{want:16.4f}{err:14.2e}")

    print("\nIf the two columns agree, your environment works and you have")
    print("just done the only thing that makes a computed answer trustworthy.")
