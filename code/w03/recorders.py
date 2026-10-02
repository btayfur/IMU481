"""Week 3 - two ways to get results out, and why you need both.

    python code/w03/recorders.py

OpenSees will tell you what happened, but only if you ask in the right way, and
there are exactly two ways to ask.

    ops.nodeDisp(...)   asks the model what it looks like RIGHT NOW.
    ops.recorder(...)   asks OpenSees to write something down at EVERY step.

The difference does not show up in a one-step linear analysis, because "now"
and "the whole history" are the same thing. Run the same truss in ten steps and
the difference becomes the entire point: nodeDisp knows only the last step, and
everything that happened on the way there exists only if a recorder was set up
BEFORE the analysis ran. A recorder cannot be added afterwards -- the steps are
gone.

This is the most expensive lesson in the course to learn the hard way: a long
nonlinear analysis that finished successfully and recorded nothing has to be
run again.

Units: N, mm, s.
"""

import sys
from pathlib import Path

import openseespy.opensees as ops

# Later weeks reuse earlier weeks' models rather than copying them. Putting
# code/ on the path lets this script be run directly as well as imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w02.truss import TRUSS, build_model            # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


# --8<-- [start:recorder]
def setup_recorder(path, precision=6):
    """Ask for the apex displacement to be written at every converged step.

    Set this up AFTER the model exists -- the recorder refers to node 3 by tag,
    so node 3 has to be there -- and BEFORE the analysis runs, because a
    recorder only ever sees the steps that come after it.

    '-precision' matters more than it looks. A recorder writes TEXT, and the
    default is six significant figures, so what comes back out of the file is
    not bit-identical to what was in memory: -0.723380 rather than
    -0.7233796296296295. Six figures is plenty for a displacement you are going
    to plot, and nowhere near enough if you are subtracting two nearly equal
    recorded numbers -- a residual, a drift between two storeys, an increment
    between two steps. Ask for more when you are going to do arithmetic on the
    output.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return ops.recorder("Node", "-file", str(path),
                        "-precision", precision,
                        "-time",                 # first column = pseudo-time
                        "-node", 3,
                        "-dof", 1, 2, "disp")
# --8<-- [end:recorder]


# --8<-- [start:stepped]
def run_stepped(n_steps):
    """Apply the load in n_steps equal increments instead of all at once.

    LoadControl's argument is the load factor added per step, so n steps of
    1/n arrive at exactly the same place as one step of 1 -- for a linear
    model. The reason to do it anyway is that it produces a PATH, and a path is
    what you plot, check and compare against a test.
    """
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandSPD")
    ops.test("NormDispIncr", 1.0e-8, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0 / n_steps)
    ops.analysis("Static")

    for step in range(n_steps):
        if ops.analyze(1) != 0:
            raise RuntimeError(f"failed at step {step + 1} of {n_steps}")
# --8<-- [end:stepped]


# --8<-- [start:readback]
def read_history(path):
    """Read a recorder file back into columns.

    A Node recorder writes one row per step and one column per requested
    quantity, whitespace separated, no header. With '-time' the first column is
    the pseudo-time; here that is the load factor, running 0.1 to 1.0.
    """
    rows = [[float(x) for x in line.split()]
            for line in path.read_text().splitlines() if line.strip()]
    return {
        "lambda": [r[0] for r in rows],
        "ux": [r[1] for r in rows],
        "uy": [r[2] for r in rows],
    }
# --8<-- [end:readback]


# --8<-- [start:history]
def history(n_steps=10, path=None):
    """Build, record, run, and read the path back.

    Note the order, because it is not negotiable: model, recorder, analysis,
    then ops.wipe() to flush the file. A recorder file is buffered, and reading
    it before the buffer is flushed gives a short file or an empty one.
    """
    path = path or (OUT / "apex.out")
    build_model(TRUSS)
    setup_recorder(path)
    run_stepped(n_steps)

    final_now = ops.nodeDisp(3, 2)     # what the model knows: the last step only
    ops.wipe()                         # closes the recorder and flushes the file

    return final_now, read_history(path)
# --8<-- [end:history]


if __name__ == "__main__":
    final_now, path = history(10)

    print("Two-bar truss under 50 kN, applied in 10 equal steps")
    print(f"  {'load factor':>12}{'ux [mm]':>14}{'uy [mm]':>14}")
    for lam, ux, uy in zip(path["lambda"], path["ux"], path["uy"]):
        print(f"  {lam:12.1f}{ux:14.4e}{uy:14.4f}")

    print()
    print(f"  ops.nodeDisp(3, 2) after the run : {final_now:.4f} mm")
    print(f"  last row of the recorder file    : {path['uy'][-1]:.4f} mm")
    print(f"  rows the recorder captured       : {len(path['uy'])}")
    print()
    print("  The two agree -- and that is the point. nodeDisp gave one number;")
    print("  the recorder gave the ten that led to it. Only one of those can be")
    print("  plotted, and only one of them survives the analysis finishing.")
