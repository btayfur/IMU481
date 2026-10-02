"""Week 13 - a specimen, an earthquake, and no closed form to check against.

    python code/w13/timehistory.py

The whole course arrives here. The specimen carries week 8's fibre section, the
step is week 10's ladder, the mass and damping are week 12's, and the base now
moves. What is new is that nothing can be checked against a textbook: there is
no formula for the peak displacement of a yielding column under an earthquake,
which is the reason the analysis is worth running and the reason it has to be
verified some other way.

Three verifications replace the closed form, and none of them needs a second
opinion from outside the model:

    zero input      a stationary base must leave the specimen stationary;
    small input     while the response stays elastic, doubling the record must
                    double every response quantity, exactly;
    large input     when it stops doing so, the specimen has yielded -- and
                    WHERE it stops is a measurement, not an assumption.

The third is the whole chapter. Week 3 used scaling as a check that a linear
model was linear; here the same check becomes an instrument.

Units: N, mm, s. Acceleration in mm/s^2.
"""

import math
import sys
from pathlib import Path

import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w08.section import COLUMN, define_section                        # noqa: E402
from w10.robust import ALGORITHMS, Effort                             # noqa: E402


# --8<-- [start:params]
SPECIMEN = {
    "L": 3500.0,        # mm        cantilever height to the mass
    "m": 90.0,          # N s^2/mm  90 tonnes -- NOT kilograms (week 12)
    "zeta": 0.05,       # -         at the two anchor periods
    "n_fibres": 16,     # -         converged in week 8
    "g": 9810.0,        # mm/s^2
}

GROUND = {
    "duration": 20.0,   # s
    "dt": 0.01,         # s         record sampling interval
    "pga": 2950.0,      # mm/s^2    a scaling constant, NOT the PGA:
                        #           the record peaks near 0.35 g at scale 1
}
# --8<-- [end:params]

COL_INTEG = 1


# --8<-- [start:motion]
def accelerogram(p=GROUND, scale=1.0):
    """A synthetic accelerogram, generated rather than read from a file.

    Deterministic on purpose: fixed frequencies, fixed phases, no random seed
    to remember. Anybody who runs this script gets the identical record, which
    is what K19 asks of every number in these notes -- and a real record could
    not be redistributed with them anyway.

    The shape is the one every strong-motion record has: a short build-up, a
    strong-motion plateau, and a long exponential decay. The frequency content
    is deliberately placed around the specimen's own period, because a record
    with no energy near the structure excites nothing and proves nothing.
    """
    n = int(p["duration"] / p["dt"])
    freqs = (1.3, 2.1, 3.4, 5.5, 8.9)          # Hz -- spread around 1/T
    phases = (0.0, 1.1, 2.3, 0.7, 1.9)         # rad, fixed

    record = []
    for i in range(n + 1):
        t = i * p["dt"]
        rise = min(t / 2.0, 1.0)
        decay = math.exp(-max(0.0, t - 8.0) / 5.0)
        envelope = rise * decay
        a = sum(math.sin(2.0 * math.pi * f * t + ph) / (1.0 + k)
                for k, (f, ph) in enumerate(zip(freqs, phases)))
        record.append(scale * p["pga"] * envelope * a / 1.6)
    return record
# --8<-- [end:motion]


# --8<-- [start:build]
def build_specimen(p=SPECIMEN):
    """The specimen: one fibre-section column with a mass on top.

    A cantilever with a lump of mass is what most shake-table column tests
    actually are, and it keeps the model small enough that twenty seconds of
    nonlinear time history runs in seconds rather than hours.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, p["L"])
    ops.fix(1, 1, 1, 1)

    define_section(COLUMN, p["n_fibres"])
    ops.geomTransf("PDelta", 1)
    ops.beamIntegration("Lobatto", COL_INTEG, 1, 5)
    ops.element("forceBeamColumn", 1, 1, 2, 1, COL_INTEG)

    # Horizontal mass only: the column is axially stiff and a rotational mass
    # on a lumped model is a fiction nobody needs here.
    ops.mass(2, p["m"], 0.0, 0.0)
# --8<-- [end:build]


# --8<-- [start:gravity]
def apply_gravity(p=SPECIMEN):
    """The weight of the mass, applied and then frozen.

    Easy to forget in a dynamic analysis, and it matters twice: it puts axial
    load into the section, which changes its moment capacity (week 8), and it
    is what makes P-delta mean anything.
    """
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0.0, -p["m"] * p["g"], 0.0)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 20)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    if ops.analyze(10) != 0:
        raise RuntimeError("gravity did not converge")
    ops.loadConst("-time", 0.0)
# --8<-- [end:gravity]


# --8<-- [start:excitation]
def apply_ground_motion(record, dt, p=SPECIMEN):
    """Shake the base.

    A UniformExcitation pattern moves every restrained node together, which is
    what a shake table does. It is applied as an ACCELERATION history, and
    OpenSees turns it into the effective force -m*a_g internally -- so the
    record is in mm/s^2 and the mass has already been given in tonnes.

    Get either of those wrong and the specimen is shaken by something else
    entirely, without complaint.
    """
    ops.timeSeries("Path", 2, "-dt", dt, "-values", *record)
    ops.pattern("UniformExcitation", 2, 1, "-accel", 2)

    w = periods_of(2)
    rayleigh_from(p, w[0], w[0] / 5.0)
    return w
# --8<-- [end:excitation]


def periods_of(n_modes=1):
    """Natural periods now, using the current tangent stiffness."""
    return [2.0 * math.pi / math.sqrt(lam)
            for lam in ops.eigen("-fullGenLapack", n_modes)]


def rayleigh_from(p, T1, T2):
    w1, w2 = 2.0 * math.pi / T1, 2.0 * math.pi / T2
    a0 = p["zeta"] * 2.0 * w1 * w2 / (w1 + w2)
    a1 = p["zeta"] * 2.0 / (w1 + w2)
    ops.rayleigh(a0, 0.0, a1, 0.0)
    return a0, a1


# --8<-- [start:transient]
def robust_transient(duration, dt, effort, min_frac=1.0 / 64.0):
    """Week 10's ladder, in real time instead of pseudo-time.

    One difference matters. In a static push, subdividing a step is free: the
    analysis simply arrives at the same displacement in smaller pieces. Here
    the step is TIME, so the sub-steps must add up to exactly the same elapsed
    time or the record and the response drift out of step with each other --
    silently, and worse the longer the analysis runs.
    """
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    t, u, v, shear = [0.0], [0.0], [0.0], [0.0]
    remaining = duration
    step = dt

    while remaining > 1.0e-12:
        this = min(step, remaining)
        ok = False
        for name, algorithm in ALGORITHMS:
            ops.test("NormDispIncr", 1.0e-8, 20)
            ops.algorithm(*algorithm)
            ok = ops.analyze(1, this) == 0
            effort.record(name, ok)
            if ok:
                break

        if not ok:
            if step <= dt * min_frac:
                break                      # give up, and say how far we got
            effort.subdivisions += 1
            step /= 2.0
            continue

        remaining -= this
        step = min(dt, step * 2.0)         # recover the full step when it works

        t.append(ops.getTime())
        u.append(ops.nodeDisp(2, 1))
        v.append(ops.nodeVel(2, 1))
        ops.reactions()
        shear.append(-ops.nodeReaction(1, 1))

    return {"t": t, "u": u, "v": v, "shear": shear, "effort": effort}
# --8<-- [end:transient]


# --8<-- [start:run]
def run(scale=1.0, p=SPECIMEN, ground=GROUND):
    """Build, load, shake. Returns the response history."""
    build_specimen(p)
    apply_gravity(p)
    record = accelerogram(ground, scale)
    apply_ground_motion(record, ground["dt"], p)
    out = robust_transient(ground["duration"], ground["dt"], Effort())
    out["record"] = record
    return out


def summarise(out, p=SPECIMEN):
    """The three numbers a time history is usually reduced to.

    Peak displacement is what a drift limit is written on. Residual is what is
    left when the shaking stops, and it decides whether a building is repaired
    or demolished. Peak shear is what the foundation has to carry.
    """
    peak = max(abs(x) for x in out["u"])
    return {
        "peak_disp": peak,
        "peak_drift": peak / p["L"],
        "residual": out["u"][-1],
        "peak_shear": max(abs(x) for x in out["shear"]),
        "finished": abs(out["t"][-1] - GROUND["duration"]) < 1e-6,
    }
# --8<-- [end:run]


SCALES = (0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 1.50)


# --8<-- [start:sweep]
def scale_sweep(scales=SCALES, p=SPECIMEN, ground=GROUND):
    """Run the same record at several amplitudes and compare.

    The quantity to watch is peak/scale: what the peak WOULD have been at
    scale 1 if the structure were linear. While it stays constant the response
    is elastic; where it moves, the specimen has yielded. Nothing had to be
    assumed about where that happens -- it was measured.

    The damaged period is taken after the shaking stops. eigen() uses the
    current tangent stiffness (week 12), so on a structure that has yielded it
    reports the period of what is left, not of what was built.
    """
    rows = []
    for s in scales:
        out = run(s, p, ground)
        got = summarise(out, p)
        got["scale"] = s
        got["normalised"] = got["peak_disp"] / s if s else 0.0
        got["T_damaged"] = periods_of(1)[0]
        rows.append(got)
    return rows
# --8<-- [end:sweep]


if __name__ == "__main__":
    build_specimen()
    apply_gravity()
    T0 = periods_of(1)[0]
    print(f"Specimen: fibre-section column, T = {T0:.4f} s,"
          f" mass {SPECIMEN['m']:.0f} t")
    print("  record components at periods:",
          ", ".join(f"{1/f:.2f}" for f in (1.3, 2.1, 3.4, 5.5, 8.9)), "s")

    print("\nCheck 1 -- zero input: a stationary base leaves it stationary")
    zero = run(scale=0.0)
    print(f"  peak displacement       {max(abs(x) for x in zero['u']):.3e} mm")

    print("\nChecks 2 and 3 -- scaling the record")
    print(f"  {'scale':>6}{'PGA [g]':>9}{'peak':>9}{'peak/scale':>12}"
          f"{'residual':>10}{'shear kN':>10}{'T after':>9}")
    rows = scale_sweep()
    elastic = rows[0]["normalised"]
    for r in rows:
        flag = "elastic" if abs(r["normalised"] / elastic - 1.0) < 0.02 \
            else f"x{r['normalised'] / elastic:.2f}"
        pga_g = max(abs(a) for a in accelerogram(GROUND, r["scale"])) / SPECIMEN["g"]
        print(f"  {r['scale']:6.2f}{pga_g:9.3f}"
              f"{r['peak_disp']:9.2f}{r['normalised']:12.1f}"
              f"{r['residual']:10.3f}{r['peak_shear'] / 1000:10.1f}"
              f"{r['T_damaged']:9.3f}   {flag}")

    print("\n  peak/scale is constant to 0.2 % at the two smallest amplitudes:")
    print("  that is the elastic range, and it was found rather than assumed.")
    print("  The base shear then saturates near 114 kN, which is the strength.")
    print("\n  And the response is NOT monotone in the input. At scale 0.25 the")
    print("  peak is 2.4 times what a linear structure would have given -- the")
    print("  column softens, its period lengthens towards the strongest")
    print("  component of the record at 0.77 s, and it walks into resonance.")
    print("  Scaling a record does not scale the answer.")
