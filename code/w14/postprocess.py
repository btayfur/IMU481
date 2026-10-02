"""Week 14 - turning a pile of numbers into something somebody can act on.

    python code/w14/postprocess.py

A twenty-second time history at 0.01 s produces two thousand rows for every
quantity recorded, on every node, in every element. Week 13 reduced that to
three numbers and drew four pictures. This chapter is about doing that
deliberately rather than by habit.

Three things it covers, in the order they bite:

    extraction     reading recorder output at scale, and the arithmetic that
                   goes wrong when you difference numbers that came from a file;
    reduction      the small set of scalars an engineer actually reports, and
                   what each one hides;
    presentation   a figure somebody else can read, and a 3D export for the
                   cases where a plot cannot carry it.

There is no new mechanics here at all. That is deliberate: post-processing is
where results get quietly ruined, and it deserves a chapter of its own rather
than a paragraph at the end of one.

Units: N, mm, s.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w13.timehistory import GROUND, SPECIMEN, run                     # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


# --8<-- [start:reduce]
def reduce_history(out, p=SPECIMEN):
    """The scalars a time history is usually reported as -- and what each hides.

    peak        the largest excursion, in either direction. One instant out of
                two thousand, and the one most sensitive to phasing.
    residual    where it finished. Decides repair or demolition, and cannot
                exist in a linear analysis.
    peak_shear  what the foundation has to carry.
    energy      the area swept in the base shear against displacement plane:
                everything the specimen absorbed instead of giving back.
    cycles      how many times it crossed zero. A single large excursion and
                thirty small ones can share a peak and are not the same event.
    """
    u, v = out["u"], out["shear"]
    crossings = sum(1 for i in range(1, len(u)) if u[i - 1] * u[i] < 0.0)
    return {
        "peak": max(abs(x) for x in u),
        "peak_drift": max(abs(x) for x in u) / p["L"],
        "residual": u[-1],
        "peak_shear": max(abs(x) for x in v),
        "energy": hysteretic_energy(u, v),
        "half_cycles": crossings,
    }


def hysteretic_energy(u, shear):
    """Area swept in the force-displacement plane, by the trapezoid rule.

    The same calculation as week 7's dissipated_energy, one level up: there it
    integrated stress over strain for a material, here force over displacement
    for a whole specimen. Units are N*mm = mJ.
    """
    total = 0.0
    for i in range(1, len(u)):
        total += 0.5 * (shear[i] + shear[i - 1]) * (u[i] - u[i - 1])
    return abs(total)
# --8<-- [end:reduce]


# --8<-- [start:envelope]
def drift_envelope(out, n_bins=40):
    """The running maximum: how the damage accumulated, not just its total.

    A peak is a scalar and tells you nothing about when it happened. The
    envelope shows whether the specimen was damaged early and then rode it out,
    or crept up steadily -- two very different events with the same headline
    number.
    """
    t, u = out["t"], out["u"]
    step = max(1, len(t) // n_bins)
    running, envelope = 0.0, []
    for i in range(0, len(t), step):
        running = max(running, max(abs(x) for x in u[max(0, i - step):i + 1]))
        envelope.append((t[i], running))
    return envelope
# --8<-- [end:envelope]


# --8<-- [start:differencing]
def round_to(x, digits):
    """Round to `digits` significant figures, as a recorder file would."""
    if x == 0.0:
        return 0.0
    mag = math.floor(math.log10(abs(x)))
    factor = 10.0 ** (digits - 1 - mag)
    return round(x * factor) / factor


def velocity_error(out, digits=16):
    """What it costs to RECONSTRUCT a quantity the model already computed.

    The model carries velocity as a state variable; nodeVel returns it. It is
    also tempting to take it from the displacement record by differencing,
    especially when the record is all you kept -- and this function prices
    that choice, three ways.

    Two error sources are separated deliberately, because they behave quite
    differently and the obvious suspect is not the culprit.

    ROUNDING, from writing the record at finite precision: shrinks as `digits`
    grows, exactly as you would expect.

    TRUNCATION, from the difference formula itself: does NOT shrink with
    precision at all. A backward difference is first-order accurate, so its
    error is proportional to the time step, and at dt = 0.01 s that dominates
    everything else. A central difference is second-order and roughly eleven
    times better here for one extra line of code.

    Neither is a reason to difference at all when nodeVel is available. This
    exists to show what it costs when it is not.
    """
    t, u, v = out["t"], out["u"], out["v"]
    u_file = [round_to(x, digits) for x in u]
    peak = max(abs(x) for x in v)

    backward = max(abs((u_file[i] - u_file[i - 1]) / (t[i] - t[i - 1]) - v[i])
                   for i in range(1, len(t)) if t[i] > t[i - 1])
    central = max(abs((u_file[i + 1] - u_file[i - 1]) / (t[i + 1] - t[i - 1])
                      - v[i])
                  for i in range(1, len(t) - 1) if t[i + 1] > t[i - 1])

    return {
        "worst_disp_error": max(abs(a - b) for a, b in zip(u, u_file)),
        "backward": backward,
        "central": central,
        "backward_fraction": backward / peak,
        "central_fraction": central / peak,
        "peak_velocity": peak,
    }
# --8<-- [end:differencing]


# --8<-- [start:export]
def write_csv(out, path=None):
    """One tidy table, one row per step, ready for anything else.

    Comma-separated, one header line, full precision. Not because anybody
    reads sixteen digits, but because this file is an INTERMEDIATE -- whatever
    reads it next may difference it, and week 3 explained what that costs.
    """
    path = path or (OUT / "response.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("time_s,disp_mm,vel_mm_s,base_shear_N\n")
        for t, u, v, s in zip(out["t"], out["u"], out["v"], out["shear"]):
            f.write(f"{t!r},{u!r},{v!r},{s!r}\n")
    return path
# --8<-- [end:export]


def report(out, p=SPECIMEN):
    """The whole reduction, as an engineer would write it down."""
    r = reduce_history(out, p)
    return (f"  peak displacement   {r['peak']:10.2f} mm"
            f"   ({r['peak_drift'] * 100:.2f} % drift)\n"
            f"  residual            {r['residual']:10.2f} mm\n"
            f"  peak base shear     {r['peak_shear'] / 1000:10.1f} kN\n"
            f"  energy absorbed     {r['energy'] / 1e6:10.2f} kN m\n"
            f"  zero crossings      {r['half_cycles']:10d}")


if __name__ == "__main__":
    out = run(scale=1.0)

    print("Week 13's specimen, reduced")
    print(report(out))

    print("\nReconstructing the velocity instead of asking for it")
    print(f"  {'digits':>8}{'disp err':>12}{'backward':>12}{'central':>12}")
    for digits in (4, 6, 8, 12):
        d = velocity_error(out, digits)
        print(f"  {digits:8d}{d['worst_disp_error']:12.2e}"
              f"{d['backward_fraction']:11.2%}{d['central_fraction']:12.2%}")
    print("  The rounding error falls with every digit. The differencing")
    print("  error does not move at all -- it is the FORMULA, not the file.")
    print("  A backward difference is first order; a central one is second,")
    print("  and eleven times better here for one more line of code.")
    print("  Better still: nodeVel already had it, exactly.")

    path = write_csv(out)
    print(f"\n  full-precision table written to {path.name}"
          f" ({len(out['t'])} rows)")

    env = drift_envelope(out)
    half = next(t for t, e in env if e > 0.5 * max(x for _, x in env))
    print(f"\n  half the eventual peak was reached by t = {half:.1f} s"
          f" of {GROUND['duration']:.0f} s")
    print("  A peak is a scalar. When it happened is a different question,")
    print("  and the envelope is what answers it.")
