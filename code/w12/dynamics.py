"""Week 12 - mass, damping, periods, and the first analysis with real time in it.

    python code/w12/dynamics.py

Everything so far has been static. Loads were applied through a pseudo-time
that meant nothing, and the structure had no idea how fast anything happened.
This week adds two properties that only matter when it does: MASS, which
resists acceleration, and DAMPING, which removes energy.

The chapter is deliberately linear. There is quite enough new here without
also asking what happens when a material yields, and everything in it can be
checked against a closed form -- which is exactly why it comes before week 13
rather than after.

Read the unit note below before anything else. It is the one week 0 warned
about, and it is about to cost somebody a marked assignment.

Units: N, mm, s. MASS IS IN N*s^2/mm, WHICH IS TONNES.
"""

import math
import sys
from pathlib import Path

import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --8<-- [start:params]
# A single column carrying a lump of mass at the top: the simplest structure
# with a period. Concrete, 400 x 600, one storey high.
SDOF = {
    "L": 3500.0,        # mm      column height
    "E": 30000.0,       # MPa     concrete, roughly 5000*sqrt(fc)
    "I": 7.2e9,         # mm^4    400 x 600 about the strong axis
    "A": 240000.0,      # mm^2
    "m": 90.0,          # N*s^2/mm  = 90 tonnes = 90 000 kg
    "zeta": 0.05,       # -       5 % of critical damping
}
# --8<-- [end:params]


# --8<-- [start:units]
def mass_note(p=SDOF):
    """What the mass unit is, and what entering kilograms costs.

    F = ma fixes the unit of mass once force and length are chosen. In N and
    mm that unit is N*s^2/mm, which happens to equal 1000 kg -- one tonne.

    A mass entered in kilograms is therefore 1000 times too large. Period goes
    as sqrt(m), so every period comes out sqrt(1000) = 31.6 times too long: a
    half-second building reports fifteen seconds. That is far too large to be
    roundoff and far too orderly to be a modelling error, which is exactly why
    it gets misdiagnosed as one.
    """
    k = 3.0 * p["E"] * p["I"] / p["L"] ** 3
    right = 2.0 * math.pi * math.sqrt(p["m"] / k)
    wrong = 2.0 * math.pi * math.sqrt(p["m"] * 1000.0 / k)
    return {"k": k, "T_correct": right, "T_in_kg": wrong,
            "factor": wrong / right}
# --8<-- [end:units]


# --8<-- [start:build]
def build_sdof(p, mass_in_kg=False):
    """A cantilever column with a mass at the top.

    ops.mass() takes one value per degree of freedom, in the same order as
    ops.fix(). Only the horizontal one is given a value here: the column is
    axially very stiff, so vertical inertia contributes nothing at the periods
    that matter, and leaving a rotational mass at zero is standard for a lumped
    model.

    Note that a zero mass on a FREE degree of freedom is legal. It gives that
    direction an infinite frequency, and eigen() will happily return it.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, p["L"])
    ops.fix(1, 1, 1, 1)

    ops.geomTransf("Linear", 1)
    ops.element("elasticBeamColumn", 1, 1, 2, p["A"], p["E"], p["I"], 1)

    m = p["m"] * 1000.0 if mass_in_kg else p["m"]
    ops.mass(2, m, 0.0, 0.0)
    return m
# --8<-- [end:build]


# --8<-- [start:eigen]
def periods(n_modes=1, solver="-fullGenLapack"):
    """Natural periods, in seconds.

    eigen() returns the eigenvalues lambda = omega^2, so the period is
    2*pi/sqrt(lambda). It uses the CURRENT tangent stiffness, which for a
    linear model is the only stiffness there is -- but from week 13 it means
    the period of a damaged structure differs from the period of an intact
    one, and asking at the wrong moment gives the wrong answer.

    The solver is not a detail. The default is Arpack, an ITERATIVE method
    built for large systems, and on a lumped-mass model it has to work with a
    singular mass matrix -- most degrees of freedom carry no mass at all. On a
    model this small it simply gives up:

        ArpackSolver::Error with _saupd info = -9999
        Could not build an Arnoldi factorization.

    '-fullGenLapack' solves the whole generalized problem directly. It is
    correct on small models and, as OpenSees itself warns, very slow on large
    ones. Massless degrees of freedom come back as eigenvalues of about 1e308,
    which is the honest answer -- infinite frequency -- and the reason to ask
    for only as many modes as the model has masses.
    """
    eigenvalues = ops.eigen(solver, n_modes)
    return [2.0 * math.pi / math.sqrt(lam) for lam in eigenvalues]
# --8<-- [end:eigen]


# --8<-- [start:closed_form]
def sdof_closed_form(p):
    """The textbook answer for a cantilever with a tip mass.

        k = 3EI/L^3        T = 2*pi*sqrt(m/k)

    Every dynamic model in this course is checked against something. This is
    the one case where the something is exact.
    """
    k = 3.0 * p["E"] * p["I"] / p["L"] ** 3
    omega = math.sqrt(k / p["m"])
    return {"k": k, "omega": omega, "T": 2.0 * math.pi / omega}
# --8<-- [end:closed_form]


# --8<-- [start:damping]
def rayleigh_damping(p, T1, T2):
    """Rayleigh damping matched to zeta at two periods.

        C = a0*M + a1*K
        a0 = zeta * 2*w1*w2/(w1+w2)      a1 = zeta * 2/(w1+w2)

    The two periods are a CHOICE, and a consequential one. Damping is exactly
    zeta at w1 and w2 and higher everywhere outside them, so a badly chosen
    pair can put ten or twenty per cent damping on modes you meant to leave
    alone -- quietly, and in the direction that makes a structure look safer.
    """
    w1, w2 = 2.0 * math.pi / T1, 2.0 * math.pi / T2
    a0 = p["zeta"] * 2.0 * w1 * w2 / (w1 + w2)
    a1 = p["zeta"] * 2.0 / (w1 + w2)
    ops.rayleigh(a0, 0.0, a1, 0.0)
    return a0, a1


def rayleigh_ratio(a0, a1, T):
    """The damping ratio this Rayleigh pair actually applies at period T."""
    w = 2.0 * math.pi / T
    return 0.5 * (a0 / w + a1 * w)
# --8<-- [end:damping]


# --8<-- [start:free_vibration]
def free_vibration(p, u0=20.0, duration=2.0, dt=0.002):
    """Pull the mass sideways, let go, and watch it ring down.

    The cleanest dynamic verification there is. The decay of successive peaks
    gives the damping ratio back through the logarithmic decrement, and the
    spacing of the peaks gives the period -- both independent of anything the
    model was told.
    """
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    ops.sp(2, 1, u0)                       # impose the initial displacement

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-10, 20)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    if ops.analyze(1) != 0:
        raise RuntimeError("could not impose the initial displacement")
    ops.loadConst("-time", 0.0)
    ops.remove("loadPattern", 1)           # release it: nothing holds it now

    ops.integrator("Newmark", 0.5, 0.25)   # average acceleration, unconditionally stable
    ops.analysis("Transient")

    t, u = [0.0], [ops.nodeDisp(2, 1)]
    for _ in range(int(duration / dt)):
        if ops.analyze(1, dt) != 0:
            break
        t.append(ops.getTime())
        u.append(ops.nodeDisp(2, 1))
    return t, u
# --8<-- [end:free_vibration]


# --8<-- [start:decrement]
def logarithmic_decrement(t, u):
    """Recover the period and the damping ratio from the response alone.

        delta = ln(u_i / u_{i+1})        zeta = delta / sqrt(4*pi^2 + delta^2)

    Nothing here knows what damping was applied. That is the point: this is a
    measurement of the model, not a restatement of its input.
    """
    peaks = [(t[i], u[i]) for i in range(1, len(u) - 1)
             if u[i] > u[i - 1] and u[i] > u[i + 1] and u[i] > 0.0]
    if len(peaks) < 3:
        return None

    period = (peaks[-1][0] - peaks[0][0]) / (len(peaks) - 1)
    ratios = [math.log(peaks[i][1] / peaks[i + 1][1])
              for i in range(len(peaks) - 1)]
    delta = sum(ratios) / len(ratios)
    zeta = delta / math.sqrt(4.0 * math.pi ** 2 + delta ** 2)
    return {"period": period, "delta": delta, "zeta": zeta,
            "n_peaks": len(peaks)}
# --8<-- [end:decrement]


def sdof_period(p=SDOF, mass_in_kg=False):
    """Build and ask. Returns the first period in seconds."""
    build_sdof(p, mass_in_kg)
    return periods(1)[0]


if __name__ == "__main__":
    exact = sdof_closed_form(SDOF)
    print("Cantilever with a tip mass, units N-mm-s")
    print(f"  stiffness  3EI/L^3      {exact['k']:12.1f} N/mm")
    print(f"  mass                    {SDOF['m']:12.1f} N s^2/mm"
          f"  (= {SDOF['m'] * 1000:.0f} kg)")
    print(f"  period, closed form     {exact['T']:12.4f} s")
    print(f"  period, eigen()         {sdof_period():12.4f} s")

    note = mass_note()
    print("\nWhat happens if the mass is entered in kilograms")
    print(f"  correct                 {note['T_correct']:12.4f} s")
    print(f"  mass in kg              {note['T_in_kg']:12.4f} s")
    print(f"  ratio                   {note['factor']:12.4f}"
          f"   (sqrt(1000) = {math.sqrt(1000):.4f})")
    print(f"  eigen() agrees          {sdof_period(mass_in_kg=True):12.4f} s")

    print("\nFree vibration: pull it 20 mm and let go")
    build_sdof(SDOF)
    T1 = periods(1)[0]
    a0, a1 = rayleigh_damping(SDOF, T1, T1 / 5.0)
    print(f"  Rayleigh a0, a1         {a0:12.5f} {a1:.6f}")
    t, u = free_vibration(SDOF)
    got = logarithmic_decrement(t, u)
    print(f"  peaks found             {got['n_peaks']:12d}")
    print(f"  period, measured        {got['period']:12.4f} s"
          f"   (eigen said {T1:.4f})")
    print(f"  damping, measured       {got['zeta']:12.4f}"
          f"   (asked for {SDOF['zeta']:.4f})")

    print("\n  Neither the period nor the damping was read back from what the")
    print("  model was told. Both were measured from how it moved.")
