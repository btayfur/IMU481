"""Week 7 - what materials do when the load comes back the other way.

    python code/w07/cyclic.py

Week 6 pulled materials in one direction and released them once. Under an
earthquake a member is pushed and pulled dozens of times, and everything that
matters about a material - how much energy it absorbs, how quickly it weakens,
whether it survives at all - is invisible until you do that.

Two materials are added here, and the reason for each is a shape:

    Steel02  rounds the corner at yield instead of turning it sharply,
             because real steel does, and because a sharp corner is hard
             for the solver to get round (week 10).
    Concrete01  carries almost nothing in tension and crushes in compression,
             which makes its loop nothing like steel's.

The reusable idea is the CYCLIC PROTOCOL: a list of strains describing a
loading history, generated rather than typed. It is the same idea as the tag
formula in week 5, applied to time instead of space.

Units: N, mm, s. Stress in MPa.
"""

import sys
from pathlib import Path

import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w06.materials import STEEL, probe                       # noqa: E402


# --8<-- [start:params]
CONCRETE = {
    "fc": 30.0,          # MPa   compressive strength, POSITIVE magnitude
    "eps_c0": 0.002,     # -     strain at peak stress
    "fcu": 6.0,          # MPa   crushing strength, positive magnitude
    "eps_cu": 0.0035,    # -     strain at crushing
}
# --8<-- [end:params]


# --8<-- [start:steel02]
def steel02(p=STEEL):
    """Giuffre-Menegotto-Pinto steel: a rounded yield corner.

    Arguments: fy, E0, b, then R0, cR1, cR2 which control how sharp the corner
    is. R0 = 18 with 0.925 and 0.15 are the values almost everyone uses; they
    are not physical constants so much as a shape that fits test data.

    Rounding matters twice over. Real steel does not turn a corner - the
    Bauschinger effect softens it as soon as the load reverses. And a sharp
    corner is a discontinuity in the tangent, which is exactly what makes a
    Newton solver overshoot and fail to converge in week 10.
    """
    ops.uniaxialMaterial("Steel02", 1, p["fy"], p["E"], p["b"],
                         18.0, 0.925, 0.15)
# --8<-- [end:steel02]


# --8<-- [start:concrete01]
def concrete01(p=CONCRETE):
    """Kent-Scott-Park concrete with no tensile strength.

    The arguments are written negative because that is the convention, and
    every reference you will read insists on it. Test it, though, and you find
    Concrete01 does not care: it takes the magnitudes and applies the
    compression sign itself, so positive arguments give an identical material.
    Exercise 7.1 is that test, and the point of it is that a warning repeated
    everywhere is still worth checking once yourself.

    What DOES bite is the sign of the strain you impose and the stress you read
    back. Concrete01 has no tensile strength at all, so interrogate it at a
    positive strain and it returns zero -- not an error, not a warning, just a
    material that appears to have vanished.
    """
    ops.uniaxialMaterial("Concrete01", 1,
                         -p["fc"], -p["eps_c0"],
                         -p["fcu"], -p["eps_cu"])
# --8<-- [end:concrete01]


# --8<-- [start:protocol]
def cyclic_protocol(amplitudes, cycles_each=2, points_per_cycle=80):
    """Build a symmetric strain history from a list of amplitudes.

    Each amplitude is applied `cycles_each` times before moving to the next --
    which is how a real loading protocol is written, because a material that
    survives one cycle at an amplitude may not survive three.

    Returned as a plain list of strains, so it can be fed straight to the rig.
    """
    path = [0.0]
    for amp in amplitudes:
        for _ in range(cycles_each):
            quarter = points_per_cycle // 4
            for i in range(1, quarter + 1):          # 0 -> +amp
                path.append(amp * i / quarter)
            for i in range(1, 2 * quarter + 1):      # +amp -> -amp
                path.append(amp * (1.0 - 2.0 * i / (2 * quarter)))
            for i in range(1, quarter + 1):          # -amp -> 0
                path.append(-amp * (1.0 - i / quarter))
    return path
# --8<-- [end:protocol]


# --8<-- [start:energy]
def dissipated_energy(strains, stresses):
    """Area enclosed by the stress-strain loop, per unit volume.

    The trapezoidal integral of stress with respect to strain. Over a closed
    cycle this is the energy the material turned into heat and damage instead
    of giving back -- which is the whole reason a ductile structure survives an
    earthquake and a brittle one does not.

    Units: MPa x (mm/mm) = N mm per mm^3.
    """
    total = 0.0
    for i in range(1, len(strains)):
        d_eps = strains[i] - strains[i - 1]
        total += 0.5 * (stresses[i] + stresses[i - 1]) * d_eps
    return total
# --8<-- [end:energy]


def peak_stress(stresses):
    """The largest tension and compression reached."""
    return max(stresses), min(stresses)


if __name__ == "__main__":
    from w06.materials import elastic_pp, hardening

    print("Cyclic response, three steel models")
    protocol = cyclic_protocol([0.004, 0.008, 0.012], cycles_each=2)
    print(f"  protocol: {len(protocol)} strain points,"
          f" amplitudes 0.4 %, 0.8 %, 1.2 %, two cycles each")

    print(f"\n  {'material':<12}{'max tension':>14}{'max compr.':>14}"
          f"{'energy':>14}")
    for name, define in (("ElasticPP", elastic_pp),
                         ("Hardening", hardening),
                         ("Steel02", steel02)):
        stress, _ = probe(protocol, define)
        hi, lo = peak_stress(stress)
        w = dissipated_energy(protocol, stress)
        print(f"  {name:<12}{hi:14.1f}{lo:14.1f}{w:14.3f}")

    print("\nConcrete01, pushed into compression and released")
    path = [-0.0005 * i for i in range(9)] + [-0.004 + 0.0005 * i
                                              for i in range(1, 9)]
    stress, _ = probe(path, concrete01)
    print(f"  {'strain':>10}{'stress [MPa]':>14}")
    for e, s in list(zip(path, stress))[:9]:
        print(f"  {e:10.4f}{s:14.2f}")
    print("  ... released ...")
    print(f"  {path[-1]:10.4f}{stress[-1]:14.2f}")
    print("\n  Concrete01 carries nothing in tension: pull it and the stress")
    print("  is zero at every strain, which is why the last value is 0.")
