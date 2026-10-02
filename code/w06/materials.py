"""Week 6 - a test rig for one material, with no structure around it.

    python code/w06/materials.py

Up to now every material has been `Elastic`, which is a straight line and
cannot be wrong. From here on materials are the interesting part of the model,
and the first thing to learn is how to look at one on its own.

OpenSees will let you do exactly that. testUniaxialMaterial() takes a material
out of the model and hands you a lever: setStrain() imposes a strain,
getStress() reads the stress back, getTangent() reads the current slope. No
nodes, no elements, no analysis. Whatever the material does, you can see it.

This matters more than it sounds. A material is eight numbers in a manual, and
the difference between two of them is often invisible in the documentation and
obvious in a picture. Ten lines of test rig will save you hours in week 8, when
the material is buried inside a fibre section inside an element and a wrong
answer has three places to hide.

Units: N, mm, s. Stress in MPa.
"""

import sys
from pathlib import Path

import openseespy.opensees as ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --8<-- [start:params]
STEEL = {
    "E": 200000.0,      # MPa    modulus of elasticity
    "fy": 420.0,        # MPa    yield strength
    "b": 0.02,          # -      strain-hardening ratio (2 % of E)
}
STEEL["eps_y"] = STEEL["fy"] / STEEL["E"]      # 0.0021, the yield strain
# --8<-- [end:params]


# --8<-- [start:rig]
def probe(strains, define_material):
    """Impose a strain path on one material and record what it does.

    `define_material` is a function that creates the material with tag 1. It is
    passed in rather than hard-coded so that the same rig tests every material
    in this chapter -- which is the whole point of having a rig.

    The strains are imposed IN ORDER and the material remembers. That is what
    makes this a test of a material rather than of a formula: the stress at a
    given strain depends on how the material got there.
    """
    ops.wipe()
    define_material()
    ops.testUniaxialMaterial(1)

    stress, tangent = [], []
    for eps in strains:
        ops.setStrain(eps)
        stress.append(ops.getStress())
        tangent.append(ops.getTangent())
    return stress, tangent
# --8<-- [end:rig]


# --8<-- [start:materials]
def elastic(p=STEEL):
    """A straight line, for ever. Never yields, never fails."""
    ops.uniaxialMaterial("Elastic", 1, p["E"])


def elastic_pp(p=STEEL):
    """Elastic-perfectly plastic: a straight line to yield, then flat.

    ElasticPP is given the YIELD STRAIN, not the yield stress -- a place where
    it is easy to hand OpenSees 420 where it wanted 0.0021 and get a material
    a thousand times too strong, with no complaint.
    """
    ops.uniaxialMaterial("ElasticPP", 1, p["E"], p["eps_y"])


def hardening(p=STEEL):
    """Bilinear with kinematic hardening: the second slope is b*E.

    Arguments: E, sigmaY, H_iso, H_kin. Setting the isotropic modulus to zero
    and the kinematic one to b*E/(1-b) gives a second branch of slope b*E,
    which is how a hardening ratio is usually quoted.
    """
    h_kin = p["b"] * p["E"] / (1.0 - p["b"])
    ops.uniaxialMaterial("Hardening", 1, p["E"], p["fy"], 0.0, h_kin)
# --8<-- [end:materials]


# --8<-- [start:paths]
def monotonic(eps_max=0.02, n=400):
    """Pull steadily from zero to eps_max."""
    return [eps_max * i / n for i in range(n + 1)]


def load_unload(eps_max=0.01, n=200):
    """Pull to eps_max, then release all the way back to zero strain.

    This is the path that separates the three materials. All of them look
    identical going up; only the way they come down tells them apart.
    """
    up = [eps_max * i / n for i in range(n + 1)]
    down = [eps_max * (1.0 - i / n) for i in range(1, n + 1)]
    return up + down
# --8<-- [end:paths]


# --8<-- [start:residual]
def residual_strain(define_material, eps_max=0.01):
    """The strain left behind after loading to eps_max and unloading to zero
    stress -- the permanent set.

    Unloading to zero STRAIN and unloading to zero STRESS are different
    questions, and this is the second one. It is what a bar keeps after the
    load is taken off.
    """
    path = load_unload(eps_max)
    stress, _ = probe(path, define_material)

    # Walk back down the unloading branch until the stress changes sign.
    for i in range(len(path) - 1, 0, -1):
        if stress[i] <= 0.0 <= stress[i - 1] or stress[i] >= 0.0 >= stress[i - 1]:
            # linear interpolation between the two bracketing points
            s0, s1 = stress[i - 1], stress[i]
            e0, e1 = path[i - 1], path[i]
            if s1 == s0:
                return e1
            return e0 + (0.0 - s0) * (e1 - e0) / (s1 - s0)
    return 0.0
# --8<-- [end:residual]


MATERIALS = (("Elastic", elastic), ("ElasticPP", elastic_pp),
             ("Hardening", hardening))


if __name__ == "__main__":
    p = STEEL
    print(f"Steel: E = {p['E']:.0f} MPa, fy = {p['fy']:.0f} MPa,"
          f" eps_y = {p['eps_y']:.5f}, b = {p['b']}")

    print("\nStress at four strains, MPa")
    probes = [0.001, p["eps_y"], 0.010, 0.020]
    print(f"  {'material':<12}" + "".join(f"{e:>12.5f}" for e in probes))
    for name, define in MATERIALS:
        stress, _ = probe(probes, define)
        print(f"  {name:<12}" + "".join(f"{s:12.1f}" for s in stress))

    print("\nAfter loading to 1 % strain and releasing the load")
    print(f"  {'material':<12}{'residual strain':>18}{'as a fraction':>16}")
    for name, define in MATERIALS:
        r = residual_strain(define, 0.010)
        print(f"  {name:<12}{r:18.6f}{r / 0.010:16.3f}")

    print("\n  Elastic returns to zero: it has no memory, so a bar of it would")
    print("  spring back perfectly however far you stretched it. That is the")
    print("  one thing real steel certainly does not do.")
