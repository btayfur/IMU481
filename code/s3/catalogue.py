"""S3 - what materials this OpenSees actually has, sorted into a tree.

    python code/s3/catalogue.py

The OpenSees material library has grown for thirty years by accretion. The
online manual lists well over a hundred entries with no organising principle
beyond the order they were contributed in, several are documented but absent
from any given build, and a few are present but do not do what their name
promises. Choosing from that list is a real difficulty, and it is not one the
documentation helps with.

So this file does two things the manual does not.

First, it ASKS THE BUILD rather than reading a list. Every name below is
constructed with no arguments at all, and OpenSees is made to say which of two
things is wrong: an unknown material reports "material type X is unknown",
while a known one reports "Invalid #args". The distinction is only visible on
the C-level stderr stream, so the probe captures file descriptor 2 rather than
sys.stderr. The result is a catalogue that is true of the interpreter you are
running, not of somebody's documentation.

Second, it groups them by WHAT THEY MODEL, so the choice becomes a walk down a
tree rather than a search through a list.

Units: N, mm, s.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openseespy.opensees as ops                                   # noqa: E402


# --8<-- [start:probe]
def _stderr_of(fn):
    """Run `fn` and return whatever it wrote to the real stderr stream.

    OpenSees is a C++ library. Its warnings go to file descriptor 2 directly
    and never pass through sys.stderr, so contextlib.redirect_stderr cannot
    see them. Capturing them means swapping the descriptor itself.
    """
    fd = 2
    saved = os.dup(fd)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), fd)
        try:
            fn()
        except Exception:
            pass                               # the message is the result
        finally:
            sys.stderr.flush()
            os.dup2(saved, fd)
            os.close(saved)
        tmp.seek(0)
        return tmp.read().decode("utf-8", "replace")


def available(kind, name):
    """Is `name` compiled into THIS build? Ask it, do not look it up.

    Called with no constructor arguments on purpose. A material that exists
    then complains about the argument count; one that does not exist complains
    about the name. Those are different sentences, and telling them apart is
    the whole trick.
    """
    add = ops.uniaxialMaterial if kind == "uniaxial" else ops.nDMaterial
    msg = _stderr_of(lambda: add(name, 999999))
    if "is unknown" in msg or "not recognized" in msg:
        return False
    return True
# --8<-- [end:probe]


# --8<-- [start:tree]
# The tree. Each branch answers one question about the physics, and the
# question is always "what does this material have to be able to do?" -- not
# "what is it made of". A steel brace that needs to buckle and a concrete
# fibre that needs to crush belong to the same branch, because both need
# strength loss.
UNIAXIAL = {
    "elastic and piecewise linear": [
        "Elastic", "ElasticPP", "ElasticPPGap", "ENT", "ElasticMultiLinear",
        "MultiLinear", "Hysteretic",
    ],
    "metals: yield, then harden": [
        "Steel01", "Steel02", "Steel4", "Hardening", "RambergOsgoodSteel",
        "ReinforcingSteel", "SteelMPF", "Dodd_Restrepo", "UVCuniaxial",
    ],
    "concrete: crush, crack, and lose strength": [
        "Concrete01", "Concrete02", "Concrete04", "Concrete06", "Concrete07",
        "ConcreteCM", "ConfinedConcrete01", "Concrete01WithSITC",
    ],
    "degradation: strength and stiffness lost with cycling": [
        "Pinching4", "BWBN", "ModIMKPeakOriented", "ModIMKPinching",
        "ModIMKBilin", "Clough", "SAWS", "Bilin",
    ],
    "rate dependent: force depends on velocity": [
        "Viscous", "ViscousDamper", "BilinearOilDamper", "Maxwell",
    ],
    "wrappers: modify another material rather than define one": [
        "Parallel", "Series", "MinMax", "InitStrain", "InitStress", "Fatigue",
        "PathIndependent",
    ],
    "contact, impact and cables": [
        "ImpactMaterial", "HyperbolicGapMaterial", "Cable", "ElasticBilin",
    ],
}

ND = {
    "elastic continua": [
        "ElasticIsotropic", "ElasticOrthotropic",
    ],
    "metal plasticity: pressure independent": [
        "J2Plasticity", "PlaneStressSimplifiedJ2",
    ],
    "soil and rock: strength depends on confining pressure": [
        "DruckerPrager", "PressureIndependMultiYield",
        "PressureDependMultiYield", "PressureDependMultiYield02",
        "ManzariDafalias", "PM4Sand", "PM4Silt", "CycLiqCP",
    ],
    "concrete continua": [
        "PlasticDamageConcrete3d", "PlaneStressUserMaterial", "FSAM",
        "ConcreteMcftNonLinear5", "Damage2p",
    ],
    "dimension wrappers: project a 3D law onto an element": [
        "PlaneStrain", "PlaneStress", "PlateFiber", "BeamFiber",
        "InitStressNDMaterial",
    ],
    "contact": [
        "ContactMaterial2D", "ContactMaterial3D",
    ],
}
# --8<-- [end:tree]


def survey():
    """Walk the tree and record what is present. Returns {kind: {branch: {}}}."""
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 3)
    out = {}
    for kind, tree in (("uniaxial", UNIAXIAL), ("nD", ND)):
        out[kind] = {}
        for branch, names in tree.items():
            out[kind][branch] = {n: available(kind, n) for n in names}
    return out


def totals(result):
    """(present, listed) over the whole survey."""
    present = listed = 0
    for tree in result.values():
        for branch in tree.values():
            listed += len(branch)
            present += sum(branch.values())
    return present, listed


# --8<-- [start:levels]
def constructs(kind, name, args):
    """Level 2: does it accept a plausible set of parameters and build?

    A name can be recognised by the parser and still be unusable -- because it
    needs an external subroutine, because the build lacks a dependency, or
    because the argument list in the manual belongs to a different version.
    """
    add = ops.uniaxialMaterial if kind == "uniaxial" else ops.nDMaterial
    msg = _stderr_of(lambda: add(name, 999998, *args))
    return msg.strip() == ""


def tension_carried(tag, strains=(0.00005, 0.0001, 0.0002, 0.0005)):
    """Level 3: does it BEHAVE the way the name implies? Only a rig can say.

    This is week 6's material test rig, used here for a different purpose:
    not to plot a curve, but to ask a yes/no question about a material whose
    documentation does not make the answer obvious.

    Concrete01 and Concrete02 both pass levels 1 and 2, are both named for the
    same material, and differ on whether concrete can carry tension at all.
    Nothing but running them tells you which is which.
    """
    ops.testUniaxialMaterial(tag)
    out = []
    for e in strains:
        ops.setStrain(e)
        out.append(ops.getStress())
    return out
# --8<-- [end:levels]


def three_levels():
    """The Concrete01 / Concrete02 comparison, as numbers."""
    ops.wipe()
    ops.model('basic', '-ndm', 1, '-ndf', 1)
    ops.uniaxialMaterial('Concrete01', 1, -30.0, -0.002, -6.0, -0.006)
    ops.uniaxialMaterial('Concrete02', 2, -30.0, -0.002, -6.0, -0.006,
                         0.1, 3.0, 3000.0)
    return {
        "Concrete01": tension_carried(1),
        "Concrete02": tension_carried(2),
    }


# --8<-- [start:choose]
# The decision, written as questions rather than as a list of names. Each
# answer eliminates a branch, and the leaf names the smallest material that
# still does the job. "Smallest" matters: every extra parameter is another
# number you cannot justify, and week 7 showed what an unjustified parameter
# does to a result.
CHOOSE_UNIAXIAL = [
    ("Does it ever leave the elastic range?",
     "no", "Elastic -- and check afterwards that the strain stayed inside it"),
    ("Does it need to carry compression only, or tension only?",
     "yes", "ENT for no-tension, ElasticPPGap for a gap that closes"),
    ("Is it steel that yields and hardens, without losing strength?",
     "yes", "Steel01 if a corner is acceptable, Steel02 if the cyclic "
            "transition matters -- see week 7"),
    ("Is it concrete, so that it crushes and loses strength?",
     "yes", "Concrete01 for a first model, Concrete02 to get tension, "
            "Concrete04 for a smooth Popovics curve"),
    ("Does it lose strength with repeated cycles at the same amplitude?",
     "yes", "Pinching4 or one of the ModIMK family -- these are the only "
            "branch that models cyclic deterioration"),
    ("Does the force depend on how fast it moves?",
     "yes", "Viscous or ViscousDamper -- and note this makes a static "
            "analysis meaningless"),
    ("Do you need an existing material to stop at a strain limit?",
     "yes", "MinMax, which wraps another material rather than replacing it"),
]
# --8<-- [end:choose]


if __name__ == "__main__":
    result = survey()
    present, listed = totals(result)

    print("What this build actually has\n")
    for kind in ("uniaxial", "nD"):
        print(f"  {kind}Material")
        for branch, names in result[kind].items():
            have = [n for n, ok in names.items() if ok]
            miss = [n for n, ok in names.items() if not ok]
            print(f"    {branch}")
            print(f"      present ({len(have)}): {', '.join(have) or '-'}")
            if miss:
                print(f"      ABSENT  ({len(miss)}): {', '.join(miss)}")
        print()

    print(f"  {present} of {listed} probed names are in this build.")
    print("  The absent ones are all documented online. That is the point of")
    print("  probing rather than reading: the manual describes the union of")
    print("  every build that has ever existed.")

    print("\nRecognised is not the same as usable, and usable is not the same")
    print("as correct. Three levels, and only the third needs a test rig:")
    lv = three_levels()
    print(f"  {'strain':>10}{'Concrete01':>14}{'Concrete02':>14}")
    for i, e in enumerate((0.00005, 0.0001, 0.0002, 0.0005)):
        print(f"  {e:10.5f}{lv['Concrete01'][i]:13.3f} "
              f"{lv['Concrete02'][i]:13.3f}")
    print("  Both are named for the same material and both build without a")
    print("  murmur. One carries no tension at all -- not a little, exactly")
    print("  none -- and the other cracks at 3 MPa and softens. The name does")
    print("  not tell you which, and neither does the fact that it ran.")

    print("\nChoosing one, as a sequence of questions")
    for i, (q, _, leaf) in enumerate(CHOOSE_UNIAXIAL, start=1):
        print(f"  {i}. {q}")
        print(f"       -> {leaf}")
