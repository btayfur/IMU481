"""S2 - the OpenSeesPy commands this course actually uses, counted from source.

    python code/s2/commands.py

The OpenSeesPy interface has a few hundred functions. A reference that lists
all of them is the manual, and the manual already exists. What is missing is
the much shorter list you need in order to read every script in this course --
and, more usefully, the order those commands have to be called in.

So this file does not contain a list. It COUNTS one, by reading every script
under code/ and recording every ops.* call in it. Two consequences worth
having: the reference cannot drift away from the code, and a command that
appears in the notes but in no script shows up immediately as an error.

Units: N, mm, s.
"""

import ast
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CODE = Path(__file__).resolve().parent.parent


# --8<-- [start:scan]
def scan(root=CODE):
    """Every ops.<name>( call in every script, with the weeks it appears in.

    Parsed with `ast` rather than matched with a regular expression. A regex
    would also find ops.node inside a string or a comment, and this file's own
    docstring would poison its own result -- which is a small illustration of
    a general point: parse the language, do not pattern-match it.
    """
    found = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "commands.py":
            continue
        where = path.parent.name
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if (isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "ops"):
                found.setdefault(fn.attr, set()).add(where)
    return found
# --8<-- [end:scan]


# --8<-- [start:groups]
# The commands, grouped by WHEN they are called rather than by what they are.
# This ordering is the actual content of the reference: OpenSees has no
# declaration phase, so the sequence below is the program, and getting it out
# of order is a large fraction of all beginner errors.
STAGES = [
    ("1. start clean", [
        "wipe", "model",
    ]),
    ("2. geometry", [
        "node", "fix", "equalDOF", "rigidDiaphragm", "mass",
    ]),
    ("3. materials and sections", [
        "uniaxialMaterial", "nDMaterial", "section", "patch", "layer",
        "fiber", "beamIntegration", "geomTransf",
    ]),
    ("4. elements", [
        "element",
    ]),
    ("5. loads", [
        "timeSeries", "pattern", "load", "eleLoad", "sp", "remove",
    ]),
    ("6. the analysis object", [
        "system", "numberer", "constraints", "test", "algorithm",
        "integrator", "analysis",
    ]),
    ("7. run it", [
        "analyze", "eigen", "loadConst", "wipeAnalysis", "setTime",
        "rayleigh",
    ]),
    ("8. ask for results", [
        "reactions", "nodeDisp", "nodeVel", "nodeAccel", "nodeReaction",
        "nodeCoord", "nodeMass", "eleResponse", "eleNodes", "recorder",
        "getNodeTags", "getEleTags", "getTime", "getFixedDOFs",
        "printModel",
    ]),
    ("9. test a material on its own", [
        "testUniaxialMaterial", "setStrain", "getStress", "getTangent",
        "getStrain",
    ]),
]


def coverage(found=None):
    """Which staged commands the course uses, and which it uses unstaged.

    The second list is the interesting one: anything in it is a command the
    scripts rely on that this reference has not placed in the build order,
    which means the reference is incomplete rather than the script wrong.
    """
    found = scan() if found is None else found
    staged = {c for _, cs in STAGES for c in cs}
    used = set(found)
    return {
        "used": sorted(used),
        "staged_and_used": sorted(used & staged),
        "staged_but_unused": sorted(staged - used),
        "used_but_unstaged": sorted(used - staged),
    }
# --8<-- [end:groups]


def by_stage(found=None):
    """The reference table: stage, command, and where in the course it turns up."""
    found = scan() if found is None else found
    table = []
    for stage, commands in STAGES:
        rows = []
        for c in commands:
            weeks = sorted(found.get(c, ()))
            rows.append((c, weeks))
        table.append((stage, rows))
    return table


def week_order(where):
    """Sort key that puts w01..w14 before s1..s4."""
    m = re.match(r"w(\d+)", where)
    return (0, int(m.group(1))) if m else (1, where)


if __name__ == "__main__":
    found = scan()
    cov = coverage(found)

    print("The OpenSeesPy commands this course actually uses\n")
    for stage, rows in by_stage(found):
        print(f"  {stage}")
        for command, weeks in rows:
            if not weeks:
                continue
            ws = ", ".join(sorted(weeks, key=week_order)[:6])
            more = "" if len(weeks) <= 6 else f" +{len(weeks) - 6}"
            print(f"    {command:24s} {ws}{more}")
        print()

    print(f"  {len(cov['used'])} distinct commands across all scripts.")
    print(f"  {len(cov['staged_and_used'])} of them are placed in the "
          f"build order above.")
    if cov["used_but_unstaged"]:
        print(f"  Not yet placed: {', '.join(cov['used_but_unstaged'])}")
    if cov["staged_but_unused"]:
        print(f"  Listed for completeness, used by no script: "
              f"{', '.join(cov['staged_but_unused'])}")
    print("\n  That last line matters. A reference is only trustworthy if")
    print("  something checks it against the code, and this is that check.")
