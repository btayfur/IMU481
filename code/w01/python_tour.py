"""Week 1 - the four pieces of Python this course actually uses.

    python code/w01/python_tour.py

You have written programs before, so this is not an introduction to
programming. It is a short tour of the four constructs that make up almost
every script in this course, shown doing the job they will be doing for the rest
of the term rather than in the abstract.

    a dictionary     to hold the parameters of a thing
    a function       to turn parameters into an answer
    an if statement  to turn an answer into a decision
    a for loop       to do it for every candidate instead of one

The task is a real one: four steel sections are offered for a 3 m cantilever,
and we want to know which of them are stiff enough to satisfy a deflection
limit of L/250. No OpenSees is needed -- this is arithmetic -- which is exactly
why it is a good place to look at the language on its own.

Units: N, mm, s.
"""

# --8<-- [start:data]
# A dict groups the numbers that belong to ONE thing, and names them. A list
# holds several of those things. Almost every model in this course is described
# by a structure of exactly this shape.
SECTIONS = [
    {"name": "IPE 200", "I": 1.943e7},      # mm^4, strong axis
    {"name": "IPE 240", "I": 3.892e7},
    {"name": "IPE 300", "I": 8.356e7},
    {"name": "IPE 360", "I": 1.627e8},
]

BEAM = {"L": 3000.0, "E": 200000.0, "P": 10000.0}
# --8<-- [end:data]


# --8<-- [start:function]
def tip_deflection(I, L=3000.0, E=200000.0, P=10000.0):
    """Tip deflection of a cantilever under a point load, PL^3/3EI.

    Arguments after the first have DEFAULT values, so a caller who only wants
    to vary the section writes tip_deflection(I) and a caller who wants a
    different span writes tip_deflection(I, L=4500.0). The name in the call
    makes it obvious which argument is which -- worth doing whenever a function
    takes more than two or three.

    The magnitude is returned, not the signed displacement: this function
    answers "how far", and the direction is not in question.
    """
    return P * L ** 3 / (3.0 * E * I)
# --8<-- [end:function]


# --8<-- [start:conditional]
def verdict(delta, L=3000.0, limit_ratio=250.0):
    """Turn a deflection into a decision: does it satisfy L/250?

    An if statement is how a script stops being a calculator and starts being
    an engineer. The comparison is the interesting line; the strings are just
    how it reports itself.
    """
    allowed = L / limit_ratio
    if delta <= allowed:
        return "OK"
    elif delta <= 1.5 * allowed:
        return "marginal"
    else:
        return "too flexible"
# --8<-- [end:conditional]


# --8<-- [start:loop]
def survey(sections=SECTIONS, beam=BEAM):
    """Apply the same two functions to every candidate section.

    This is the loop that matters in this course. Writing the calculation once
    and letting the loop repeat it is what makes a parameter study cost nothing
    -- and from week 5 the thing being repeated is a whole finite element
    model rather than one division.
    """
    results = []
    for section in sections:
        delta = tip_deflection(section["I"], L=beam["L"],
                               E=beam["E"], P=beam["P"])
        results.append({
            "name": section["name"],
            "delta": delta,
            "verdict": verdict(delta, L=beam["L"]),
        })
    return results
# --8<-- [end:loop]


# --8<-- [start:report]
def report(results, beam=BEAM):
    """Print the survey as a table.

    An f-string interpolates a value into text: {r['delta']:8.2f} means "this
    number, in a field eight characters wide, to two decimal places". Lining
    numbers up is not decoration -- a column of figures is read by eye, and a
    ragged one hides the outlier you were looking for.
    """
    allowed = beam["L"] / 250.0
    print(f"Cantilever, L = {beam['L']:.0f} mm, P = {beam['P'] / 1000:.0f} kN")
    print(f"Deflection limit L/250 = {allowed:.2f} mm\n")
    print(f"  {'section':<10}{'delta [mm]':>12}{'verdict':>16}")
    for r in results:
        print(f"  {r['name']:<10}{r['delta']:12.2f}{r['verdict']:>16}")
# --8<-- [end:report]


if __name__ == "__main__":
    found = survey()
    report(found)

    # A loop and a conditional together answer the question that was actually
    # asked, rather than printing a table and leaving it to the reader.
    passing = [r["name"] for r in found if r["verdict"] == "OK"]
    print(f"\n  Acceptable sections: {', '.join(passing)}")
    print(f"  Lightest acceptable: {passing[0]}")
