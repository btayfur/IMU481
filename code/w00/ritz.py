"""Week 0 - an approximate method is a small system of linear equations.

    python code/w00/ritz.py

The cantilever of week 1, now carrying a uniform load, solved twice: exactly,
by integrating the beam equation, and approximately, by the Rayleigh-Ritz
method with one, two and three assumed shapes.

Running it teaches three things the rest of the course relies on:

* an approximate method turns a differential equation into K a = F -- the same
  kind of problem every finite element program solves, OpenSees included;
* with enough of the right shapes, the approximation stops being approximate;
* a deflection can be exact while the bending moment taken from the same
  approximation is badly wrong, because the moment is a second derivative.

Sign convention, for this file only: w is measured DOWNWARD, in the direction
of the load, so every number printed is positive. OpenSees measures along the
global y axis, upward, and would print the same deflections negative.

Units: N, mm, s.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w01.environment_check import CANTILEVER                     # noqa: E402

# --8<-- [start:params]
# The week 1 cantilever, with its tip load replaced by a uniform one.
BEAM = {
    "L": CANTILEVER["L"],     # mm      3000, span
    "E": CANTILEVER["E"],     # MPa     200000, steel
    "I": CANTILEVER["I"],     # mm^4    8.356e7, IPE 300
    "q": 10.0,                # N/mm    = 10 kN/m, uniform, downward
}
# --8<-- [end:params]


# --8<-- [start:exact]
def exact(p, x):
    """The exact solution: EI w'''' = q integrated four times, with the four
    constants fixed by w(0) = 0 and w'(0) = 0 at the wall and by zero moment
    and zero shear at the free end."""
    L, EI, q = p["L"], p["E"] * p["I"], p["q"]
    return {
        "w":     q * x**2 * (6*L**2 - 4*L*x + x**2) / (24*EI),   # deflection
        "theta": q * x * (3*L**2 - 3*L*x + x**2) / (6*EI),       # slope
        "M":     q * (L - x)**2 / 2,                             # hogging moment
        "V":     q * (L - x),                                    # shear
    }
# --8<-- [end:exact]


# --8<-- [start:ritz]
def ritz(p, n_terms, P_tip=0.0):
    """Rayleigh-Ritz with the assumed shapes (x/L)^2, (x/L)^3, ...

    Every one of these shapes has zero deflection and zero slope at x = 0, so
    each already respects the wall. The method only has to decide HOW MUCH of
    each shape to use: the amplitudes a_i, in mm. Making the total potential
    energy stationary with respect to every a_i gives K a = F, with

        K_ij = integral of EI phi_i'' phi_j'' dx     (stiffness)
        F_i  = integral of q phi_i dx + P phi_i(L)   (load)

    For powers of x both integrals are known in closed form, so they are
    written out rather than computed numerically. Every shape equals one at
    the tip, so a tip load P simply adds P to every F_i.
    """
    L, EI, q = p["L"], p["E"] * p["I"], p["q"]
    powers = np.arange(2, n_terms + 2)              # 2, 3, ..., n_terms + 1
    K = np.zeros((n_terms, n_terms))
    F = np.zeros(n_terms)
    for a, i in enumerate(powers):
        F[a] = q * L / (i + 1) + P_tip
        for b, j in enumerate(powers):
            K[a, b] = EI * i*(i-1) * j*(j-1) / ((i + j - 3) * L**3)
    amplitudes = np.linalg.solve(K, F)
    return powers, amplitudes
# --8<-- [end:ritz]


# --8<-- [start:evaluate]
def evaluate(p, powers, amplitudes, x):
    """Deflection, slope and moment of an assumed-shape solution at x.

    The moment is EI times the SECOND derivative of the assumed shape. Each
    differentiation loses a power of x, so a one-term parabola has a constant
    moment -- whatever the load is doing.
    """
    L, EI = p["L"], p["E"] * p["I"]
    s = np.asarray(x, dtype=float) / L
    w = sum(a * s**i for i, a in zip(powers, amplitudes))
    theta = sum(a * i * s**(i - 1) / L for i, a in zip(powers, amplitudes))
    M = EI * sum(a * i*(i-1) * s**(i - 2) / L**2 for i, a in zip(powers, amplitudes))
    return {"w": w, "theta": theta, "M": M}
# --8<-- [end:evaluate]


def tip_values(p, n_terms, P_tip=0.0):
    """The three numbers the chapter compares: tip deflection, tip rotation
    and the moment at the wall."""
    powers, amps = ritz(p, n_terms, P_tip)
    tip = evaluate(p, powers, amps, p["L"])
    wall = evaluate(p, powers, amps, 0.0)
    return {"w_tip": float(tip["w"]), "theta_tip": float(tip["theta"]),
            "M_wall": float(wall["M"]), "amplitudes": amps}


def exact_values(p):
    tip, wall = exact(p, p["L"]), exact(p, 0.0)
    return {"w_tip": tip["w"], "theta_tip": tip["theta"], "M_wall": wall["M"]}


# --8<-- [start:compare]
def compare(p=BEAM, terms=(1, 2, 3)):
    """One row per approximation, each quantity also as a fraction of exact."""
    ref = exact_values(p)
    print(f"Cantilever, L = {p['L']:.0f} mm, q = {p['q']:.0f} N/mm, "
          f"w measured downward")
    print("          tip deflection      tip rotation       wall moment")
    print("  terms     [mm]  /exact       [rad]  /exact     [kNm]  /exact")
    for n in terms:
        r = tip_values(p, n)
        print(f"  {n:<5d} {r['w_tip']:7.4f} {r['w_tip']/ref['w_tip']:7.4f}"
              f"  {r['theta_tip']:10.6f} {r['theta_tip']/ref['theta_tip']:7.4f}"
              f"  {r['M_wall']/1e6:8.3f} {r['M_wall']/ref['M_wall']:7.4f}")
    print(f"  exact {ref['w_tip']:7.4f}          {ref['theta_tip']:10.6f}"
          f"          {ref['M_wall']/1e6:8.3f}")
# --8<-- [end:compare]


if __name__ == "__main__":
    compare()
