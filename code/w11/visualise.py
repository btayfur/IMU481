"""Week 11 - looking at the model, and at what it did.

    python code/w11/visualise.py

Two pictures, and they answer different questions.

    plot_model()   is the structure you built the structure you meant?
    plot_defo()    what did it do?

Skipping the first is how a beautifully presented analysis of the wrong
building gets published. It costs one line, and it catches the errors that are
hardest to find any other way -- a node at the wrong coordinate, a member
joined to the storey below, a support missing from one column line. None of
those produce an error message. All of them are obvious in a picture.

opsvis reads the LIVE DOMAIN. There is no data structure to hand it: it asks
OpenSees what nodes and elements exist and where they have moved to. So the
model has to still be in memory, and for a deformed shape it has to still be
deformed -- call ops.wipe() first and you get an empty picture rather than an
error.

Units: N, mm, s.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import opsvis as opsv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w11.pushover import FRAME, capacity_curve                        # noqa: E402


# --8<-- [start:plots]
def draw(sfac=6.0):
    """Draw the model and the deformed shape side by side.

    `sfac` is the displacement magnification. A real 250 mm sway on a 10.5 m
    frame is visible but undramatic; six times makes the mechanism obvious.
    Always say what the factor was -- an unlabelled exaggerated shape has
    misled more than one review meeting.
    """
    fig = plt.figure(figsize=(11, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    opsv.plot_model(node_labels=0, element_labels=0, ax=ax1,
                    local_axes=False, gauss_points=False)
    ax1.set_title("model")

    ax2 = fig.add_subplot(122, projection="3d")
    opsv.plot_defo(sfac=sfac, unDefoFlag=1, ax=ax2)
    ax2.set_title(f"deformed, x{sfac:g}")

    return fig
# --8<-- [end:plots]


if __name__ == "__main__":
    # The push must run first: opsvis reads the domain, so there has to be a
    # deformed structure in it before there is anything to draw.
    roof, shear, _, _ = capacity_curve(FRAME, target=250.0, n_steps=120)
    print(f"pushed to {roof[-1]:.0f} mm, base shear {shear[-1] / 1000:.1f} kN")

    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)
    fig = draw()
    fig.savefig(out / "frame.png", dpi=150, bbox_inches="tight")
    print(f"written to {out / 'frame.png'}")
    print("\n  Run this before every analysis, not only after one.")
