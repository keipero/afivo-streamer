#!/usr/bin/env python3
"""Make a GIF movie of an axisymmetric afivo-streamer run.

Shows electron density and electric field side by side, mirrored about the
symmetry axis, with a fixed color scale over all frames.

Usage (from a program folder such as programs/standard_2d):
    python3 ../../tools/make_movie.py output/streamer_cyl -o plots/streamer_cyl.gif

Requires afivo/tools/silo_to_raw (build it with `make` in afivo/tools).
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

AFIVO_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "afivo", "tools")

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("basename", help="Output basename, e.g. output/streamer_cyl")
p.add_argument("-o", "--output", default="movie.gif", help="Output GIF file")
p.add_argument("-r_max", type=float, default=4e-3,
               help="Max radius to show (m)")
p.add_argument("-z_max", type=float, default=None,
               help="Max z to show (m), default: full domain")
p.add_argument("-fps", type=int, default=6, help="Frames per second")
p.add_argument("-variables", nargs=2, default=["e", "electric_fld"],
               help="Two variables to show")
args = p.parse_args()

files = sorted(glob.glob(args.basename + "_[0-9]*.silo"))
if not files:
    sys.exit(f"No silo files found for {args.basename}")


def load(silo, var, tmpdir):
    """Convert one variable of one silo file to a uniform grid."""
    npz = os.path.join(tmpdir, "tmp.npz")
    r_max = [args.r_max, args.z_max if args.z_max else 1e10]
    subprocess.run([sys.executable, os.path.join(AFIVO_TOOLS, "plot_raw_data.py"),
                    silo, "-variable", var, "-r_max", *map(str, r_max),
                    "-save_npz", npz, "-q"], check=True)
    d = np.load(npz)
    # uniform_data has shape (n_r, n_z); arr_0 = r coords, arr_1 = z coords
    return d["uniform_data"], d["arr_0"], d["arr_1"], float(d["time"])


frames = []
with tempfile.TemporaryDirectory() as tmp:
    for i, f in enumerate(files):
        print(f"\rReading {i+1}/{len(files)}", end="", flush=True)
        frame = {}
        for v in args.variables:
            data, r, z, t = load(f, v, tmp)
            frame[v] = data
        frame["t"] = t
        frames.append(frame)
print()

# Mirror about the axis: r -> [-r_max, r_max]
extent = [-r[-1], r[-1], z[0], z[-1]]


def mirrored(a):
    return np.concatenate([a[::-1, :], a], axis=0).T


labels = {"e": "Electron density (m$^{-3}$)",
          "electric_fld": "Electric field (V/m)",
          "phi": "Potential (V)"}
vmax = {v: max(fr[v].max() for fr in frames) for v in args.variables}

fig, axes = plt.subplots(1, 2, figsize=(7, 6), constrained_layout=True)
images = []
for ax, v, cmap in zip(axes, args.variables, ["inferno", "viridis"]):
    im = ax.imshow(mirrored(frames[0][v]), origin="lower", extent=extent,
                   vmin=0, vmax=vmax[v], cmap=cmap, aspect="equal")
    ax.set_xlabel("r (mm)")
    ax.set_ylabel("z (mm)")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x*1e3:g}")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x*1e3:g}")
    fig.colorbar(im, ax=ax, shrink=0.8).set_label(labels.get(v, v))
    images.append(im)
title = fig.suptitle("")


def update(i):
    for im, v in zip(images, args.variables):
        im.set_data(mirrored(frames[i][v]))
    title.set_text(f"t = {frames[i]['t']*1e9:.2f} ns")
    return images + [title]


os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
anim = FuncAnimation(fig, update, frames=len(frames))
anim.save(args.output, writer=PillowWriter(fps=args.fps), dpi=100)
print(f"Saved {args.output}")
