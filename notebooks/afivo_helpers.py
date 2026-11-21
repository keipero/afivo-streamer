"""Helpers to run afivo-streamer from Python and read its output.

Used by the notebooks in this folder.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "afivo" / "tools"


def _fmt(value):
    """Convert a Python value to an afivo-streamer config value string."""
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, (list, tuple)):
        return " ".join(_fmt(v) for v in value)
    return str(value)


def run_streamer(program, cfg, name, overrides=None, threads=4, show_every=1,
                 quiet=False):
    """Run ./streamer in programs/<program> and return the output base path.

    program   : folder under programs/ (e.g. "standard_2d")
    cfg       : base config file (e.g. "streamer_cyl.cfg")
    name      : run name; output goes to programs/<program>/output/<name>_*
    overrides : settings to override, {name: value}
    threads   : number of OpenMP threads (about 4 is fastest on this PC)
    """
    workdir = REPO / "programs" / program
    (workdir / "output").mkdir(exist_ok=True)
    overrides = dict(overrides or {})
    overrides["output%name"] = f"output/{name}"

    cmd = ["./streamer", cfg] + [f"-{k}={_fmt(v)}" for k, v in overrides.items()]
    if not quiet:
        print("実行:", " ".join(cmd))
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))

    t0 = time.time()
    logfile = workdir / "output" / f"{name}_stdout.txt"
    with open(logfile, "w") as log, subprocess.Popen(
            cmd, cwd=workdir, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True) as proc:
        n = 0
        for line in proc.stdout:
            log.write(line)
            if not quiet and "%" in line and "it:" in line:
                n += 1
                if n % show_every == 0:
                    print(f"  [{time.time()-t0:6.0f} s] {line.strip()}")
    if proc.returncode != 0:
        print(open(logfile).read()[-3000:])
        raise RuntimeError(f"streamer failed for {name} (code {proc.returncode})")
    if not quiet:
        print(f"完了: {time.time()-t0:.0f} 秒  (標準出力: {logfile})")
    return workdir / "output" / name


def load_log(base):
    """Read <base>_log.txt as a DataFrame (columns: time, v, max(E), ...)."""
    return pd.read_csv(f"{base}_log.txt", sep=r"\s+")


def silo_files(base):
    base = Path(base)
    return sorted(base.parent.glob(base.name + "_[0-9]*.silo"))


def load_silo(silo, variable, r_max=None):
    """Interpolate one variable of a silo file to a uniform grid.

    Returns (data, coords..., time). For 2D, data has shape (n_r, n_z) and
    the coordinates are r and z; for 1D, data has shape (n_z,).
    """
    npz = Path("/tmp") / f"_afivo_{os.getpid()}.npz"
    cmd = [sys.executable, str(TOOLS / "plot_raw_data.py"), str(silo),
           "-variable", variable, "-save_npz", str(npz), "-q"]
    if r_max is not None:
        cmd += ["-r_max", *map(str, r_max)]
    subprocess.run(cmd, check=True)
    d = np.load(npz)
    coords = [d[k] for k in sorted(d.files) if k.startswith("arr_")]
    return (d["uniform_data"], *coords, float(d["time"]))


def front_velocity(t, z, z_lo, z_hi, t_min=1e-9):
    """Fit a front velocity (m/s) to front positions z(t).

    Only points with z_lo < z < z_hi and t >= t_min are used, and of those
    the second half (to skip the initial transient). Returns nan if there
    are too few points.
    """
    t, z = np.asarray(t), np.asarray(z)
    ok = (z > z_lo) & (z < z_hi) & (t >= t_min)
    t, z = t[ok], z[ok]
    if len(t) < 4:
        return np.nan
    h = len(t) // 2
    return abs(np.polyfit(t[h:], z[h:], 1)[0])


def lineout_settings(ndim, variables=("e", "electric_fld"), npoints=500):
    """Config overrides that write values along a line at every output.

    1D: the whole domain. 2D (axisymmetric): the symmetry axis r = 0.
    Silo files of 1D runs cannot be read by plot_raw_data.py, so use this.
    """
    rmin, rmax = ([0.0], [1.0]) if ndim == 1 else ([0.0, 0.0], [0.0, 1.0])
    return {"lineout%write": True, "lineout%varname": list(variables),
            "lineout%npoints": npoints,
            "lineout%rmin": rmin, "lineout%rmax": rmax}


def load_lineout(base, index=-1):
    """Read <base>_line_<n>.txt written with lineout_settings().

    Returns (DataFrame, time). Columns are x (and y in 2D) plus the variables.
    index selects the output number (-1 = last).
    """
    base = Path(base)
    files = sorted(base.parent.glob(base.name + "_line_[0-9]*.txt"))
    f = files[index]
    with open(f) as fp:
        names = fp.readline().lstrip("#").split()
    df = pd.read_csv(f, sep=r"\s+", comment="#", names=names)
    n = int(f.stem.rsplit("_", 1)[1])
    log = load_log(base)
    t = float(log["time"].iloc[n]) if n < len(log) else np.nan
    return df, t


def load_transport(base):
    """Read <base>_summary.txt (electron transport data used in the run)."""
    d = pd.read_csv(f"{base}_summary.txt", sep=r"\s+")
    return d.rename(columns={
        "E/N[Td]": "EN", "E[V/m]": "E",
        "Electron_mobility[m^2/(Vs)]": "mu",
        "Electron_diffusion[m^2/s]": "D",
        "Townsend_ioniz._coef._alpha[1/m]": "alpha",
        "Townsend_attach._coef._eta[1/m]": "eta"})


def pulled_front_velocity(transport, E):
    """Asymptotic velocity of a planar negative front without background
    ionization: v* = mu*E + 2*sqrt(D*nu), nu = (alpha - eta)*mu*E.

    Returns (v_star, v_drift, nu) for field strength E (V/m).
    """
    mu, D, a, eta = (np.interp(E, transport["E"], transport[c])
                     for c in ["mu", "D", "alpha", "eta"])
    vd = mu * E
    nu = (a - eta) * vd
    return vd + 2 * np.sqrt(D * max(nu, 0.0)), vd, nu
