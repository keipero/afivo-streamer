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
                 quiet=False, reuse=False):
    """Run ./streamer in programs/<program> and return the output base path.

    program   : folder under programs/ (e.g. "standard_2d")
    cfg       : base config file (e.g. "streamer_cyl.cfg")
    name      : run name; output goes to programs/<program>/output/<name>_*
    overrides : settings to override, {name: value}
    threads   : number of OpenMP threads (about 4 is fastest on this PC)
    reuse     : if True and a finished run with this name exists (its log
                reaches end_time, or it stopped at end_streamer_length),
                skip running and return its base path
    """
    workdir = REPO / "programs" / program
    (workdir / "output").mkdir(exist_ok=True)
    overrides = dict(overrides or {})
    overrides["output%name"] = f"output/{name}"
    base = workdir / "output" / name

    if reuse and "end_time" in overrides and Path(f"{base}_log.txt").exists():
        t_last = load_log(base)["time"].iloc[-1]
        stdout = Path(f"{base}_stdout.txt")
        stopped = stdout.exists() and "reached its desired length" in stdout.read_text(errors="ignore")
        if t_last >= 0.999 * float(overrides["end_time"]) or stopped:
            if not quiet:
                print(f"計算済みの結果を使います: {base}")
            return base

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


def needle_plane_settings(gap, voltage, tip_radius=200e-6, shank_radius=0.5e-3,
                          cone_length=2e-3, needle_length=4e-3, photoi=True,
                          background_density=1e13, chemistry="simple",
                          ion_mobility=2.2e-4):
    """Config overrides for an axisymmetric needle-plane gap.

    Use with programs/standard_2d and the base config
    streamer_cyl_electrode.cfg. The needle hangs from the top of a square
    domain of size L = gap + needle_length; the grounded plane is z = 0.

    gap          : distance from the needle tip to the plane (m)
    voltage      : needle voltage (V); > 0 positive needle, < 0 negative
    tip_radius   : radius of curvature of the needle tip (m)
    shank_radius : radius of the cylindrical part of the needle (m)
    cone_length  : length of the conical part (m)
    needle_length: length of the needle inside the domain (m)
    photoi       : enable photoionization
    background_density : initial electron and ion density (1/m^3)
    chemistry    : "simple"   - ionization and attachment only (td_air_siglo_swarm.txt)
                   "detailed" - air_chemistry_v3 (detachment, ion conversion,
                                recombination) plus N2(C3) for light emission
                                (transport_data/air_chemistry_v3_light.txt)
    ion_mobility : mobility of the ions (m^2/(V s)) when chemistry = "detailed"

    Returns (overrides, info). info holds the geometry in meters.
    """
    L = gap + needle_length
    # rod_cone_top: r0 = base of the rod, r1 = end of the cone. The tip is a
    # sphere tangent to the cone, so the apex is slightly below r1. Solve for
    # r1 such that the apex is exactly at z = gap (two fixed-point steps).
    z1 = gap
    for _ in range(3):
        a = np.arctan((shank_radius - tip_radius) / cone_length)
        r_curv = tip_radius / np.cos(a)
        z1 = gap + r_curv * (1 - np.sin(a))
    cone_frac = cone_length / (L - z1)
    if not 0 < cone_frac < 1:
        raise ValueError("cone_length must be shorter than the needle")

    overrides = {
        "cylindrical": True,
        "domain_len": [L, L],
        "use_electrode": True,
        "field_electrode_type": "rod_cone_top",
        "field_electrode_grounded": False,
        "field_rod_r0": [0.0, 1.0],
        "field_rod_r1": [0.0, z1 / L],
        "field_rod_radius": shank_radius,
        "cone_tip_radius": tip_radius,
        "cone_length_frac": cone_frac,
        "refine_electrode_dx": min(50e-6, tip_radius / 4),
        "field_given_by": f"voltage {voltage:g}",
        "background_density": background_density,
        "photoi%enabled": photoi,
        "fixes%source_factor": "flux",   # 'original_flux' in the example cfg is no longer valid
    }
    if chemistry == "simple":
        overrides["photoi%species"] = "M_plus"
    elif chemistry == "detailed":
        ions = ["N2_plus", "N4_plus", "O2_plus", "O4_plus", "O_min", "O2_min", "O3_min"]
        overrides.update({
            "input_data%file": "../../transport_data/air_chemistry_v3_light.txt",
            "input_data%old_style": False,
            "photoi%species": "O2_plus",
            "input_data%mobile_ions": ions,
            "input_data%ion_mobilities": [ion_mobility] * len(ions),
        })
    else:
        raise ValueError("chemistry must be 'simple' or 'detailed'")
    info = {"L": L, "gap": gap, "tip_z": gap, "z1": z1, "cone_half_angle_deg": np.degrees(a),
            "mean_field": abs(voltage) / gap}
    return overrides, info


# Radiative decay rate of N2(C3) (second positive system), 1/s (lifetime 42 ns)
N2_C3_DECAY_RATE = 2.381e7


def streak_image(base, r_max, z_max, variable="N2_C3", factor=N2_C3_DECAY_RATE,
                 mode="slit", every=1):
    """Synthetic streak image I(t, z) from an axisymmetric run.

    The emission density is factor * <variable> (for N2_C3: photons/(m^3 s)
    of the second positive system). For each silo file it computes

      mode = "slit": line-of-sight integral through the axis,
                     I(z) = 2 * integral_0^R eps(r, z) dr          [1/(m^2 s)]
                     (a streak camera whose slit is imaged on the axis)
      mode = "total": light per unit length in z,
                     I(z) = integral_0^R eps(r, z) 2 pi r dr       [1/(m s)]

    Returns (t, z, I) with I of shape (n_times, n_z).
    """
    files = silo_files(base)[::every]
    times, rows = [], []
    for f in files:
        data, r, z, t = load_silo(f, variable, r_max=[r_max, z_max])
        eps = factor * data                       # shape (n_r, n_z)
        dr = r[1] - r[0]
        if mode == "slit":
            I = 2 * eps.sum(axis=0) * dr
        elif mode == "total":
            I = (eps * (2 * np.pi * r)[:, None]).sum(axis=0) * dr
        else:
            raise ValueError("mode must be 'slit' or 'total'")
        times.append(t)
        rows.append(I)
    return np.array(times), z, np.array(rows)
