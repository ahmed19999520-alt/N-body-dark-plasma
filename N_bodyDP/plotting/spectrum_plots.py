import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..constants import PAPER_TARGETS, DarkSectorParameters
from ..dark_acoustic import dao_modulation
from ..gmo import OCTET, spectrum
from ..gravitational_waves import ConfinementTransition, LisaNoiseModel, pli_sensitivity_paper
from ..self_interaction import SelfInteractionModel


def _finish(fig, path):
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def octet_weight_diagram(path=None, p: DarkSectorParameters = DarkSectorParameters()):
    m = spectrum(p)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for s in OCTET:
        i3 = float(s.isospin) if s.name.startswith("chi") else (0.0 if s.name in ("Sigma0", "Lambda") else (-1.0 if s.name == "Sigma-" else 1.0))
        if s.name.startswith("chi"):
            i3 = {"chi0": -1.5, "chi-": -0.5, "chi+": 0.5, "chi++": 1.5}[s.name]
        colour = "goldenrod" if s.name == "chi0" else "tab:blue"
        ax.scatter(i3, s.hypercharge, s=160, color=colour, zorder=3)
        ax.annotate(f"{s.name}\n{m[s.name]:.2f}", (i3, s.hypercharge), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8)
    ax.set_xlabel("I3")
    ax.set_ylabel("Y")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.4, 1.8)
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def gw_spectrum_plot(transition=None, path=None):
    tr = transition or ConfinementTransition()
    f = np.geomspace(1.0e-5, 1.0e-1, 800)
    comp = tr.components(f)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(f, comp["sound_wave"], label="sound waves")
    ax.loglog(f, comp["bubble"], "--", label="bubble collisions")
    ax.loglog(f, comp["turbulence"], ":", label="MHD turbulence")
    ax.loglog(f, comp["total"], "r", lw=2, label="total")
    ax.loglog(f, pli_sensitivity_paper(f), "k-.", label="LISA PLI")
    f_star = tr.composite_peak_frequency()
    sub = f[f < 0.3 * f_star]
    ax.loglog(sub, tr.total_peak_amplitude() * (sub / f_star) ** 3 * 0.3, color="grey", ls="--", lw=1)
    ax.axvline(f_star, color="grey", lw=0.6)
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("Omega_GW h^2")
    ax.set_ylim(1.0e-16, 1.0e-6)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    return _finish(fig, path)


def self_interaction_plot(path=None):
    model = SelfInteractionModel()
    v = np.geomspace(3.0, 2000.0, 300)
    s = model.sigma_over_m(v)
    e = model.uncertainty(v)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.loglog(v, s, "b", label="model")
    ax.fill_between(v, np.maximum(s - e, 1e-6), s + e, color="b", alpha=0.2)
    for name, (vv, sv, ev, lim) in PAPER_TARGETS["self_interaction"]["table"].items():
        ax.errorbar(vv, sv, yerr=ev, fmt="o", color="k", ms=4)
        ax.plot([vv * 0.8, vv * 1.25], [lim, lim], color="r", lw=1)
    ax.set_xlabel("v [km/s]")
    ax.set_ylabel("sigma_T/m [cm^2/g]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _finish(fig, path)


def boltzmann_yield_plot(result, names=("chi0", "chi++", "Sigma0"), path=None):
    from ..gmo import STATE_NAMES

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for n in names:
        i = STATE_NAMES.index(n)
        ax.loglog(result.x, np.maximum(result.y[i], 1e-30), label=n)
    ax.loglog(result.x, np.maximum(result.y_eq[0], 1e-30), "k:", label="equilibrium chi0")
    ax.set_ylim(1.0e-16, 1.0e-1)
    ax.set_xlabel("x = M0/T")
    ax.set_ylabel("Y")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    return _finish(fig, path)


def dao_plot(path=None):
    k = np.linspace(0.05, 6.0, 600)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(k, dao_modulation(k))
    ax.set_xlabel("k [1/Mpc]")
    ax.set_ylabel("P/P_smooth")
    ax.grid(alpha=0.3)
    return _finish(fig, path)


def lisa_sensitivity_plot(path=None):
    f = np.geomspace(1.0e-4, 1.0e-1, 500)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(f, LisaNoiseModel().omega_sensitivity_h2(f), label="TDI A channel")
    ax.loglog(f, pli_sensitivity_paper(f), "--", label="PLI four years")
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("Omega h^2")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    return _finish(fig, path)
