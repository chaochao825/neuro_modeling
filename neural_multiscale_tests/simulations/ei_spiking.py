"""Leaky integrate-and-fire E/I network for oscillation checks."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from analyses.metrics import dmd_summary, psd_summary, spike_phase_locking


def simulate_lif_ei(
    n_e: int = 56,
    n_i: int = 14,
    t_steps: int = 1600,
    mode: str = "gamma_sync",
    seed: int = 4,
) -> Dict[str, object]:
    gen = np.random.default_rng(seed)
    n = n_e + n_i
    is_i = np.zeros(n, dtype=bool)
    is_i[n_e:] = True
    v = -65.0 + 4.0 * gen.normal(size=n)
    refractory = np.zeros(n, dtype=int)
    spikes = np.zeros((t_steps, n), dtype=float)
    exc = np.zeros(n)
    inh = np.zeros(n)
    tau_m = 20.0
    tau_e = 4.0
    tau_i = 9.0 if mode == "gamma_sync" else 14.0
    if mode == "gamma_sync":
        base_e, base_i = 17.2, 15.8
        w_ee, w_ei, w_ie, w_ii = 0.12, 0.42, 0.72, 0.28
        noise = 2.5
    else:
        base_e, base_i = 15.5, 14.7
        w_ee, w_ei, w_ie, w_ii = 0.06, 0.18, 0.24, 0.14
        noise = 4.0
    conn = gen.random((n, n))
    p = np.where(is_i[:, None], 0.22, 0.16)
    conn = conn < p
    np.fill_diagonal(conn, False)
    pulse_t = t_steps // 2
    reset_pre = 0.0
    reset_post = 0.0
    for t in range(1, t_steps):
        exc *= np.exp(-1.0 / tau_e)
        inh *= np.exp(-1.0 / tau_i)
        prev = spikes[t - 1].astype(bool)
        if prev.any():
            e_prev = prev & ~is_i
            i_prev = prev & is_i
            if e_prev.any():
                targets = conn[e_prev].sum(axis=0)
                exc += targets * np.where(is_i, w_ei, w_ee) / max(1, int(e_prev.sum()))
            if i_prev.any():
                targets = conn[i_prev].sum(axis=0)
                inh += targets * np.where(is_i, w_ii, w_ie) / max(1, int(i_prev.sum()))
        drive = np.where(is_i, base_i, base_e) + noise * gen.normal(size=n)
        if pulse_t <= t < pulse_t + 4:
            drive[~is_i] += 10.0
        active = refractory <= 0
        dv = ((-65.0 - v) + drive + 18.0 * exc - 18.0 * inh) / tau_m
        v[active] += dv[active]
        refractory[~active] -= 1
        fired = v > -50.0
        spikes[t, fired] = 1.0
        v[fired] = -70.0
        refractory[fired] = 3
        pop_rate = spikes[: t + 1, :n_e].sum(axis=1)
        if t == pulse_t - 1:
            reset_pre = _phase_concentration(pop_rate[-120:])
        if t == pulse_t + 90:
            reset_post = _phase_concentration(pop_rate[pulse_t : pulse_t + 90])
    pop_e = spikes[:, :n_e].sum(axis=1)
    return {
        "mode": mode,
        "spikes": spikes,
        "population_rate": pop_e,
        "metrics": {
            "psd": psd_summary(pop_e, dt=0.001),
            "dmd": dmd_summary(spikes[:, : min(40, n)]),
            "phase_locking": spike_phase_locking(spikes, pop_e),
            "phase_reset_proxy": float(reset_post - reset_pre),
            "mean_rate_per_bin": float(spikes.mean()),
        },
    }


def _phase_concentration(signal: np.ndarray) -> float:
    from analyses.metrics import analytic_phase

    phase = analytic_phase(signal)
    return float(np.abs(np.mean(np.exp(1j * phase))))


def run_ei_suite(seed: int = 4) -> Dict[str, object]:
    cases: List[Dict[str, object]] = []
    for i, mode in enumerate(["balanced_async", "gamma_sync"]):
        out = simulate_lif_ei(mode=mode, seed=seed + i)
        cases.append({"mode": mode, "metrics": out["metrics"]})
    return {"name": "ei_lif_oscillation", "cases": cases}

