"""
BrineGrid-MPC: 24-Hour Coupled Physics & CBF-MPC Supervisory Benchmark
======================================================================
Target Plant : 5 MLD (5,000 m3/day) Zero Liquid Discharge (ZLD) CETP, Tiruppur, TN
Tariff Basis : TNERC Suo-Motu Tariff Order No. 6 of 2025 (HT-I, incl. 5% tax)
Grid Factor  : CEA CO2 Baseline Database v22.0 (0.675 tCO2 / MWh)

Note: This script is the public plant-level supervisory benchmark used for
K! Hacks 4.0 validation. Proprietary edge PLC register maps, EKF tuning matrices,
and low-level interlock firmware are intentionally excluded.

Usage:
    python brinegrid_sim.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@dataclass(frozen=True)
class PlantConfig:
    """Physical and tariff parameters for a 5 MLD Tiruppur ZLD CETP."""
    n_steps: int = 24
    dt_hr: float = 1.0
    daily_flow_m3: float = 5000.0

    # Existing tank capacities (Zero Civil CapEx: no new tanks added)
    v_eq_cap_m3: float = 1800.0     # Equalization buffer (~8.6h HRT)
    v_br_cap_m3: float = 600.0      # Brine buffer ahead of MVR

    # Discrete-time CBF safe operating bounds [20%, 85%]
    tank_min_frac: float = 0.20
    tank_max_frac: float = 0.85
    gamma_cbf: float = 0.75

    # 4-Stage RO recovery (baseline static PID vs. fouling-aware MPC)
    rec_base: float = 0.82
    rec_mpc: float = 0.85

    # Specific Energy Consumption (kWh/m3) & auxiliaries (kW)
    sec_ro_base: float = 2.52       # 4-stage high-salinity textile RO (throttled/fouled)
    sec_ro_mpc: float = 2.35        # BEP-staged RO + salinity smoothing + flush
    sec_mvr_rms: float = 20.50      # MVR evaporator + crystallizer / RMS thermal train
    p_aux_base_kw: float = 147.0    # Pre-treatment, blowers, dosing, SCADA
    p_aux_mpc_kw: float = 145.0

    # Power factor (static throttled motors vs. BEP-staged VFD + APFC)
    pf_base: float = 0.919
    pf_mpc: float = 0.962

    # Actuator envelopes (existing plant pumps & MVR sized for 82% recovery)
    q_ro_min_m3h: float = 158.0
    q_ro_max_m3h: float = 240.0
    q_mvr_min_m3h: float = 10.0     # Anti-surge turndown floor (~25%)
    q_mvr_max_m3h: float = 44.0     # Existing installed MVR capacity
    p_cmd_cap_kw: float = 1448.0    # Revised TNPDCL Contracted Demand ceiling (1,505 kVA)

    # TNERC Order No. 6/2025 HT-I rates (incl. 5% electricity tax)
    rate_normal_inr: float = 7.875  # 10:00-18:00 (Rs 7.50 * 1.05)
    rate_peak_inr: float = 9.85     # Peak C1 (06-10) & C2 (18-22): +25%
    rate_night_inr: float = 7.48    # Night C5 (22-06): -5%
    md_charge_inr_kva_mo: float = 608.0

    # Conservative annualization & CEA v22.0 emission factor
    operating_days_yr: int = 330
    field_realization: float = 0.80 # 20% conservative haircut
    cea_tco2_per_mwh: float = 0.675

    # Economics (INR Lakhs)
    capex_5mld_lakh: float = 45.0
    amc_5mld_lakh: float = 5.5
    cip_savings_gross_5mld_lakh: float = 20.0  # RO chemical CIP & membrane life extension
    capex_1mld_lakh: float = 14.5
    amc_1mld_lakh: float = 2.0


def build_profiles(cfg: PlantConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Constructs 24h influent flow, dyebath TDS, and TNERC HT-I tariff arrays."""
    t = np.arange(cfg.n_steps, dtype=float)
    t_mid = t + 0.5

    peak_c1 = (t_mid >= 6.0) & (t_mid < 10.0)
    peak_c2 = (t_mid >= 18.0) & (t_mid < 22.0)
    peak_mask = peak_c1 | peak_c2
    night_mask = (t_mid >= 22.0) | (t_mid < 6.0)

    tod = np.full(cfg.n_steps, cfg.rate_normal_inr, dtype=float)
    tod[peak_mask] = cfg.rate_peak_inr
    tod[night_mask] = cfg.rate_night_inr

    # Realistic diurnal Tiruppur dyeing cluster inflow profile summing to 5,000 m3/day
    q_shape = np.array([
        168.0, 164.0, 160.0, 158.0, 161.0, 166.0,
        170.0, 178.0, 190.0, 204.0,
        230.0, 246.0, 258.0, 266.0, 267.5, 264.0, 256.0, 244.0,
        230.0, 214.0, 196.0, 184.0,
        176.0, 171.0,
    ], dtype=float)
    q_in = q_shape * (cfg.daily_flow_m3 / np.sum(q_shape))

    # Dyebath salinity swings between 5,000 and 15,000 mg/L TDS
    tds_in = (
        9200.0
        + 4800.0 * np.sin(2.0 * np.pi * (t_mid - 8.5) / 24.0)
        + 1200.0 * np.sin(4.0 * np.pi * (t_mid - 11.0) / 24.0)
    )
    tds_in = np.clip(tds_in, 5000.0, 15000.0)

    masks = {
        "peak_c1": peak_c1,
        "peak_c2": peak_c2,
        "peak": peak_mask,
        "night": night_mask,
    }
    return t, q_in, tds_in, tod, masks


def simulate_baseline_pid(cfg: PlantConfig, q_in: np.ndarray, tds_in: np.ndarray, tod: np.ndarray) -> Dict[str, object]:
    """Simulates static 1:1 level-following PID control at 82% RO recovery."""
    p_ro = cfg.sec_ro_base * q_in * (1.0 + 0.03 * (tds_in - 9500.0) / 5000.0)
    q_brine = (1.0 - cfg.rec_base) * q_in
    p_mvr = cfg.sec_mvr_rms * q_brine
    p_tot = p_ro + p_mvr + cfg.p_aux_base_kw

    flux_norm = q_in / np.mean(q_in)
    tds_norm = tds_in / 10000.0
    tmp_drift = 0.0252 * (tds_norm ** 1.35) * (flux_norm ** 1.25)

    return {
        "q_ro_m3h": q_in.copy(),
        "q_mvr_m3h": q_brine.copy(),
        "p_kw": p_tot,
        "daily_mwh": float(np.sum(p_tot) * cfg.dt_hr / 1000.0),
        "peak_kw": float(np.max(p_tot)),
        "peak_kva": float(np.max(p_tot) / cfg.pf_base),
        "tmp_drift_bar": float(np.sum(tmp_drift)),
        "daily_energy_cost_inr": float(np.sum(tod * p_tot * cfg.dt_hr)),
        "brine_m3_day": float(np.sum(q_brine) * cfg.dt_hr),
    }


def optimize_brinegrid_mpc(cfg: PlantConfig, q_in: np.ndarray, tds_in: np.ndarray, tod: np.ndarray) -> Dict[str, object]:
    """
    Solves the 24-hour supervisory MPC problem with SLSQP under discrete-time
    Control Barrier Function (CBF) tank safety constraints:
        h_i(x_{t+1}) - h_i(x_t) >= -gamma_i * h_i(x_t),   gamma_i in (0, 1]
    """
    n = cfg.n_steps
    dt = cfg.dt_hr
    v_eq_init = 0.36 * cfg.v_eq_cap_m3
    v_br_init = 0.65 * cfg.v_br_cap_m3

    eq_min = cfg.tank_min_frac * cfg.v_eq_cap_m3
    eq_max = cfg.tank_max_frac * cfg.v_eq_cap_m3
    br_min = cfg.tank_min_frac * cfg.v_br_cap_m3
    br_max = cfg.tank_max_frac * cfg.v_br_cap_m3

    def unpack(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        return x[:n] * 200.0, x[n:2 * n] * 30.0, float(x[2 * n] * 1000.0)

    def rollout(x: np.ndarray):
        q_ro, q_mvr, _ = unpack(x)
        v_eq = np.zeros(n + 1)
        v_br = np.zeros(n + 1)
        tds_eq = np.zeros(n + 1)
        v_eq[0] = v_eq_init
        v_br[0] = v_br_init
        tds_eq[0] = 9200.0

        for k in range(n):
            v_eq[k + 1] = v_eq[k] + (q_in[k] - q_ro[k]) * dt
            v_br[k + 1] = v_br[k] + ((1.0 - cfg.rec_mpc) * q_ro[k] - q_mvr[k]) * dt
            # CSTR salinity mixing dynamics in equalization buffer
            v_mid = max(0.5 * (v_eq[k] + v_eq[k + 1]), 200.0)
            alpha = min(0.80, (q_in[k] * dt) / v_mid)
            tds_eq[k + 1] = (1.0 - alpha) * tds_eq[k] + alpha * tds_in[k]

        tds_feed = tds_eq[1:]
        p_ro = cfg.sec_ro_mpc * q_ro * (1.0 + 0.02 * (tds_feed - 9500.0) / 5000.0)
        p_mvr = cfg.sec_mvr_rms * q_mvr
        p_tot = p_ro + p_mvr + cfg.p_aux_mpc_kw

        flux_norm = q_ro / np.mean(q_in)
        tds_norm = tds_feed / 10000.0
        tmp_step = 0.0114 * (tds_norm ** 1.15) * (flux_norm ** 1.10)
        return q_ro, q_mvr, v_eq, v_br, tds_feed, p_tot, tmp_step

    c_md_daily = (cfg.md_charge_inr_kva_mo / 30.0) / cfg.pf_mpc
    lambda_foul = 1200.0

    def objective(x: np.ndarray) -> float:
        q_ro, q_mvr, _, _, _, p_tot, tmp_step = rollout(x)
        p_cap = x[2 * n] * 1000.0
        energy_cost = np.sum(tod * p_tot * dt)
        fouling_cost = lambda_foul * np.sum(tmp_step)
        demand_cost = c_md_daily * p_cap
        smooth_reg = 1e-4 * np.sum(np.diff(q_ro) ** 2) + 1e-3 * np.sum(np.diff(q_mvr) ** 2)
        return float((energy_cost + fouling_cost + demand_cost + smooth_reg) / 10000.0)

    def cbf_ineq(x: np.ndarray) -> np.ndarray:
        _, _, v_eq, v_br, _, p_tot, _ = rollout(x)
        p_cap = x[2 * n] * 1000.0
        h_eq_lo = (v_eq - eq_min) / cfg.v_eq_cap_m3
        h_eq_hi = (eq_max - v_eq) / cfg.v_eq_cap_m3
        h_br_lo = (v_br - br_min) / cfg.v_br_cap_m3
        h_br_hi = (br_max - v_br) / cfg.v_br_cap_m3
        g = cfg.gamma_cbf
        return np.concatenate([
            (p_cap - p_tot) / 1000.0,
            h_eq_lo[1:] - (1.0 - g) * h_eq_lo[:-1],
            h_eq_hi[1:] - (1.0 - g) * h_eq_hi[:-1],
            h_br_lo[1:] - (1.0 - g) * h_br_lo[:-1],
            h_br_hi[1:] - (1.0 - g) * h_br_hi[:-1],
        ])

    def cyclic_eq(x: np.ndarray) -> np.ndarray:
        _, _, v_eq, v_br, _, _, _ = rollout(x)
        return np.array([
            (v_eq[-1] - v_eq_init) / cfg.v_eq_cap_m3,
            (v_br[-1] - v_br_init) / cfg.v_br_cap_m3,
        ])

    x0 = np.concatenate([
        np.full(n, (cfg.daily_flow_m3 / n) / 200.0),
        np.full(n, ((1.0 - cfg.rec_mpc) * cfg.daily_flow_m3 / n) / 30.0),
        [cfg.p_cmd_cap_kw / 1000.0],
    ])
    bounds = (
        [(cfg.q_ro_min_m3h / 200.0, cfg.q_ro_max_m3h / 200.0)] * n
        + [(cfg.q_mvr_min_m3h / 30.0, cfg.q_mvr_max_m3h / 30.0)] * n
        + [(0.80, cfg.p_cmd_cap_kw / 1000.0)]
    )

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "ineq", "fun": cbf_ineq},
            {"type": "eq", "fun": cyclic_eq},
        ],
        options={"maxiter": 600, "ftol": 1e-8},
    )

    q_ro, q_mvr, v_eq, v_br, tds_feed, p_tot, tmp_step = rollout(res.x)
    return {
        "converged": bool(res.success),
        "solver_message": str(res.message),
        "iterations": int(res.nit),
        "q_ro_m3h": q_ro,
        "q_mvr_m3h": q_mvr,
        "p_kw": p_tot,
        "eq_pct": 100.0 * v_eq / cfg.v_eq_cap_m3,
        "br_pct": 100.0 * v_br / cfg.v_br_cap_m3,
        "tds_feed_mg_l": tds_feed,
        "daily_mwh": float(np.sum(p_tot) * dt / 1000.0),
        "peak_kw": float(np.max(p_tot)),
        "peak_kva": float(np.max(p_tot) / cfg.pf_mpc),
        "tmp_drift_bar": float(np.sum(tmp_step)),
        "daily_energy_cost_inr": float(np.sum(tod * p_tot * dt)),
        "brine_m3_day": float(np.sum(q_mvr) * dt),
    }


def compute_economics(cfg: PlantConfig, base: Dict[str, object], mpc: Dict[str, object]) -> Dict[str, float]:
    """Computes annualized energy, carbon, brine diversion, and ROI metrics."""
    days = cfg.operating_days_yr
    real = cfg.field_realization

    saved_mwh_day = base["daily_mwh"] - mpc["daily_mwh"]
    annual_energy_gwh = (saved_mwh_day * days * real) / 1000.0
    annual_co2_t = annual_energy_gwh * 1000.0 * cfg.cea_tco2_per_mwh
    annual_brine_diverted_kl = (base["brine_m3_day"] - mpc["brine_m3_day"]) * days

    # Enterprise Tier (5 MLD CETP)
    gross_energy_tod_lakh = (base["daily_energy_cost_inr"] - mpc["daily_energy_cost_inr"]) * days / 1e5
    kva_saved = base["peak_kva"] - mpc["peak_kva"]
    gross_md_lakh = (kva_saved * cfg.md_charge_inr_kva_mo * 12.0) / 1e5
    gross_total_5mld_lakh = gross_energy_tod_lakh + gross_md_lakh + cfg.cip_savings_gross_5mld_lakh
    net_savings_5mld_lakh = gross_total_5mld_lakh * real - cfg.amc_5mld_lakh
    payback_5mld_mo = (cfg.capex_5mld_lakh / net_savings_5mld_lakh) * 12.0

    # MSME Tier (1 MLD Standalone IETP — 1/5th hydraulic scale)
    gross_total_1mld_lakh = gross_total_5mld_lakh / 5.0
    net_savings_1mld_lakh = gross_total_1mld_lakh * real - cfg.amc_1mld_lakh
    payback_1mld_mo = (cfg.capex_1mld_lakh / net_savings_1mld_lakh) * 12.0

    return {
        "annual_energy_saved_gwh": round(annual_energy_gwh, 2),
        "annual_co2_avoided_t": round(annual_co2_t, 1),
        "annual_brine_diverted_kl": round(annual_brine_diverted_kl, 0),
        "enterprise_5mld_gross_energy_tod_lakh": round(gross_energy_tod_lakh, 2),
        "enterprise_5mld_gross_md_lakh": round(gross_md_lakh, 2),
        "enterprise_5mld_net_savings_lakh": round(net_savings_5mld_lakh, 1),
        "enterprise_5mld_payback_months": round(payback_5mld_mo, 1),
        "msme_1mld_net_savings_lakh": round(net_savings_1mld_lakh, 1),
        "msme_1mld_payback_months": round(payback_1mld_mo, 1),
    }


def save_benchmark_plot(
    t: np.ndarray,
    base_kw: np.ndarray,
    mpc_kw: np.ndarray,
    eq_pct: np.ndarray,
    br_pct: np.ndarray,
    out_path: str,
) -> None:
    """Exports a publication-grade dark-slate 24h benchmark PNG."""
    if not HAS_MPL:
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=220)
    bg = "#0D1628"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # Shaded TNERC Peak C1 & C2 tariff bands
    ax.axvspan(6, 10, color="#DC2626", alpha=0.18, zorder=1)
    ax.axvspan(18, 22, color="#DC2626", alpha=0.18, zorder=1)
    ax.text(8.0, 2065, "C1 PEAK · Rs. 9.85", color="#FCA5A5", ha="center", va="center",
            fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#7F1D1D", edgecolor="#EF4444", alpha=0.9))
    ax.text(20.0, 2065, "C2 PEAK · Rs. 9.85", color="#FCA5A5", ha="center", va="center",
            fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#7F1D1D", edgecolor="#EF4444", alpha=0.9))

    # Periodic 0..24h x-grid
    t_line = np.arange(25, dtype=float)
    base_ext = np.append(base_kw, base_kw[0])
    mpc_ext = np.append(mpc_kw, mpc_kw[0])

    ax.plot(t_line, base_ext, linestyle="--", color="#E2E8F0", linewidth=2.2,
            label=f"Baseline Static PID (Peak {np.max(base_kw):,.0f} kW)", zorder=3)
    ax.step(t_line, mpc_ext, where="post", color="#38BDF8", linewidth=2.8,
            label=f"BrineGrid-MPC (Cap {np.max(mpc_kw):,.0f} kW)", zorder=4)
    ax.fill_between(t_line, 450, mpc_ext, step="post", color="#38BDF8", alpha=0.12, zorder=2)

    # Peak marker on baseline
    pk_idx = int(np.argmax(base_kw))
    ax.scatter([pk_idx], [base_kw[pk_idx]], color="#FFFFFF", edgecolor="#0D1628", s=55, zorder=5)
    ax.annotate(
        f"{base_kw[pk_idx]:,.0f} kW",
        xy=(pk_idx, base_kw[pk_idx]),
        xytext=(pk_idx, base_kw[pk_idx] + 105),
        ha="center", color="#F8FAFC", fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#1E293B", edgecolor="#64748B"),
    )

    ax.set_xlim(0, 24)
    ax.set_ylim(450, 2180)
    ax.set_xticks([0, 6, 10, 14, 18, 22, 24])
    ax.set_xticklabels(["00:00", "06:00", "10:00", "14:00", "18:00", "22:00", "24:00"], color="#CBD5E1", fontsize=9.5)
    ax.set_yticks([500, 1000, 1500, 2000])
    ax.set_yticklabels(["500", "1000", "1500", "2000"], color="#CBD5E1", fontsize=9.5)
    ax.grid(axis="y", color="#334155", linestyle="-", linewidth=0.7, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")

    ax.set_xlabel("Time of Day", color="#E2E8F0", fontsize=10.5, fontweight="bold", labelpad=8)
    ax.set_ylabel("Electrical Power (kW)", color="#E2E8F0", fontsize=10.5, fontweight="bold", labelpad=8)
    ax.set_title(
        "BrineGrid-MPC: 24h Physics-Model Simulation (5 MLD Tiruppur CETP · TNERC HT-I)",
        color="#F8FAFC", fontsize=12.5, fontweight="bold", pad=32,
    )

    leg = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2,
        frameon=True, facecolor="#111C33", edgecolor="#334155", fontsize=9.2,
    )
    for text in leg.get_texts():
        text.set_color("#E2E8F0")

    fig.text(
        0.5, 0.015,
        f"Tank Bounds Held: Eq Tank {np.min(eq_pct):.1f}-{np.max(eq_pct):.1f}%  |  "
        f"Brine Tank {np.min(br_pct):.1f}-{np.max(br_pct):.1f}%  (PLC CBF Safe Limit: 20-85%)",
        ha="center", color="#94A3B8", fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, facecolor=bg)
    plt.close(fig)


def main() -> None:
    cfg = PlantConfig()
    t, q_in, tds_in, tod, masks = build_profiles(cfg)
    base = simulate_baseline_pid(cfg, q_in, tds_in, tod)
    mpc = optimize_brinegrid_mpc(cfg, q_in, tds_in, tod)
    econ = compute_economics(cfg, base, mpc)

    peak_kwh_base = float(np.sum(base["p_kw"][masks["peak"]]))
    peak_kwh_mpc = float(np.sum(mpc["p_kw"][masks["peak"]]))

    summary = {
        "solver_converged": mpc["converged"],
        "solver_iterations": mpc["iterations"],
        "daily_energy_mwh": {
            "baseline": round(base["daily_mwh"], 2),
            "mpc": round(mpc["daily_mwh"], 2),
            "reduction_pct": round(100.0 * (mpc["daily_mwh"] - base["daily_mwh"]) / base["daily_mwh"], 1),
        },
        "peak_c1_c2_kwh": {
            "baseline": round(peak_kwh_base, 0),
            "mpc": round(peak_kwh_mpc, 0),
            "reduction_pct": round(100.0 * (peak_kwh_mpc - peak_kwh_base) / peak_kwh_base, 1),
        },
        "peak_power_kw": {
            "baseline": round(base["peak_kw"], 1),
            "mpc": round(mpc["peak_kw"], 1),
        },
        "peak_demand_kva": {
            "baseline": round(base["peak_kva"], 1),
            "mpc": round(mpc["peak_kva"], 1),
            "reduction_pct": round(100.0 * (mpc["peak_kva"] - base["peak_kva"]) / base["peak_kva"], 1),
        },
        "ro_fouling_tmp_drift_bar": {
            "baseline": round(base["tmp_drift_bar"], 2),
            "mpc": round(mpc["tmp_drift_bar"], 2),
            "reduction_pct": round(100.0 * (mpc["tmp_drift_bar"] - base["tmp_drift_bar"]) / base["tmp_drift_bar"], 1),
        },
        "tank_bounds_pct": {
            "eq_min": round(float(np.min(mpc["eq_pct"])), 1),
            "eq_max": round(float(np.max(mpc["eq_pct"])), 1),
            "brine_min": round(float(np.min(mpc["br_pct"])), 1),
            "brine_max": round(float(np.max(mpc["br_pct"])), 1),
            "safe_limit": [20.0, 85.0],
        },
        "ro_brine_m3_day": {
            "baseline_82pct": round(base["brine_m3_day"], 1),
            "mpc_85pct": round(mpc["brine_m3_day"], 1),
            "reduction_pct": round(100.0 * (mpc["brine_m3_day"] - base["brine_m3_day"]) / base["brine_m3_day"], 1),
        },
        "annualized_economics": econ,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_benchmark_plot(t, base["p_kw"], mpc["p_kw"], mpc["eq_pct"], mpc["br_pct"], "assets/benchmark_24h.png")

    print("=== BRINEGRID-MPC 24-HOUR SIMULATION BENCHMARK (5 MLD TIRUPPUR CETP) ===")
    print(f"SLSQP Converged    : {mpc['converged']} ({mpc['iterations']} iterations)")
    print(f"Daily Energy       : {base['daily_mwh']:.1f} -> {mpc['daily_mwh']:.1f} MWh ({summary['daily_energy_mwh']['reduction_pct']:.1f}%)")
    print(f"Peak C1+C2 Draw    : {peak_kwh_base:,.0f} -> {peak_kwh_mpc:,.0f} kWh/day ({summary['peak_c1_c2_kwh']['reduction_pct']:.1f}%)")
    print(f"Peak Power         : {base['peak_kw']:,.0f} -> {mpc['peak_kw']:,.0f} kW")
    print(f"Peak Demand        : {base['peak_kva']:,.0f} -> {mpc['peak_kva']:,.0f} kVA ({summary['peak_demand_kva']['reduction_pct']:.1f}%)")
    print(f"24h RO TMP Drift   : +{base['tmp_drift_bar']:.2f} -> +{mpc['tmp_drift_bar']:.2f} bar ({summary['ro_fouling_tmp_drift_bar']['reduction_pct']:.1f}%)")
    print(f"RO Brine Reject    : {base['brine_m3_day']:.0f} -> {mpc['brine_m3_day']:.0f} m3/day ({summary['ro_brine_m3_day']['reduction_pct']:.1f}%)")
    print(f"Eq Tank Range      : {summary['tank_bounds_pct']['eq_min']:.1f}% - {summary['tank_bounds_pct']['eq_max']:.1f}% (Safe: 20-85%)")
    print(f"Brine Tank Range   : {summary['tank_bounds_pct']['brine_min']:.1f}% - {summary['tank_bounds_pct']['brine_max']:.1f}% (Safe: 20-85%)")
    print("\n=== ANNUALIZED ECONOMICS (330 DAYS, 80% REALIZATION, AMC DEDUCTED) ===")
    print(f"Energy Conserved   : {econ['annual_energy_saved_gwh']:.2f} GWh/yr")
    print(f"Carbon Avoided     : {econ['annual_co2_avoided_t']:.0f} tCO2e/yr (CEA v22.0: {cfg.cea_tco2_per_mwh} tCO2/MWh)")
    print(f"Brine Diverted     : {econ['annual_brine_diverted_kl']:,.0f} kL/yr")
    print(f"5 MLD Enterprise   : Rs. {econ['enterprise_5mld_net_savings_lakh']:.1f} Lakh/yr | Payback: {econ['enterprise_5mld_payback_months']:.1f} months")
    print(f"1 MLD MSME Tier    : Rs. {econ['msme_1mld_net_savings_lakh']:.1f} Lakh/yr | Payback: {econ['msme_1mld_payback_months']:.1f} months")


if __name__ == "__main__":
    main()
