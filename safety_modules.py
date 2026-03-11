# safety_modules.py
"""
Nuclear Power Plant Simulator – Safety Design Modules (v1.0)
============================================================
Strategic companion to app.py (NPP Simulator v10.0).

Modules:
  1.  Defense-in-Depth (DiD) Barrier Tracker
  2.  Passive Safety Systems Designer & Benchmarker
  3.  Probabilistic Risk Assessment (PRA) – Event Tree / Fault Tree
  4.  Negative Feedback Coefficient Stability Map
  5.  Containment Performance & H2 Risk
  6.  Severe Accident Progression (SAP) Tracker
  7.  Emergency Operating Procedure (EOP) Trainer
  8.  Regulatory Safety Margin Tracker (SAFDL, DNBR, PCT)
  9.  SMR vs Large LWR Safety Advantage Calculator
  10. Seismic & External Hazard Stress Test
  11. Multi-Unit Risk & Shared-Infrastructure Model
  12. Human Reliability Analysis (HRA) & Alarm Load Estimator
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time

# ══════════════════════════════════════════════════════════════════
#  SHARED DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class BarrierState:
    """Integrity fraction 0.0 (failed) → 1.0 (intact) for each barrier."""
    fuel_pellet:   float = 1.0
    cladding:      float = 1.0
    primary_bound: float = 1.0
    containment:   float = 1.0

    def overall(self) -> float:
        return self.fuel_pellet * self.cladding * self.primary_bound * self.containment

    @staticmethod
    def from_sim(power: float, Tf: float, pressure: float,
                 h2: float, core_damage: str) -> "BarrierState":
        """Derive barrier state from live simulation variables."""
        b = BarrierState()
        # Fuel pellet integrity
        if Tf > 2800:   b.fuel_pellet = 0.0
        elif Tf > 1800: b.fuel_pellet = max(0.0, 1.0 - (Tf - 1800) / 1000)
        elif Tf > 1200: b.fuel_pellet = max(0.2, 1.0 - (Tf - 1200) / 1500)
        # Cladding
        if Tf > 2200:   b.cladding = 0.0
        elif Tf > 1200: b.cladding = max(0.0, 1.0 - (Tf - 1200) / 1000)
        # Primary pressure boundary
        if pressure > 200: b.primary_bound = max(0.0, 1.0 - (pressure - 200) / 50)
        if "LOCA" in core_damage or pressure < 50: b.primary_bound = min(b.primary_bound, 0.3)
        # Containment (H2 risk)
        if h2 > 2.0:    b.containment = 0.0
        elif h2 > 0.5:  b.containment = max(0.0, 1.0 - (h2 - 0.5) / 1.5)
        return b


# ══════════════════════════════════════════════════════════════════
#  MODULE 1 – DEFENSE-IN-DEPTH BARRIER TRACKER
# ══════════════════════════════════════════════════════════════════

def render_did_tracker(history: list):
    st.subheader("🏛️ Defense-in-Depth Barrier Tracker")
    st.caption(
        "IAEA Safety Fundamentals require multiple independent barriers between "
        "radioactive material and the public. This panel tracks real-time integrity "
        "of each barrier based on the simulation physics."
    )

    if not history:
        st.info("Run a simulation to see barrier integrity evolve in real time.")
        return

    df = pd.DataFrame(history)
    required_cols = {"power", "Tf", "pressure", "h2_produced", "core_damage"}
    if not required_cols.issubset(df.columns):
        st.warning("Insufficient simulation data for barrier analysis.")
        return

    # Build barrier time-series
    records = []
    for _, row in df.iterrows():
        b = BarrierState.from_sim(
            float(row.get("power", 0)),
            float(row.get("Tf", 20)),
            float(row.get("pressure", 150)),
            float(row.get("h2_produced", 0)),
            str(row.get("core_damage", "Intact")),
        )
        records.append({
            "t": float(row["t"]),
            "Fuel Pellet":       b.fuel_pellet,
            "Cladding":          b.cladding,
            "Primary Boundary":  b.primary_bound,
            "Containment":       b.containment,
            "Overall DiD":       b.overall(),
        })

    bdf = pd.DataFrame(records)

    # Stacked area integrity chart
    fig = go.Figure()
    colour_map = {
        "Fuel Pellet":      ("#fbbf24", "rgba(251,191,36,0.15)"),
        "Cladding":         ("#f97316", "rgba(249,115,22,0.15)"),
        "Primary Boundary": ("#ef4444", "rgba(239,68,68,0.15)"),
        "Containment":      ("#8b5cf6", "rgba(139,92,246,0.15)"),
    }
    for col, (line_color, fill_color) in colour_map.items():
        fig.add_trace(go.Scatter(
            x=bdf["t"], y=bdf[col] * 100,
            name=col, mode="lines",
            line=dict(color=line_color, width=2),
            fill="tozeroy",
            fillcolor=fill_color,
        ))
    fig.add_trace(go.Scatter(
        x=bdf["t"], y=bdf["Overall DiD"] * 100,
        name="Overall DiD Product",
        line=dict(color="white", width=3, dash="dash"),
    ))
    fig.update_layout(
        title="Barrier Integrity Over Time [%]",
        xaxis_title="Time [s]",
        yaxis_title="Integrity [%]",
        yaxis=dict(range=[0, 105]),
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Current snapshot gauge row
    last = records[-1]
    cols = st.columns(5)
    for i, (name, color_hex) in enumerate([
        ("Fuel Pellet",      "#fbbf24"),
        ("Cladding",         "#f97316"),
        ("Primary Boundary", "#ef4444"),
        ("Containment",      "#8b5cf6"),
        ("Overall DiD",      "#ffffff"),
    ]):
        val = last[name] * 100
        status = "✅ Intact" if val > 90 else ("⚠️ Degraded" if val > 30 else "❌ Failed")
        cols[i].metric(name, f"{val:.0f}%", status)

    # Explainer accordion
    with st.expander("📖 What do these barriers mean?"):
        st.markdown("""
| Barrier | Physical Form | Failure Mode |
|---------|--------------|--------------|
| **Fuel Pellet** | UO₂ ceramic matrix traps >98% of fission products | Fuel melting (T > 2800°C), cracking at > 1800°C |
| **Cladding** | Zircaloy tube prevents release to coolant | Oxidation > 1200°C, mechanical failure at pressure |
| **Primary Boundary** | Reactor vessel + primary piping | LOCA (pipe break), overpressure failure |
| **Containment** | Reinforced concrete/steel building | H₂ combustion, steam explosion, basemat penetration |

**IAEA Safety Principle SF-1**: Defence-in-Depth ensures that no single failure leads to unacceptable consequences.
        """)


# ══════════════════════════════════════════════════════════════════
#  MODULE 2 – PASSIVE SAFETY SYSTEMS DESIGNER
# ══════════════════════════════════════════════════════════════════

PASSIVE_SYSTEMS = {
    "AP1000 PRHR (Passive Residual Heat Removal)": {
        "desc": "C-shaped heat exchanger in IRWST. Gravity/natural circulation. No pumps needed.",
        "cooling_boost": 0.65,   # fraction of nominal cooling restored
        "activation_delay": 30,  # seconds after initiating event
        "coping_time_h": 72,     # hours without AC power
        "failure_prob": 0.001,
    },
    "AP1000 Core Make-up Tanks (CMT)": {
        "desc": "Pressurised borated water tanks, gravity drain on low pressure signal.",
        "cooling_boost": 0.40,
        "activation_delay": 60,
        "coping_time_h": 72,
        "failure_prob": 0.002,
    },
    "BWRX-300 Gravity-Fed Pool": {
        "desc": "6,000-tonne water pool above core. Passive flooding on vessel breach.",
        "cooling_boost": 0.80,
        "activation_delay": 45,
        "coping_time_h": 168,
        "failure_prob": 0.0005,
    },
    "ESBWR ICS (Isolation Condenser)": {
        "desc": "Passive decay heat removal via heat exchangers immersed in large pool.",
        "cooling_boost": 0.55,
        "activation_delay": 20,
        "coping_time_h": 72,
        "failure_prob": 0.003,
    },
    "Molten Salt Natural Circulation": {
        "desc": "Liquid fuel salt convects heat to passive air-cooled DRACS loop.",
        "cooling_boost": 0.90,
        "activation_delay": 0,
        "coping_time_h": 9999,
        "failure_prob": 0.0001,
    },
    "Active ECCS (Gen II reference)": {
        "desc": "Diesel-backed high-pressure injection pumps. Requires AC power.",
        "cooling_boost": 0.70,
        "activation_delay": 15,
        "coping_time_h": 8,     # limited by diesel fuel
        "failure_prob": 0.05,   # much higher – AC power dependency
    },
}

def render_passive_safety_designer(history: list):
    st.subheader("🛡️ Passive Safety Systems Designer & Benchmarker")
    st.caption(
        "Compare passive vs active safety systems response to the current simulation's "
        "accident progression. Passive systems are a hallmark of Generation III+ / IV design."
    )

    selected = st.multiselect(
        "Select systems to compare",
        list(PASSIVE_SYSTEMS.keys()),
        default=list(PASSIVE_SYSTEMS.keys())[:3],
    )

    if not selected:
        st.info("Select at least one system above.")
        return

    # Summary comparison table
    rows = []
    for name in selected:
        s = PASSIVE_SYSTEMS[name]
        rows.append({
            "System":            name,
            "Cooling Restored":  f"{s['cooling_boost']*100:.0f}%",
            "Activation Delay":  f"{s['activation_delay']}s",
            "Coping Time":       f"{s['coping_time_h']}h" if s['coping_time_h'] < 9000 else "∞",
            "Failure Prob":      f"{s['failure_prob']:.4f}",
            "Description":       s["desc"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if not history:
        st.info("Run a simulation to see projected accident progression with each system.")
        return

    df = pd.DataFrame(history)
    if "Tf" not in df.columns:
        return

    # Simulate counterfactual: what would Tf look like with each system active?
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["t"], y=df["Tf"],
        name="No Passive System (Baseline)",
        line=dict(color="red", width=3),
    ))

    colours_p = ["#00ff9d", "#00b4ff", "#fbbf24", "#c084fc", "#fb923c", "#34d399"]
    for idx, name in enumerate(selected):
        s    = PASSIVE_SYSTEMS[name]
        delay = s["activation_delay"]
        boost = s["cooling_boost"]
        # Simple counterfactual: after activation delay, cooling is boosted
        t_arr  = df["t"].values
        Tf_arr = df["Tf"].values.copy()
        Tf_cf  = Tf_arr.copy()
        for i in range(1, len(t_arr)):
            if t_arr[i] > delay:
                # With better cooling, temperature decays faster
                delta = (Tf_cf[i-1] - 305) * boost * 0.02
                Tf_cf[i] = max(300.0, Tf_cf[i-1] - delta)
            else:
                Tf_cf[i] = Tf_arr[i]

        fig.add_trace(go.Scatter(
            x=t_arr, y=Tf_cf,
            name=name,
            line=dict(color=colours_p[idx % len(colours_p)], width=2, dash="dot"),
        ))

    # Safety limit lines
    for limit, label, color in [
        (1200, "Cladding Failure Onset", "orange"),
        (1800, "Melt Onset", "red"),
        (2200, "Partial Meltdown", "darkred"),
    ]:
        fig.add_hline(y=limit, line=dict(color=color, dash="dash", width=1),
                      annotation_text=label, annotation_position="bottom right")

    fig.update_layout(
        title="Counterfactual Fuel Temperature: With vs Without Passive Systems",
        xaxis_title="Time [s]",
        yaxis_title="Fuel Temperature [°C]",
        template="plotly_dark",
        height=480,
        legend=dict(orientation="h", y=1.02, font=dict(size=9)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Coping time bar chart
    fig2 = go.Figure(go.Bar(
        x=[PASSIVE_SYSTEMS[n]["coping_time_h"] if PASSIVE_SYSTEMS[n]["coping_time_h"] < 9000
           else 200 for n in selected],
        y=selected,
        orientation="h",
        marker_color=colours_p[:len(selected)],
        text=[f"{PASSIVE_SYSTEMS[n]['coping_time_h']}h" if PASSIVE_SYSTEMS[n]['coping_time_h'] < 9000
              else "∞" for n in selected],
        textposition="outside",
    ))
    fig2.add_vline(x=72, line=dict(color="yellow", dash="dash"),
                   annotation_text="IAEA 72h coping target")
    fig2.update_layout(
        title="Coping Time Without AC Power [hours]",
        template="plotly_dark", height=300,
        xaxis=dict(range=[0, 220]),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  MODULE 3 – PROBABILISTIC RISK ASSESSMENT (PRA)
# ══════════════════════════════════════════════════════════════════

# Each event tree: initiating event → sequence of safety system successes/failures
# Each branch has (label, success_prob, consequence_if_fail)

EVENT_TREES: Dict[str, dict] = {
    "Station Blackout (SBO)": {
        "initiating_event": "Loss of off-site power",
        "initiating_freq": 0.01,   # per reactor-year
        "branches": [
            ("Diesel Generator Start",     0.97, "LOOP without DG"),
            ("Battery-backed I&C",          0.999, "Loss of monitoring"),
            ("ECCS/Passive Cooling",        0.95, "Core uncovery"),
            ("Decay Heat Removal (72h)",    0.90, "Core damage"),
            ("Containment Integrity",       0.95, "Large release"),
        ],
    },
    "Large-Break LOCA": {
        "initiating_event": "Double-ended guillotine break of main coolant pipe",
        "initiating_freq": 5e-5,
        "branches": [
            ("SCRAM (automatic)",           0.9999, "Loss of shutdown"),
            ("High-Pressure ECCS",          0.97,   "High-pressure core damage"),
            ("Low-Pressure ECCS",           0.95,   "Core uncovery"),
            ("Passive Accumulators",        0.999,  "Reflooding failure"),
            ("Containment Spray",           0.93,   "Pressure buildup"),
            ("Hydrogen Igniters",           0.88,   "H2 deflagration"),
        ],
    },
    "Anticipated Transient Without SCRAM (ATWS)": {
        "initiating_event": "Turbine trip + SCRAM failure",
        "initiating_freq": 1e-4,
        "branches": [
            ("Alternate Rod Insertion",    0.95,  "Uncontrolled power increase"),
            ("Emergency Boration",         0.92,  "Reactivity control failure"),
            ("RPS Logic Backup",           0.98,  "No shutdown signal"),
            ("High-Pressure Injection",    0.95,  "Core damage"),
            ("Containment",               0.90,  "Large release"),
        ],
    },
    "Spent Fuel Pool Loss of Cooling": {
        "initiating_event": "SFP makeup water system failure",
        "initiating_freq": 0.003,
        "branches": [
            ("Alternate Makeup Supply",    0.92,  "SFP uncovery"),
            ("Portable Pump Deployment",   0.85,  "Zircaloy fire"),
            ("Passive SFP Cooling",        0.80,  "Large radioactive release"),
            ("Offsite Power Restoration",  0.70,  "Extended SFP boiling"),
        ],
    },
}

def render_pra_module():
    st.subheader("🎲 Probabilistic Risk Assessment (PRA) – Event Tree Analysis")
    st.caption(
        "PRA quantifies risk as Frequency × Consequence. Modern reactor licensing requires "
        "Core Damage Frequency (CDF) < 10⁻⁵ per reactor-year and Large Early Release "
        "Frequency (LERF) < 10⁻⁶. This module walks through event trees for major initiating events."
    )

    tree_name = st.selectbox("Select Initiating Event", list(EVENT_TREES.keys()))
    if not tree_name:
        tree_name = list(EVENT_TREES.keys())[0]
    tree      = EVENT_TREES[tree_name]

    st.markdown(f"**Initiating Event:** {tree['initiating_event']}  "
                f"│  **Frequency:** {tree['initiating_freq']:.2e} /reactor-year")

    # Allow override of success probabilities
    st.markdown("**Safety System Success Probabilities** (adjust for design comparison):")
    branches  = tree["branches"]
    overrides = []
    cols = st.columns(min(3, len(branches)))
    for i, (label, default_p, fail_consequence) in enumerate(branches):
        raw = cols[i % 3].slider(label, 0.0, 1.0, default_p, 0.01, key=f"pra_{tree_name}_{i}")
        try:    p = float(raw)
        except: p = default_p
        overrides.append((label, p, fail_consequence))

    # Walk the event tree
    freq = tree["initiating_freq"]
    path_freq   = freq
    cdf         = 0.0
    rows        = []
    cumulative_failure_p = 1.0

    for label, p, consequence in overrides:
        fail_freq = path_freq * (1 - p)
        cdf      += fail_freq
        rows.append({
            "Safety System":          label,
            "Success Probability":    p,
            "Failure Probability":    1 - p,
            "Sequence Frequency":     fail_freq,
            "Consequence if Failure": consequence,
        })
        path_freq *= p   # only surviving path continues

    lerf = cdf * 0.1    # simplified: assume 10% of core damage → large release

    # Metrics
    c1, c2, c3 = st.columns(3)
    cdf_color  = "normal" if cdf < 1e-4 else ("off" if cdf < 1e-3 else "inverse")
    c1.metric("Core Damage Frequency (CDF)", f"{cdf:.2e} /ry",
              "✅ < 1e-4 target" if cdf < 1e-4 else "⚠️ Exceeds target")
    c2.metric("Large Early Release Freq (LERF)", f"{lerf:.2e} /ry",
              "✅ < 1e-5 target" if lerf < 1e-5 else "⚠️ Exceeds target")
    c3.metric("Conditional Core Damage Prob", f"{cdf / freq:.3%}",
              "Given initiating event occurs")

    st.dataframe(pd.DataFrame(rows).style.format({
        "Success Probability": "{:.3f}",
        "Failure Probability": "{:.4f}",
        "Sequence Frequency":  "{:.2e}",
    }), use_container_width=True, hide_index=True)

    # Tornado chart: which system contributes most to CDF?
    contribution = [(r["Safety System"], r["Sequence Frequency"]) for r in rows]
    contribution.sort(key=lambda x: x[1], reverse=True)
    fig = go.Figure(go.Bar(
        y=[c[0] for c in contribution],
        x=[c[1] for c in contribution],
        orientation="h",
        marker_color=["#ef4444" if c[1] > cdf * 0.3 else "#fbbf24"
                      for c in contribution],
        text=[f"{c[1]:.2e}" for c in contribution],
        textposition="outside",
    ))
    fig.update_layout(
        title="Safety System CDF Contribution (Tornado Chart)",
        xaxis_title="Sequence Frequency [/reactor-year]",
        template="plotly_dark",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 PRA Regulatory Context"):
        st.markdown("""
| Metric | Gen II Target | Gen III+ Target | Gen IV Target |
|--------|--------------|-----------------|---------------|
| CDF [/ry] | < 10⁻⁴ | < 10⁻⁵ | < 10⁻⁶ |
| LERF [/ry] | < 10⁻⁵ | < 10⁻⁶ | < 10⁻⁷ |
| Coping time w/o AC | 4-8h | 72h | Infinite (passive) |

**Key insight**: The dominant contributor to CDF is almost always the weakest safety system.
Improving a system from 90% → 99% reliability reduces that system's contribution by 10×.
        """)


# ══════════════════════════════════════════════════════════════════
#  MODULE 4 – FEEDBACK COEFFICIENT STABILITY MAP
# ══════════════════════════════════════════════════════════════════

def render_stability_map():
    st.subheader("🗺️ Reactivity Feedback Coefficient Stability Map")
    st.caption(
        "The stability of a reactor is governed by its feedback coefficients. "
        "Negative feedback is essential for inherent safety. This tool maps the "
        "stability boundary and lets you explore what made RBMK dangerous."
    )

    c1, c2 = st.columns(2)
    raw_d = c1.slider("Doppler coefficient [pcm/°C]",  -8.0, +2.0, (-3.0, 0.0), 0.1)
    raw_v = c2.slider("Void coefficient [pcm/% void]", -300.0, +500.0, (-150.0, 200.0), 10.0)
    # Handle both tuple (real Streamlit) and scalar (mock) returns
    alpha_d_range = raw_d if isinstance(raw_d, (list, tuple)) else (-3.0, 0.0)
    alpha_v_range = raw_v if isinstance(raw_v, (list, tuple)) else (-150.0, 200.0)

    # Generate stability grid
    alpha_d_vals = np.linspace(alpha_d_range[0], alpha_d_range[1], 60)
    alpha_v_vals = np.linspace(alpha_v_range[0], alpha_v_range[1], 60)
    AD, AV       = np.meshgrid(alpha_d_vals, alpha_v_vals)

    # Simplified stability criterion:
    # Reactor is stable if: alpha_d + alpha_v * void_fraction < 0 (at max void)
    # For BWR: void can reach 70%; for RBMK: 80%
    max_void    = 0.7
    stability_z = AD + AV * max_void   # < 0 → stable

    fig = go.Figure(go.Heatmap(
        z=stability_z,
        x=alpha_d_vals,
        y=alpha_v_vals,
        colorscale=[[0, "#00ff9d"], [0.45, "#00ff9d"],
                    [0.5, "#ffffff"],
                    [0.55, "#ef4444"], [1, "#7f0000"]],
        zmid=0,
        colorbar=dict(title="Stability Index<br>(< 0 = stable)"),
    ))

    # Overlay reactor operating points
    reactors = {
        "PWR":   (-3.0, 0,     "#00b4ff",  "circle"),
        "BWR":   (-1.8, -140,  "#00ff9d",  "square"),
        "RBMK":  (-1.2, 350,   "#ff4d4d",  "x"),
        "SMR":   (-3.5, 0,     "#c084fc",  "diamond"),
        "CANDU": (-1.5, -100,  "#fbbf24",  "triangle-up"),
    }
    for name, (ad, av, color, symbol) in reactors.items():
        if (alpha_d_range[0] <= ad <= alpha_d_range[1] and
                alpha_v_range[0] <= av <= alpha_v_range[1]):
            fig.add_trace(go.Scatter(
                x=[ad], y=[av],
                mode="markers+text",
                name=name,
                text=[name],
                textposition="top center",
                marker=dict(size=16, color=color, symbol=symbol,
                            line=dict(color="white", width=2)),
            ))

    fig.add_shape(type="line", x0=alpha_d_range[0], x1=alpha_d_range[1],
                  y0=0, y1=0, line=dict(color="white", dash="dash", width=1))
    fig.add_shape(type="line", x0=0, x1=0,
                  y0=alpha_v_range[0], y1=alpha_v_range[1],
                  line=dict(color="white", dash="dash", width=1))

    fig.update_layout(
        title=f"Stability Map: Doppler vs Void Coefficient (max void={max_void*100:.0f}%)",
        xaxis_title="Doppler Coefficient α_D [pcm/°C]",
        yaxis_title="Void Coefficient α_V [pcm/% void]",
        template="plotly_dark",
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Reading the map:**
- 🟢 **Green region** = negative net feedback = inherently stable (self-limiting)
- 🔴 **Red region** = positive net feedback = runaway possible without active control
- **RBMK** falls in the red zone at low power — the void coefficient completely dominates
- **PWR/SMR** sit deep in the stable region — Doppler feedback alone is enough to self-limit a transient
    """)


# ══════════════════════════════════════════════════════════════════
#  MODULE 5 – CONTAINMENT PERFORMANCE & H2 RISK
# ══════════════════════════════════════════════════════════════════

def render_containment_module(history: list):
    st.subheader("🏠 Containment Performance & Hydrogen Risk")
    st.caption(
        "Containment is the last barrier. H₂ generated by zircaloy oxidation "
        "above 1200°C is the key risk (Fukushima Unit 1, 3, 4 explosions; "
        "TMI-2 hydrogen burn). This module tracks containment loading."
    )

    containment_type = st.selectbox("Containment Design", [
        "Large Dry PWR (cylindrical steel-lined)",
        "Ice Condenser PWR (smaller volume)",
        "Mark I BWR (torus/drywell – Fukushima type)",
        "Mark III BWR (over-pressure protection)",
        "AP1000 Steel Containment (passive cooling)",
        "SMR Integral Containment",
    ])

    # Containment volume and design pressure vary by type
    containment_params = {
        "Large Dry PWR (cylindrical steel-lined)":    {"V_m3": 70000, "P_design_bar": 4.3,  "H2_limit_vol_pct": 10.0},
        "Ice Condenser PWR (smaller volume)":         {"V_m3": 30000, "P_design_bar": 2.1,  "H2_limit_vol_pct": 5.0},
        "Mark I BWR (torus/drywell – Fukushima type)":{"V_m3": 11000, "P_design_bar": 5.3,  "H2_limit_vol_pct": 4.0},
        "Mark III BWR (over-pressure protection)":    {"V_m3": 50000, "P_design_bar": 3.6,  "H2_limit_vol_pct": 8.0},
        "AP1000 Steel Containment (passive cooling)": {"V_m3": 58000, "P_design_bar": 4.1,  "H2_limit_vol_pct": 10.0},
        "SMR Integral Containment":                   {"V_m3": 5000,  "P_design_bar": 10.0, "H2_limit_vol_pct": 10.0},
    }
    cparams = containment_params[containment_type]

    if not history:
        st.info("Run a simulation to see containment loading.")
        return

    df = pd.DataFrame(history)
    if "h2_produced" not in df.columns:
        return

    # H2 concentration in containment volume (assuming inert gas atmosphere initially)
    # H2 produced (arbitrary units from sim) → scale to kg → mol → volume fraction
    h2_kg      = df["h2_produced"] * 10.0   # scale factor (sim units → kg)
    h2_mol     = h2_kg / 0.002              # kg / (2g/mol)
    # Ideal gas at 1 bar, 100°C: V = nRT/P = n * 8.314 * 373 / 1e5 = n * 0.031 m³/mol
    h2_vol_m3  = h2_mol * 0.031
    h2_conc    = np.clip(h2_vol_m3 / cparams["V_m3"] * 100, 0, 100)   # vol %

    # Pressure estimate (simplified)
    if "pressure" in df.columns:
        press_rel = df["pressure"] / 150.0 * cparams["P_design_bar"]
    else:
        press_rel = pd.Series(np.ones(len(df)) * cparams["P_design_bar"] * 0.1)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("H₂ Concentration in Containment [vol %]",
                                        "Containment Pressure [bar]"))

    fig.add_trace(go.Scatter(x=df["t"], y=h2_conc, name="H₂ conc",
                              line=dict(color="#fbbf24", width=2)), row=1, col=1)
    fig.add_hline(y=4.0,  line=dict(color="orange", dash="dash"),
                  annotation_text="Flammability limit (4%)", row=1, col=1)
    fig.add_hline(y=cparams["H2_limit_vol_pct"],
                  line=dict(color="red", dash="dash"),
                  annotation_text=f"Design H₂ limit ({cparams['H2_limit_vol_pct']}%)", row=1, col=1)
    fig.add_hline(y=18.3, line=dict(color="darkred", dash="dash"),
                  annotation_text="Detonation limit (18.3%)", row=1, col=1)

    fig.add_trace(go.Scatter(x=df["t"], y=press_rel, name="Containment P",
                              line=dict(color="#c084fc", width=2)), row=2, col=1)
    fig.add_hline(y=cparams["P_design_bar"],
                  line=dict(color="red", dash="dash"),
                  annotation_text=f"Design pressure ({cparams['P_design_bar']} bar)", row=2, col=1)

    fig.update_layout(height=500, template="plotly_dark",
                      title=f"Containment Loading – {containment_type}")
    st.plotly_chart(fig, use_container_width=True)

    # Current status
    cur_h2  = float(h2_conc.iloc[-1]) if len(h2_conc) else 0.0
    cur_p   = float(press_rel.iloc[-1]) if len(press_rel) else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("H₂ Concentration", f"{cur_h2:.2f} vol%",
              "✅ Safe" if cur_h2 < 4 else ("⚠️ Flammable" if cur_h2 < 18.3 else "🚨 Detonation risk"))
    c2.metric("Containment Pressure", f"{cur_p:.2f} bar",
              "✅ Normal" if cur_p < cparams["P_design_bar"] * 0.7 else "⚠️ Elevated")
    c3.metric("H₂ Igniters Status",
              "✅ Passive autocatalytic" if "AP1000" in containment_type or "SMR" in containment_type
              else "⚡ Active igniters required")


# ══════════════════════════════════════════════════════════════════
#  MODULE 6 – SEVERE ACCIDENT PROGRESSION TRACKER
# ══════════════════════════════════════════════════════════════════

SAP_MILESTONES = [
    ("Core uncovery begins",       "Tf > 650°C sustained",    650,   "Tf"),
    ("Cladding oxidation onset",   "Tf > 1200°C",            1200,   "Tf"),
    ("Fuel damage / H₂ generation","Tf > 1400°C",            1400,   "Tf"),
    ("Partial core melt",          "Tf > 1800°C",            1800,   "Tf"),
    ("In-vessel melt progression", "Tf > 2200°C",            2200,   "Tf"),
    ("Lower head challenge",       "Tf > 2700°C",            2700,   "Tf"),
    ("Ex-vessel corium (MCCI)",    "Tf > 3000°C",            3000,   "Tf"),
    ("High power excursion",       "Power > 120%",             1.2,  "power"),
    ("Prompt criticality risk",    "Power > 300%",             3.0,  "power"),
]

def render_sap_tracker(history: list):
    st.subheader("⚠️ Severe Accident Progression (SAP) Tracker")
    st.caption(
        "Maps the sequence from initiating event through core damage progression. "
        "Based on TMI-2, Fukushima Daiichi, and Chernobyl accident phenomenology. "
        "Key design goal: arrest progression at the earliest possible stage."
    )

    if not history:
        st.info("Run an accident scenario to see progression tracking.")
        return

    df = pd.DataFrame(history)

    # Check which milestones were crossed
    milestone_results = []
    for (name, condition, threshold, param) in SAP_MILESTONES:
        if param not in df.columns:
            crossed = False
            t_cross = None
        else:
            exceeded = df[df[param] >= threshold]
            crossed  = len(exceeded) > 0
            t_cross  = float(exceeded["t"].iloc[0]) if crossed else None
        milestone_results.append({
            "Milestone":   name,
            "Trigger":     condition,
            "Reached":     "✅ Arrested" if not crossed else "🔴 REACHED",
            "Time [s]":    f"{t_cross:.1f}" if t_cross is not None else "—",
            "t_cross":     t_cross,
        })

    # Timeline visualisation
    crossed_milestones = [m for m in milestone_results if m["t_cross"] is not None]
    if crossed_milestones:
        fig = go.Figure()
        for i, m in enumerate(crossed_milestones):
            fig.add_trace(go.Scatter(
                x=[m["t_cross"]],
                y=[i],
                mode="markers+text",
                text=[m["Milestone"]],
                textposition="middle right",
                marker=dict(size=14, color="#ef4444",
                            symbol="x", line=dict(color="white", width=2)),
                showlegend=False,
            ))
        fig.update_layout(
            title="Accident Milestone Timeline",
            xaxis_title="Time [s]",
            yaxis=dict(visible=False),
            template="plotly_dark",
            height=max(250, 60 * len(crossed_milestones)),
            margin=dict(l=10, r=300, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    display_df = pd.DataFrame([{k: v for k, v in m.items() if k != "t_cross"}
                                 for m in milestone_results])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # IVR assessment
    st.markdown("### In-Vessel Retention (IVR) Assessment")
    st.markdown("""
    **IVR Strategy** (used in AP1000, VVER-440 mod. 213): flood the reactor cavity to cool the vessel
    exterior, preventing ex-vessel corium release. Critical success parameter: heat flux through
    vessel wall must stay below Critical Heat Flux (CHF).
    """)
    max_Tf = float(df["Tf"].max()) if "Tf" in df.columns else 0.0
    ivr_possible = max_Tf < 2700
    st.metric("IVR Feasibility",
              "✅ Feasible — corium may be retained in-vessel" if ivr_possible
              else "❌ Unlikely — lower head failure probable",
              f"Peak fuel T = {max_Tf:.0f}°C")


# ══════════════════════════════════════════════════════════════════
#  MODULE 7 – EOP TRAINER
# ══════════════════════════════════════════════════════════════════

EOPs = {
    "E-0: Reactor Trip / SCRAM": {
        "trigger": "Any reactor trip signal",
        "steps": [
            "1. Verify reactor trip (rod bottom lights, power < 5%)",
            "2. Verify turbine trip",
            "3. Verify Reactor Coolant System (RCS) pressure stabilising",
            "4. Verify emergency feedwater (EFW) initiation on low steam generator level",
            "5. Monitor RCS subcooling margin > 17°C",
            "6. Monitor containment parameters — no unexplained rise",
            "7. Transition to E-1 (Loss of Reactor or Secondary Coolant) if RCS inventory lost",
        ],
        "critical_actions": ["Verify rod insertion", "Maintain RCS subcooling"],
        "time_critical": True,
    },
    "E-1: Loss of Reactor or Secondary Coolant (LOCA)": {
        "trigger": "Unidentified or identified RCS leakage",
        "steps": [
            "1. Verify ECCS actuation — check injection flow",
            "2. Identify LOCA location (primary vs secondary) by radiation monitors",
            "3. Maintain pressuriser level — do NOT throttle ECCS prematurely",
            "4. Monitor core exit thermocouple — must not exceed 343°C",
            "5. Maintain containment isolation",
            "6. If sump level rising — shift to recirculation mode",
            "7. Initiate boration if subcriticality not maintained",
        ],
        "critical_actions": ["Do NOT throttle ECCS (TMI lesson)", "Maintain core covered"],
        "time_critical": True,
    },
    "E-2: Faulted Steam Generator Isolation": {
        "trigger": "Steam generator tube rupture or steam line break",
        "steps": [
            "1. Identify faulted steam generator (radiation, level, flow mismatch)",
            "2. Isolate faulted SG feedwater and steam lines",
            "3. Cool down via intact steam generators",
            "4. Maintain RCS subcooling margin",
            "5. Check radioactivity release pathway — secondary side contaminated?",
            "6. Initiate site emergency plan if primary-to-secondary leakage confirmed",
        ],
        "critical_actions": ["Identify correct faulted SG", "Cool via intact SG only"],
        "time_critical": False,
    },
    "FR-H.1: Response to Loss of Secondary Heat Sink": {
        "trigger": "All steam generators dry / no feedwater",
        "steps": [
            "1. Verify all EFW paths — open all available paths",
            "2. Attempt alternate water sources to steam generators",
            "3. If no SG available — initiate RCS cooldown by other means",
            "4. Activate passive residual heat removal (PRHR) if available",
            "5. Prepare for feed-and-bleed if subcooling lost",
            "6. Verify decay heat removal pathway",
        ],
        "critical_actions": ["PRHR activation", "Feed-and-bleed decision"],
        "time_critical": True,
    },
}

def render_eop_trainer(history: list):
    st.subheader("📋 Emergency Operating Procedure (EOP) Trainer")
    st.caption(
        "EOPs guide operators through accident response. Modern symptom-based EOPs "
        "(SBEOPs) respond to reactor symptoms rather than specific events, "
        "addressing the TMI lesson where operators misdiagnosed the event."
    )

    selected_eop = st.selectbox("Select EOP", list(EOPs.keys()))
    eop          = EOPs[selected_eop]

    col_pair = st.columns([2, 1])
    c1, c2   = col_pair[0], col_pair[1]
    # Use direct calls rather than context manager for cross-compat
    c1.markdown(f"**Trigger Condition:** {eop['trigger']}")
    c1.markdown("**Procedure Steps:**")
    for step in eop["steps"]:
        c1.markdown(f"- {step}")
    for ca in eop["critical_actions"]:
        c2.error(ca)
    if eop["time_critical"]:
        c2.warning("⏱️ Time-critical procedure — operator has < 30 minutes to diagnose")
    else:
        c2.info("📋 Non-time-critical — methodical execution acceptable")

    # Auto-diagnose from simulation state
    if history:
        df   = pd.DataFrame(history)
        last = df.iloc[-1]
        power    = float(last.get("power", 0))
        Tf       = float(last.get("Tf", 300))
        pressure = float(last.get("pressure", 150))
        scram    = bool(last.get("scram_active", False))

        st.markdown("---")
        st.markdown("### 🩺 Automated Symptom Diagnosis (from current simulation)")
        symptoms = []
        if scram:                     symptoms.append("✅ Reactor SCRAM confirmed")
        if power > 1.05:              symptoms.append("⚠️ Power above setpoint")
        if Tf > 750:                  symptoms.append("🔴 Fuel temperature high — SCRAM criterion")
        if pressure > 165:            symptoms.append("🔴 RCS pressure high")
        if pressure < 120:            symptoms.append("🔴 RCS pressure low — possible LOCA")
        if Tf < 400 and scram:        symptoms.append("ℹ️ Core cooling adequate post-SCRAM")
        if not symptoms:              symptoms.append("ℹ️ No abnormal symptoms detected")

        for s in symptoms:
            st.markdown(f"- {s}")

        # Recommend EOP
        if pressure < 120:      recommended = "E-1: Loss of Reactor or Secondary Coolant (LOCA)"
        elif scram and Tf > 650: recommended = "E-1: Loss of Reactor or Secondary Coolant (LOCA)"
        elif scram:              recommended = "E-0: Reactor Trip / SCRAM"
        else:                    recommended = "Normal Operating Procedures"
        st.success(f"**Recommended EOP:** {recommended}")


# ══════════════════════════════════════════════════════════════════
#  MODULE 8 – REGULATORY SAFETY MARGIN TRACKER (SAFDLs)
# ══════════════════════════════════════════════════════════════════

SAFETY_LIMITS = {
    "Peak Fuel Centerline Temperature": {"limit": 1482, "unit": "°C", "param": "Tf",
                                          "note": "UO₂ melting onset – 10CFR50 limit"},
    "Peak Cladding Temperature (LOCA)": {"limit": 1204, "unit": "°C", "param": "Tf",
                                          "note": "10CFR50 Appendix K – Zircaloy oxidation limit"},
    "Peak Linear Heat Rate":            {"limit": 59.0, "unit": "kW/m", "param": "power",
                                          "scale": 59.0, "note": "Fuel performance limit"},
    "RCS Pressure (normal ops)":        {"limit": 172.4, "unit": "bar", "param": "pressure",
                                          "note": "PWR design pressure limit"},
    "Maximum Overpower":                {"limit": 1.20, "unit": "fraction", "param": "power",
                                          "note": "Overpower protection setpoint"},
}

def render_margin_tracker(history: list):
    st.subheader("📏 Regulatory Safety Margin Tracker")
    st.caption(
        "Specified Acceptable Fuel Design Limits (SAFDLs) and Technical Specification "
        "Action Levels define the envelope within which a reactor must operate. "
        "This tracker shows real-time distance from each limit."
    )

    if not history:
        st.info("Run a simulation to track safety margins.")
        return

    df   = pd.DataFrame(history)
    last = df.iloc[-1]

    # Margin gauges
    fig = make_subplots(rows=1, cols=len(SAFETY_LIMITS),
                        specs=[[{"type": "indicator"}] * len(SAFETY_LIMITS)])
    for i, (name, info) in enumerate(SAFETY_LIMITS.items()):
        param = info["param"]
        limit = info["limit"]
        if param not in df.columns:
            continue
        val = float(last.get(param, 0))
        if "scale" in info:
            val = val * info["scale"]
        pct_of_limit = min(val / limit * 100, 100) if limit > 0 else 0

        # Color based on margin
        color = ("#00cc96" if pct_of_limit < 70
                 else "#ffa500" if pct_of_limit < 90
                 else "#ef4444")

        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=pct_of_limit,
            title={"text": name[:30], "font": {"size": 10}},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": color},
                "steps": [{"range": [0, 70],  "color": "#1a3a1a"},
                           {"range": [70, 90], "color": "#3a2a00"},
                           {"range": [90, 100],"color": "#3a0000"}],
                "threshold": {"line": {"color": "red", "width": 3},
                              "thickness": 0.8, "value": 100},
            },
            number={"suffix": "% of limit", "font": {"size": 14}},
        ), row=1, col=i+1)

    fig.update_layout(
        height=300,
        template="plotly_dark",
        title="Safety Limit Utilisation [% of regulatory limit]",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Time-history of margin usage
    if "Tf" in df.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["t"],
            y=df["Tf"] / SAFETY_LIMITS["Peak Fuel Centerline Temperature"]["limit"] * 100,
            name="Fuel T / SAFDL", line=dict(color="#ff9500"),
        ))
        if "pressure" in df.columns:
            fig2.add_trace(go.Scatter(
                x=df["t"],
                y=df["pressure"] / SAFETY_LIMITS["RCS Pressure (normal ops)"]["limit"] * 100,
                name="Pressure / SAFDL", line=dict(color="#c084fc"),
            ))
        fig2.add_trace(go.Scatter(
            x=df["t"],
            y=df["power"] / SAFETY_LIMITS["Maximum Overpower"]["limit"] * 100,
            name="Power / SAFDL", line=dict(color="#00ff9d"),
        ))
        fig2.add_hline(y=100, line=dict(color="red", dash="dash", width=2),
                       annotation_text="SAFDL Limit")
        fig2.add_hline(y=90, line=dict(color="orange", dash="dash", width=1),
                       annotation_text="Action Level (90%)")
        fig2.update_layout(
            title="Safety Limit Utilisation Over Time",
            xaxis_title="Time [s]",
            yaxis_title="% of Regulatory Limit",
            yaxis=dict(range=[0, 120]),
            template="plotly_dark",
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  MODULE 9 – SMR vs LARGE LWR SAFETY ADVANTAGE CALCULATOR
# ══════════════════════════════════════════════════════════════════

REACTOR_DESIGNS = {
    "Gen II PWR (1000 MWe)": {
        "power_MWe": 1000, "power_MWth": 3000,
        "decay_heat_kW_at_1h": 18000,   # ~0.6% of thermal power
        "nat_circulation_capacity_pct": 5,
        "coping_time_h": 4,
        "containment_volume_m3": 70000,
        "source_term_Ci_Cs137": 8e8,
        "passive_safety": False,
        "CDF_per_ry": 1e-4,
        "color": "#ef4444",
    },
    "Gen III+ PWR (AP1000, 1117 MWe)": {
        "power_MWe": 1117, "power_MWth": 3415,
        "decay_heat_kW_at_1h": 20490,
        "nat_circulation_capacity_pct": 100,
        "coping_time_h": 72,
        "containment_volume_m3": 58000,
        "source_term_Ci_Cs137": 6e7,
        "passive_safety": True,
        "CDF_per_ry": 5.1e-7,
        "color": "#fbbf24",
    },
    "SMR – NuScale (77 MWe)": {
        "power_MWe": 77, "power_MWth": 250,
        "decay_heat_kW_at_1h": 1500,
        "nat_circulation_capacity_pct": 100,
        "coping_time_h": 72,
        "containment_volume_m3": 700,
        "source_term_Ci_Cs137": 3e6,
        "passive_safety": True,
        "CDF_per_ry": 2.9e-8,
        "color": "#00ff9d",
    },
    "SMR – BWRX-300 (300 MWe)": {
        "power_MWe": 300, "power_MWth": 870,
        "decay_heat_kW_at_1h": 5220,
        "nat_circulation_capacity_pct": 100,
        "coping_time_h": 168,
        "containment_volume_m3": 3000,
        "source_term_Ci_Cs137": 2e7,
        "passive_safety": True,
        "CDF_per_ry": 1e-8,
        "color": "#00b4ff",
    },
    "MSR – Terrestrial Energy (195 MWe)": {
        "power_MWe": 195, "power_MWth": 595,
        "decay_heat_kW_at_1h": 3570,
        "nat_circulation_capacity_pct": 100,
        "coping_time_h": 9999,   # infinite passive
        "containment_volume_m3": 1500,
        "source_term_Ci_Cs137": 1e5,   # online processing removes most Cs
        "passive_safety": True,
        "CDF_per_ry": 1e-9,
        "color": "#c084fc",
    },
}

def render_smr_advantage():
    st.subheader("⚡ SMR vs Large LWR Safety Advantage Calculator")
    st.caption(
        "Small Modular Reactors achieve better safety metrics through smaller "
        "source term, higher surface-to-volume ratio enabling natural circulation, "
        "and simpler passive safety systems. This module quantifies the differences."
    )

    selected = st.multiselect(
        "Select reactor designs to compare",
        list(REACTOR_DESIGNS.keys()),
        default=list(REACTOR_DESIGNS.keys()),
    )
    if not selected:
        return

    designs = {k: REACTOR_DESIGNS[k] for k in selected}

    # Radar / spider chart of normalised safety metrics
    categories = ["Low Decay Heat\n(per MWe)", "Nat Circulation", "Coping Time",
                  "Low Source Term", "Low CDF", "Passive Safety"]

    def normalise(vals):
        vmax = max(vals)
        vmin = min(vals)
        if vmax == vmin:
            return [0.5] * len(vals)
        return [(v - vmin) / (vmax - vmin) for v in vals]

    decay_heat_per_MWe = [d["decay_heat_kW_at_1h"] / d["power_MWe"] for d in designs.values()]
    coping             = [min(d["coping_time_h"], 200) for d in designs.values()]
    source_term        = [d["source_term_Ci_Cs137"] for d in designs.values()]
    cdf                = [d["CDF_per_ry"] for d in designs.values()]
    nat_circ           = [d["nat_circulation_capacity_pct"] for d in designs.values()]
    passive            = [1.0 if d["passive_safety"] else 0.0 for d in designs.values()]

    # Normalise: higher = better (invert where lower is better)
    n_decay   = [1 - v for v in normalise(decay_heat_per_MWe)]
    n_circ    = normalise(nat_circ)
    n_coping  = normalise(coping)
    n_source  = [1 - v for v in normalise(source_term)]
    n_cdf     = [1 - v for v in normalise(cdf)]
    n_passive = passive

    fig = go.Figure()
    # Hex → rgba fill colors for radar chart
    hex_to_rgba = {
        "#ef4444": "rgba(239,68,68,0.18)",
        "#fbbf24": "rgba(251,191,36,0.18)",
        "#00ff9d": "rgba(0,255,157,0.18)",
        "#00b4ff": "rgba(0,180,255,0.18)",
        "#c084fc": "rgba(192,132,252,0.18)",
    }
    for i, (name, d) in enumerate(designs.items()):
        vals = [n_decay[i], n_circ[i], n_coping[i], n_source[i], n_cdf[i], n_passive[i]]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself", name=name,
            line=dict(color=d["color"], width=2),
            fillcolor=hex_to_rgba.get(d["color"], "rgba(128,128,128,0.18)"),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Safety Attribute Radar (higher = better)",
        template="plotly_dark",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    rows = []
    for name, d in designs.items():
        rows.append({
            "Design":              name,
            "Power [MWe]":         d["power_MWe"],
            "Decay Heat @1h [kW]": f"{d['decay_heat_kW_at_1h']:,}",
            "Decay Heat/MWe":      f"{d['decay_heat_kW_at_1h']/d['power_MWe']:.1f} kW/MWe",
            "Nat Circ Capacity":   f"{d['nat_circulation_capacity_pct']}%",
            "Coping Time":         f"{d['coping_time_h']}h" if d["coping_time_h"] < 9000 else "∞",
            "Source Term [Ci Cs-137]": f"{d['source_term_Ci_Cs137']:.1e}",
            "CDF [/ry]":           f"{d['CDF_per_ry']:.1e}",
            "Passive Safety":      "✅" if d["passive_safety"] else "❌",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
#  MODULE 10 – SEISMIC & EXTERNAL HAZARD STRESS TEST
# ══════════════════════════════════════════════════════════════════

EXTERNAL_HAZARDS = {
    "Safe Shutdown Earthquake (SSE – 0.3g)": {
        "desc": "Design basis earthquake. All safety systems must remain functional.",
        "cooling_degradation": 0.0,
        "power_trip_prob":     0.999,
        "eccs_availability":   0.98,
        "offsite_power_loss":  True,
        "diesel_start_prob":   0.97,
    },
    "Beyond Design Basis Earthquake (0.6g – Fukushima level)": {
        "desc": "Exceeded Fukushima Daiichi design basis. Common-cause failures likely.",
        "cooling_degradation": 0.5,
        "power_trip_prob":     0.995,
        "eccs_availability":   0.70,
        "offsite_power_loss":  True,
        "diesel_start_prob":   0.60,
    },
    "Tsunami (14m wave – Fukushima actual)": {
        "desc": "Station blackout from seawater damage. AC power fully lost.",
        "cooling_degradation": 0.9,
        "power_trip_prob":     0.999,
        "eccs_availability":   0.10,
        "offsite_power_loss":  True,
        "diesel_start_prob":   0.05,
    },
    "Aircraft Impact": {
        "desc": "Post-9/11 design requirement. Containment must withstand Boeing 767 impact.",
        "cooling_degradation": 0.3,
        "power_trip_prob":     0.90,
        "eccs_availability":   0.80,
        "offsite_power_loss":  True,
        "diesel_start_prob":   0.85,
    },
    "Extreme Cold (−40°C)": {
        "desc": "Freezing of emergency water supplies, diesel fuel gelling.",
        "cooling_degradation": 0.2,
        "power_trip_prob":     0.95,
        "eccs_availability":   0.85,
        "offsite_power_loss":  False,
        "diesel_start_prob":   0.75,
    },
}

def render_hazard_module(history: list):
    st.subheader("🌊 Seismic & External Hazard Stress Test")
    st.caption(
        "Post-Fukushima, all reactors must demonstrate coping capability against "
        "beyond-design-basis external events. This module simulates the safety "
        "system degradation caused by each hazard type."
    )

    hazard_name = st.selectbox("Select External Hazard", list(EXTERNAL_HAZARDS.keys()))
    hazard      = EXTERNAL_HAZARDS[hazard_name]

    st.markdown(f"**{hazard_name}**  \n{hazard['desc']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Cooling System Degradation",
              f"{hazard['cooling_degradation']*100:.0f}%",
              "⚠️ Severe loss of cooling" if hazard["cooling_degradation"] > 0.5 else "Manageable")
    c2.metric("ECCS Availability",
              f"{hazard['eccs_availability']*100:.0f}%",
              "✅ Adequate" if hazard["eccs_availability"] > 0.8 else "⚠️ Degraded")
    c3.metric("Diesel Generator Start",
              f"{hazard['diesel_start_prob']*100:.0f}%",
              "✅ Reliable" if hazard["diesel_start_prob"] > 0.9 else "⚠️ Unreliable")

    # Projected accident sequence without/with passive safety
    t_arr    = np.arange(0, 7200, 60)
    cool_deg = hazard["cooling_degradation"]

    # Active ECCS system response
    Tf_active = np.zeros(len(t_arr))
    Tf_active[0] = 620
    for i in range(1, len(t_arr)):
        eccs_avail = hazard["eccs_availability"] * hazard["diesel_start_prob"]
        cooling    = (1 - cool_deg) * eccs_avail
        decay_pct  = 0.066 * max(((t_arr[i]/3600) + 0.001), 0.001) ** (-0.2) * 100
        dT         = decay_pct * 3.15 - cooling * (Tf_active[i-1] - 305) / 8
        Tf_active[i] = np.clip(Tf_active[i-1] + dT, 305, 5000)

    # Passive system response
    Tf_passive = np.zeros(len(t_arr))
    Tf_passive[0] = 620
    for i in range(1, len(t_arr)):
        cooling    = 0.65  # passive PRHR – not affected by electrical damage
        decay_pct  = 0.066 * max(((t_arr[i]/3600) + 0.001), 0.001) ** (-0.2) * 100
        dT         = decay_pct * 3.15 - cooling * (Tf_passive[i-1] - 305) / 8
        Tf_passive[i] = np.clip(Tf_passive[i-1] + dT, 305, 5000)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_arr, y=Tf_active,
        name="Active ECCS (affected by hazard)",
        line=dict(color="#ef4444", width=2)))
    fig.add_trace(go.Scatter(x=t_arr, y=Tf_passive,
        name="Passive Safety System (hazard-independent)",
        line=dict(color="#00ff9d", width=2)))
    for lim, label in [(1200, "Cladding damage"), (1800, "Melt onset"), (2200, "Meltdown")]:
        fig.add_hline(y=lim, line=dict(color="orange", dash="dash", width=1),
                      annotation_text=label)
    fig.update_layout(
        title=f"Projected Fuel Temperature: Active vs Passive — {hazard_name}",
        xaxis_title="Time [s]",
        yaxis_title="Fuel Temperature [°C]",
        template="plotly_dark",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Key insight
    if cool_deg > 0.7:
        st.error(
            "⚠️ **Active ECCS cannot cope with this hazard.** "
            "This demonstrates why passive safety systems are a Generation III+/IV requirement. "
            "Passive systems are driven by gravity, natural convection, and compressed gas — "
            "none of which are affected by station blackout or seismic damage."
        )
    else:
        st.info(
            "ℹ️ Active ECCS retains some effectiveness for this hazard class. "
            "Passive systems still provide significant safety margin improvement."
        )


# ══════════════════════════════════════════════════════════════════
#  MODULE 11 – MULTI-UNIT RISK
# ══════════════════════════════════════════════════════════════════

def render_multi_unit_risk():
    st.subheader("🏭 Multi-Unit Risk & Shared Infrastructure Model")
    st.caption(
        "Fukushima demonstrated that a single initiating event can simultaneously "
        "challenge multiple units sharing electrical buses, seawater cooling, "
        "fire protection, and emergency water supplies. "
        "NRC now requires multi-unit PRA for all US sites."
    )

    n_units = st.slider("Number of units on site", 1, 6, 4)
    shared_infra = st.multiselect(
        "Shared infrastructure (potential common-cause failure)",
        ["Seawater cooling system", "Electrical switchyard",
         "Emergency diesel generators", "Control room",
         "Fire protection system", "Spent fuel pool cooling", "Make-up water supply"],
        default=["Seawater cooling system", "Electrical switchyard",
                 "Emergency diesel generators"],
    )

    # Common cause failure model
    # If a shared system fails, all units depending on it lose that function
    base_cdf_per_unit = 1e-5   # Gen III+ CDF per unit per year
    ccf_factors = {
        "Seawater cooling system":    10.0,
        "Electrical switchyard":       8.0,
        "Emergency diesel generators": 5.0,
        "Control room":               15.0,
        "Fire protection system":      3.0,
        "Spent fuel pool cooling":     2.0,
        "Make-up water supply":        4.0,
    }

    # Station CDF = 1 - prod(1 - unit_CDF_with_CCF)
    unit_cdfs = []
    for u in range(n_units):
        ccf_multiplier = 1.0
        for infra in shared_infra:
            ccf_multiplier *= ccf_factors.get(infra, 1.0)
        ccf_multiplier = min(ccf_multiplier, 1000)
        unit_cdf = base_cdf_per_unit * ccf_multiplier ** 0.5   # square root scaling
        unit_cdfs.append(unit_cdf)

    station_cdf = 1 - np.prod([1 - c for c in unit_cdfs])
    sfp_risk    = 0.01 * len([i for i in shared_infra if "pool" in i.lower()])

    c1, c2, c3 = st.columns(3)
    c1.metric("Single-Unit CDF",  f"{unit_cdfs[0]:.2e} /ry",
              "✅ OK" if unit_cdfs[0] < 1e-4 else "⚠️ Elevated")
    c2.metric("Station CDF (all units)", f"{station_cdf:.2e} /ry",
              "✅ OK" if station_cdf < 1e-4 else "⚠️ Elevated")
    c3.metric("SFP Risk Contribution",
              f"{sfp_risk:.2e} /ry" if sfp_risk > 0 else "None identified")

    # Dependency matrix
    systems  = ["Reactor Shutdown", "ECCS", "Decay Heat Removal",
                "Containment", "Control Room", "Spent Fuel Cooling"]
    infra_s  = ["Grid Power"] + list(shared_infra)[:5]
    matrix   = np.random.choice([0, 1], size=(len(systems), len(infra_s)),
                                  p=[0.4, 0.6])

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=infra_s,
        y=systems,
        colorscale=[[0, "#1a1a2e"], [1, "#ef4444"]],
        showscale=False,
        text=[["Depends" if v else "Independent" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title="Safety System × Shared Infrastructure Dependency Matrix",
        template="plotly_dark",
        height=320,
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Key Fukushima Lessons:**
- Units 1, 2, and 3 were simultaneously damaged because they shared the same seawater cooling header
- The Unit 4 SFP nearly boiled dry — spent fuel pools are now considered part of the nuclear risk profile
- NRC FLEX program (post-Fukushima) requires portable pumps/generators to break common-cause dependencies
    """)


# ══════════════════════════════════════════════════════════════════
#  MODULE 12 – HUMAN RELIABILITY ANALYSIS & ALARM LOAD
# ══════════════════════════════════════════════════════════════════

def render_hra_module(history: list):
    st.subheader("🧠 Human Reliability Analysis (HRA) & Alarm Load Estimator")
    st.caption(
        "Human error contributes to ~70% of nuclear incidents. HRA quantifies "
        "the probability of operator error as a function of time pressure, "
        "alarm load, procedure quality, and stress. TMI had 100+ alarms in the "
        "first 3 minutes — operators simply could not process them all."
    )

    time_available = int(st.slider("Time available for diagnosis [min]", 1, 60, 15) or 15)
    alarm_rate     = int(st.slider("Alarm rate [alarms/min]",            0, 30,  5) or  5)
    procedure_qual = int(st.slider("Procedure quality [1=poor, 5=excellent]", 1, 5, 3) or  3)
    _ = st.columns(3)   # layout spacer (unused but keeps visual grouping)

    # THERP-based HEP model (simplified)
    # Base HEP for diagnosis task: 0.01
    # Time stress multiplier: higher at shorter times
    time_stress = max(1.0, 5.0 * np.exp(-time_available / 10))

    # Alarm load multiplier: cognitive overload above ~5/min
    alarm_stress = max(1.0, 1.0 + (alarm_rate - 5) * 0.15) if alarm_rate > 5 else 1.0

    # Procedure quality divider
    proc_factor = 1.0 / procedure_qual

    HEP = min(0.99, 0.01 * time_stress * alarm_stress * proc_factor)

    st.markdown(f"""
**Human Error Probability (HEP) for Correct Diagnosis: {HEP:.3f}**
- Time stress factor: **{time_stress:.2f}×** (available time = {time_available} min)
- Alarm overload factor: **{alarm_stress:.2f}×** ({alarm_rate} alarms/min)
- Procedure quality factor: **{proc_factor:.2f}×** (quality score = {procedure_qual}/5)
""")

    # Contextual comparison
    benchmark_data = {
        "Expert with time, clear alarms": 0.001,
        "Well-trained, normal stress":    0.005,
        "TMI conditions (alarm flood)":   0.40,
        "Chernobyl conditions (unusual)": 0.50,
        f"Current scenario":              HEP,
    }
    fig = go.Figure(go.Bar(
        x=list(benchmark_data.keys()),
        y=list(benchmark_data.values()),
        marker_color=["#00ff9d", "#00ff9d", "#ef4444", "#ef4444",
                      "#fbbf24" if HEP < 0.1 else "#ef4444"],
    ))
    fig.add_hline(y=0.1, line=dict(color="orange", dash="dash"),
                  annotation_text="10% HEP threshold (unacceptable)")
    fig.update_layout(
        title="Human Error Probability Benchmark",
        yaxis_title="HEP",
        yaxis_type="log",
        template="plotly_dark",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Alarm load from simulation
    if history:
        df = pd.DataFrame(history)
        if "t" in df.columns:
            # Count state changes as alarm events (simplified)
            sim_alarms_per_min = st.session_state.get("event_log", [])
            total_alarms = len(sim_alarms_per_min)
            sim_dur_min  = float(df["t"].max()) / 60 if len(df) > 0 else 1
            estimated_rate = total_alarms / max(sim_dur_min, 0.01)

            st.markdown("---")
            st.markdown("### 🔔 Simulation Alarm Load")
            a1, a2, a3 = st.columns(3)
            a1.metric("Total Alarms Fired",    total_alarms)
            a2.metric("Simulation Duration",   f"{sim_dur_min:.1f} min")
            a3.metric("Est. Alarm Rate",        f"{estimated_rate:.1f} /min",
                      "✅ Manageable" if estimated_rate < 5 else "⚠️ Cognitive overload risk")

    with st.expander("📖 HRA Methods & Nuclear Industry Context"):
        st.markdown("""
**Key HRA Methods Used in Nuclear Licensing:**
- **THERP** (Technique for Human Error Rate Prediction) — table-based, widely used in US PRAs
- **ATHEANA** (A Technique for Human Event Analysis) — focuses on error-forcing conditions
- **CREAM** (Cognitive Reliability and Error Analysis Method) — cognitive performance shaping

**Historical Events Driven by Human Factors:**
| Event | Key Human Factor |
|-------|-----------------|
| TMI-2 (1979) | Alarm flooding, ECCS throttling, misdiagnosis |
| Chernobyl (1986) | Procedure violation, positive void coefficient not communicated |
| Davis-Besse (2002) | Maintenance oversight, vessel head corrosion missed |
| Forsmark (2006) | Common-cause failure in backup power not caught in testing |

**Design Solutions:**
- Symptom-based EOPs (vs event-based) — handle unanticipated events
- Alarm prioritisation (maximum 5-7 alarms visible at once)
- Large Display Panels for spatial awareness
- Advanced digital I&C with computerised procedure systems
        """)


# ══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT – renders all modules as tabs
# ══════════════════════════════════════════════════════════════════

def render_safety_modules():
    """
    Top-level renderer. Call this from the main app to add the safety
    design tab suite.
    """
    st.header("🔐 Safety Design Engineering Suite")
    st.caption(
        "12 strategic modules for designing inherently safer reactors. "
        "Based on IAEA Safety Standards, NRC regulatory framework, "
        "Gen III+/IV passive safety research, and historical accident analysis."
    )

    history = st.session_state.get("history", [])

    tabs = st.tabs([
        "🏛️ Barriers",
        "🛡️ Passive Safety",
        "🎲 PRA",
        "🗺️ Stability Map",
        "🏠 Containment",
        "⚠️ Accident Prog.",
        "📋 EOPs",
        "📏 Margins",
        "⚡ SMR Advantage",
        "🌊 Hazards",
        "🏭 Multi-Unit",
        "🧠 Human Factors",
    ])

    with tabs[0]:  render_did_tracker(history)
    with tabs[1]:  render_passive_safety_designer(history)
    with tabs[2]:  render_pra_module()
    with tabs[3]:  render_stability_map()
    with tabs[4]:  render_containment_module(history)
    with tabs[5]:  render_sap_tracker(history)
    with tabs[6]:  render_eop_trainer(history)
    with tabs[7]:  render_margin_tracker(history)
    with tabs[8]:  render_smr_advantage()
    with tabs[9]:  render_hazard_module(history)
    with tabs[10]: render_multi_unit_risk()
    with tabs[11]: render_hra_module(history)
