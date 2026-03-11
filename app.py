# npp_simulator_v1.0_enhanced.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import base64
import io
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import copy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

try:
    import imageio
    IMAGEIO_OK = True
except ImportError:
    IMAGEIO_OK = False

try:
    import plotly.io as pio
    PLOTLY_IO_OK = True
except ImportError:
    PLOTLY_IO_OK = False

try:
    from safety_modules import render_safety_modules as _render_safety
    SAFETY_MODULES_AVAILABLE = True
except ImportError:
    SAFETY_MODULES_AVAILABLE = False

# ──────────────────────────────────────────────────────────
#  Physical Constants
# ──────────────────────────────────────────────────────────

KEEPIN_LAMBDA = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
KEEPIN_BETA   = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
BETA_TOTAL    = float(np.sum(KEEPIN_BETA))
PROMPT_LAMBDA_BASE = 1e-4

SIGMA_XE   = 2.65e-22
YIELD_I    = 0.0639
YIELD_XE   = 0.00228
LAMBDA_I   = 2.87e-5
LAMBDA_XE  = 2.09e-5
FLUX_NOM   = 3.0e13

BORON_WORTH_PCM_PPM = 1.0

# Xenon ODE coupling inside RHS: worth in Δk/k per normalised Xe unit
XE_WORTH_PER_NORM = 0.006   # same scale as previous post-processing

# Void time constant [s] for differential dynamics
VOID_TAU = 5.0   # dvoid/dt = (target – void) / VOID_TAU

# ──────────────────────────────────────────────────────────
#  Reactor Configurations
# ──────────────────────────────────────────────────────────

REACTOR_CONFIGS: Dict[str, dict] = {
    "PWR": {
        "alpha_doppler": -3.0,
        "alpha_mod":    -25.0,
        "alpha_void":     0.0,
        "rod_worth":    -18.0,
        "tau_fuel":       4.5,
        "tau_cool":       8.0,
        "Tf0":          620.0,
        "Tc0":          305.0,
        "P_nom":        150.0,
        "boron_worth":  -1.0,
        "has_void":     False,
        "has_graphite": False,
    },
    "BWR": {
        "alpha_doppler": -1.8,
        "alpha_mod":     -8.0,
        "alpha_void":  -140.0,
        "rod_worth":   -18.0,
        "tau_fuel":      3.0,
        "tau_cool":      4.0,
        "Tf0":         550.0,
        "Tc0":         286.0,
        "P_nom":        70.0,
        "boron_worth":  -1.0,
        "has_void":     True,
        "has_graphite": False,
    },
    "RBMK": {
        "alpha_doppler": -1.2,
        "alpha_mod":      5.0,
        "alpha_void":   350.0,
        "rod_worth":   -12.0,
        "tau_fuel":      2.5,
        "tau_cool":      3.5,
        "Tf0":         700.0,
        "Tc0":         270.0,
        "P_nom":        65.0,
        "boron_worth":  -0.8,
        "has_void":     True,
        "has_graphite": True,
    },
    "SMR": {
        "alpha_doppler": -3.5,
        "alpha_mod":    -30.0,
        "alpha_void":     0.0,
        "rod_worth":    -20.0,
        "tau_fuel":       3.5,
        "tau_cool":       6.0,
        "Tf0":          600.0,
        "Tc0":          300.0,
        "P_nom":        155.0,
        "boron_worth":  -1.2,
        "has_void":     False,
        "has_graphite": False,
    },
    "CANDU": {
        "alpha_doppler": -1.5,
        "alpha_mod":    -20.0,
        "alpha_void":  -100.0,
        "rod_worth":   -15.0,
        "tau_fuel":      4.0,
        "tau_cool":      7.0,
        "Tf0":         530.0,
        "Tc0":         290.0,
        "P_nom":       100.0,
        "boron_worth":  -0.9,
        "has_void":     True,
        "has_graphite": False,
    },
}

# ──────────────────────────────────────────────────────────
#  Alarm definitions
# ──────────────────────────────────────────────────────────

ALARMS = [
    ("HIGH POWER",          "power",       1.10, "CRITICAL"),
    ("HIGH-HIGH POWER",     "power",       1.20, "EMERGENCY"),
    ("FUEL TEMP HIGH",      "Tf",        1200.0, "WARNING"),
    ("FUEL TEMP HIGH-HIGH", "Tf",        1800.0, "CRITICAL"),
    ("MELT ONSET",          "Tf",        2200.0, "EMERGENCY"),
    ("HIGH PRESSURE",       "pressure",   165.0, "WARNING"),
    ("HIGH-HIGH PRESSURE",  "pressure",   175.0, "CRITICAL"),
    ("VOID HIGH",           "void",         0.7, "WARNING"),
    ("XENON SPIKE",         "xenon",        5e13, "WARNING"),
    ("H2 GENERATION",       "h2_produced",  0.1, "CRITICAL"),
]

ALARM_COLORS = {"WARNING": "🟡", "CRITICAL": "🔴", "EMERGENCY": "🚨"}

# ──────────────────────────────────────────────────────────
#  Short base64 beep (440 Hz, 0.15 s, WAV)
# ──────────────────────────────────────────────────────────

def _make_beep_b64(freq: float = 440.0, duration: float = 0.15,
                   sample_rate: int = 8000) -> str:
    t   = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wav = (0.4 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    import wave
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wav.tobytes())
    return base64.b64encode(buf.getvalue()).decode()

BEEP_B64 = _make_beep_b64(880.0)   # generated once at import

# ──────────────────────────────────────────────────────────
#  Dataclasses
# ──────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    kp:               float = 4.0
    ki:               float = 0.15
    kd:               float = 8.0
    void_factor:      float = 1.0
    doppler_factor:   float = 1.0
    lambda_factor:    float = 1.0
    decay_factor:     float = 1.0
    rod_worth_factor: float = 1.0

@dataclass
class ReactorState:
    t:               float = 0.0
    power:           float = 0.05
    c:               np.ndarray = field(default_factory=lambda: np.zeros(6))
    Tf:              float = 620.0
    Tc:              float = 305.0
    pressure:        float = 150.0
    void:            float = 0.0
    rod_pos:         float = 95.0
    boron:           float = 800.0
    scram_active:    bool  = False
    scram_t:         float = -1.0
    rho:             float = 0.0
    prev_power:      float = 0.05
    decay_heat:      float = 0.0
    h2_produced:     float = 0.0
    core_damage:     str   = "Intact"
    xenon:           float = 0.0
    iodine:          float = 0.0
    eccs_active:     bool  = False
    emergency_boron: float = 0.0
    rod_moving_in:   bool  = False

    def to_dict(self) -> dict:
        return {
            't':               self.t,
            'power':           self.power,
            'c':               self.c.tolist(),
            'Tf':              self.Tf,
            'Tc':              self.Tc,
            'pressure':        self.pressure,
            'void':            self.void,
            'rod_pos':         self.rod_pos,
            'boron':           self.boron,
            'scram_active':    self.scram_active,
            'scram_t':         self.scram_t,
            'rho':             self.rho,
            'prev_power':      self.prev_power,
            'decay_heat':      self.decay_heat,
            'h2_produced':     self.h2_produced,
            'core_damage':     self.core_damage,
            'xenon':           self.xenon,
            'iodine':          self.iodine,
            'eccs_active':     self.eccs_active,
            'emergency_boron': self.emergency_boron,
        }

    @staticmethod
    def from_dict(d: dict) -> "ReactorState":
        s = ReactorState()
        s.t              = d.get('t', 0.0)
        s.power          = d.get('power', 0.05)
        s.c              = np.array(d.get('c', [0.0]*6))
        s.Tf             = d.get('Tf', 620.0)
        s.Tc             = d.get('Tc', 305.0)
        s.pressure       = d.get('pressure', 150.0)
        s.void           = d.get('void', 0.0)
        s.rod_pos        = d.get('rod_pos', 95.0)
        s.boron          = d.get('boron', 800.0)
        s.scram_active   = d.get('scram_active', False)
        s.scram_t        = d.get('scram_t', -1.0)
        s.rho            = d.get('rho', 0.0)
        s.prev_power     = d.get('prev_power', 0.05)
        s.decay_heat     = d.get('decay_heat', 0.0)
        s.h2_produced    = d.get('h2_produced', 0.0)
        s.core_damage    = d.get('core_damage', "Intact")
        s.xenon          = d.get('xenon', 0.0)
        s.iodine         = d.get('iodine', 0.0)
        s.eccs_active    = d.get('eccs_active', False)
        s.emergency_boron = d.get('emergency_boron', 0.0)
        return s

# ──────────────────────────────────────────────────────────
#  Physics Engine  (hand-written RK4, fully coupled)
# ──────────────────────────────────────────────────────────

class ReactorPhysics:
    def __init__(self, reactor_type: str, sim_config: SimConfig):
        self.rtype = reactor_type.upper()
        self.dt    = 0.05
        self.cfg   = REACTOR_CONFIGS.get(self.rtype, REACTOR_CONFIGS["PWR"])
        sc         = sim_config

        self.alpha_d   = self.cfg["alpha_doppler"] * sc.doppler_factor * 1e-5
        self.alpha_m   = self.cfg["alpha_mod"]                         * 1e-5
        self.alpha_v   = self.cfg["alpha_void"] * sc.void_factor       * 1e-5
        self.rod_worth = self.cfg["rod_worth"] * sc.rod_worth_factor   * 1e-5
        self.Lambda    = PROMPT_LAMBDA_BASE * sc.lambda_factor
        self.decay_f   = sc.decay_factor
        self.tau_fuel  = self.cfg["tau_fuel"]
        self.tau_cool  = self.cfg["tau_cool"]
        self.sim_config = sim_config

        boron_w_raw  = self.cfg["boron_worth"] * 1e-5
        rod_w_raw    = self.cfg["rod_worth"] * sc.rod_worth_factor * 1e-5
        self.excess_rho = (-boron_w_raw * 800.0
                           + XE_WORTH_PER_NORM
                           - rod_w_raw * (100.0 - 39.0))
        self.Tf0      = self.cfg["Tf0"]
        self.Tc0      = self.cfg["Tc0"]
        self.P_nom    = self.cfg["P_nom"]
        self.boron_w  = self.cfg["boron_worth"] * 1e-5
        self.has_void = self.cfg["has_void"]
        self.has_graphite = self.cfg["has_graphite"]

    # ── Reactivity (without xenon – handled inside ODE) ──────
    def _rho_base(self, Tf: float, Tc: float, void: float,
                  rod_pos: float, total_boron: float,
                  ext_rho: float, graphite_add: float) -> float:
        rho  = self.excess_rho + ext_rho
        rho += self.alpha_d * (Tf - self.Tf0)          # Doppler (coupled)
        rho += self.alpha_m * (Tc - self.Tc0)
        if self.has_void:
            rho += self.alpha_v * void
        rho += self.rod_worth * (100.0 - rod_pos)
        rho += self.boron_w * total_boron
        rho += graphite_add
        return rho

    # ── RK4 for kinetics ODE  ─────────────────────────────────
    # State vector y = [P, c0..c5, xe_norm, iodine_norm]  (9 elements)
    # Doppler feedback couples Tf(t) implicitly via Euler for T (fast enough)
    def _kinetics_rhs(self, y: np.ndarray, Tf: float, Tc: float, void: float,
                      rod_pos: float, total_boron: float,
                      ext_rho: float, g_add: float,
                      xe_eq: float) -> np.ndarray:
        P        = max(y[0], 1e-10)
        c        = np.maximum(y[1:7], 0.0)
        xe_norm  = max(y[7], 0.0)
        i_norm   = max(y[8], 0.0)

        # Base reactivity including live Doppler
        rho_b = self._rho_base(Tf, Tc, void, rod_pos, total_boron, ext_rho, g_add)
        # Xenon worth coupled to flux
        xe_abs = xe_norm * xe_eq
        flux   = P * FLUX_NOM
        rho_xe = -(SIGMA_XE * xe_abs * flux) / (FLUX_NOM * (1.0 + 1e-20))
        # Normalised xenon worth
        rho_xe_norm = -XE_WORTH_PER_NORM * xe_norm
        rho = rho_b + rho_xe_norm

        dP = ((rho - BETA_TOTAL) / self.Lambda) * P + np.dot(KEEPIN_LAMBDA, c)
        dc = (KEEPIN_BETA / self.Lambda) * P - KEEPIN_LAMBDA * c

        # Xenon / Iodine ODE (normalised units)
        i_abs  = i_norm * xe_eq
        xe_abs = xe_norm * xe_eq
        dI_abs  = YIELD_I * flux - LAMBDA_I * i_abs
        dXe_abs = (YIELD_XE * flux + LAMBDA_I * i_abs
                   - LAMBDA_XE * xe_abs - SIGMA_XE * flux * xe_abs)
        dI_norm  = dI_abs  / xe_eq
        dXe_norm = dXe_abs / xe_eq

        dy = np.empty(9)
        dy[0]   = dP
        dy[1:7] = dc
        dy[7]   = dXe_norm
        dy[8]   = dI_norm
        return dy

    def _rk4_kinetics(self, y0: np.ndarray, dt: float,
                      Tf: float, Tc: float, void: float,
                      rod_pos: float, total_boron: float,
                      ext_rho: float, g_add: float, xe_eq: float) -> np.ndarray:
        k1 = self._kinetics_rhs(y0,            Tf, Tc, void, rod_pos, total_boron, ext_rho, g_add, xe_eq)
        k2 = self._kinetics_rhs(y0 + dt/2*k1,  Tf, Tc, void, rod_pos, total_boron, ext_rho, g_add, xe_eq)
        k3 = self._kinetics_rhs(y0 + dt/2*k2,  Tf, Tc, void, rod_pos, total_boron, ext_rho, g_add, xe_eq)
        k4 = self._kinetics_rhs(y0 + dt*k3,    Tf, Tc, void, rod_pos, total_boron, ext_rho, g_add, xe_eq)
        return y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # ── Public step ────────────────────────────────────────────
    def step(self, state: ReactorState, rod_speed: float, ext_rho: float,
             cooling_factor: float, graphite_tip: bool = False) -> ReactorState:
        dt = self.dt

        rod_moving_in = rod_speed < 0.0
        g_add = 0.0
        if self.has_graphite and graphite_tip and rod_moving_in and state.rod_pos > 5.0:
            g_add = 1.2e-2

        xe_eq = max((YIELD_XE + YIELD_I) * FLUX_NOM /
                    (LAMBDA_XE + SIGMA_XE * FLUX_NOM), 1e-30)

        total_boron = state.boron + state.emergency_boron

        # ── RK4 kinetics (9-state: P, 6 precursors, Xe, I) ──
        y0  = np.array([state.power,
                        *state.c,
                        state.xenon,
                        state.iodine])
        y1  = self._rk4_kinetics(y0, dt, state.Tf, state.Tc, state.void,
                                  state.rod_pos, total_boron,
                                  ext_rho, g_add, xe_eq)

        P_new    = float(np.clip(y1[0], 0.0, 500.0))
        c_new    = np.maximum(y1[1:7], 0.0)
        xe_norm  = float(np.clip(y1[7], 0.0, 5.0))
        i_norm   = float(np.clip(y1[8], 0.0, 5.0))

        # ── Decay heat ─────────────────────────────────────────
        decay = 0.0
        if state.scram_active and state.scram_t >= 0.0:
            t_since = max(1.0, state.t - state.scram_t)
            t_h     = t_since / 3600.0
            decay   = float(np.clip(0.066 * (t_h ** -0.2) * self.decay_f, 0.0, 0.07))
        total_power = max(P_new, decay)

        # ── Thermal-hydraulics ─────────────────────────────────
        dT_nom = self.Tf0 - self.Tc0
        dTf = total_power * dT_nom / self.tau_fuel - (state.Tf - state.Tc) / self.tau_fuel
        dTc = (state.Tf - state.Tc) / self.tau_fuel * cooling_factor - \
              (state.Tc - 280.0) / self.tau_cool
        if state.eccs_active:
            dTc -= 20.0
        Tf_new = float(np.clip(state.Tf + dTf * dt, 0.0, 5000.0))
        Tc_new = float(np.clip(state.Tc + dTc * dt, 270.0, 380.0))

        # ── Pressure ────────────────────────────────────────────
        P_thermo = self.P_nom + 1.2 * (Tc_new - self.Tc0)
        if state.eccs_active:
            P_thermo -= 20.0
        pressure_new = float(np.clip(P_thermo + np.random.normal(0.0, 0.3), 0.0, 250.0))

        # ── Differential void dynamics ──────────────────────────
        if self.has_void:
            target_void = float(np.clip(0.55 * (total_power - 0.2), 0.0, 0.95))
            dvoid = (target_void - state.void) / VOID_TAU
            void_new = float(np.clip(state.void + dvoid * dt, 0.0, 0.95))
        else:
            void_new = 0.0

        # ── Rod position ────────────────────────────────────────
        rod_new = float(np.clip(state.rod_pos + rod_speed * dt, 0.0, 100.0))

        # ── H2 generation ───────────────────────────────────────
        h2_new = state.h2_produced
        if Tf_new > 1200.0:
            h2_rate = 0.8 * (Tf_new - 1200.0) / 1000.0
            if void_new > 0.05:
                h2_rate *= 2.0
            h2_new += h2_rate * dt / 1000.0

        damage = self.classify_damage(total_power, Tf_new)

        # ── ECCS boration ───────────────────────────────────────
        emg_b = state.emergency_boron
        if state.eccs_active and emg_b < 3000.0:
            emg_b = min(3000.0, emg_b + 50.0 * dt)

        # ── Compute reported reactivity (for display) ───────────
        rho_disp = self._rho_base(Tf_new, Tc_new, void_new,
                                  rod_new, total_boron, ext_rho, g_add)
        rho_disp -= XE_WORTH_PER_NORM * xe_norm

        return ReactorState(
            t               = state.t + dt,
            power           = total_power,
            c               = c_new,
            Tf              = Tf_new,
            Tc              = Tc_new,
            pressure        = pressure_new,
            void            = void_new,
            rod_pos         = rod_new,
            boron           = state.boron,
            scram_active    = state.scram_active,
            scram_t         = state.scram_t,
            rho             = rho_disp,
            prev_power      = state.power,
            decay_heat      = decay,
            h2_produced     = h2_new,
            core_damage     = damage,
            xenon           = xe_norm,
            iodine          = i_norm,
            eccs_active     = state.eccs_active,
            emergency_boron = emg_b,
            rod_moving_in   = rod_moving_in,
        )

    @staticmethod
    def classify_damage(power: float, Tf: float) -> str:
        if power > 100.0:  return "Steam Explosion"
        if Tf   > 3000.0:  return "Graphite Fire"
        if Tf   > 2200.0:  return "Partial Meltdown"
        if Tf   > 1800.0:  return "Melt Onset"
        if Tf   > 1200.0:  return "Cladding Failure"
        return "Intact"

# ──────────────────────────────────────────────────────────
#  Scenario Profiles  (picklable callable classes)
# ──────────────────────────────────────────────────────────

class CoolingProfile:
    def __init__(self, breakpoints: List[Tuple[float, float]]):
        self.bp = sorted(breakpoints, key=lambda x: x[0])
    def __call__(self, t: float) -> float:
        val = self.bp[0][1]
        for ts, fac in self.bp:
            if t >= ts:
                val = fac
        return val

class RhoProfile:
    def __init__(self, breakpoints: List[Tuple[float, float]]):
        self.bp = sorted(breakpoints, key=lambda x: x[0])
    def __call__(self, t: float) -> float:
        val = 0.0
        for ts, rho in self.bp:
            if t >= ts:
                val = rho
        return val

class AZ5Profile:
    def __init__(self, t_start: float, t_end: float):
        self.t_start = t_start
        self.t_end   = t_end
    def __call__(self, t: float) -> bool:
        return self.t_start <= t <= self.t_end

class AlwaysFalse:
    def __call__(self, t: float) -> bool:  return False

class AlwaysZero:
    def __call__(self, t: float) -> float: return 0.0

class AlwaysOne:
    def __call__(self, t: float) -> float: return 1.0

# ──────────────────────────────────────────────────────────
#  Scenario Registry
# ──────────────────────────────────────────────────────────

SCENARIOS: Dict[str, dict] = {
    "Free Play": {
        "desc": "No predefined events — control ext_rho, cooling & rod speed via sliders",
        "events": [],
        "cooling_factor": AlwaysOne(),
        "ext_rho":        AlwaysZero(),
        "az5_active":     AlwaysFalse(),
        "scram_t":        None,
        "free_play":      True,
    },
    "Normal Operation": {
        "desc": "Steady-state 100% power operation with PID control",
        "events": [],
        "cooling_factor": AlwaysOne(),
        "ext_rho":        AlwaysZero(),
        "az5_active":     AlwaysFalse(),
        "scram_t":        None,
    },
    "Controlled Power Ramp": {
        "desc": "Ramp from 50% to 100% power over 10 minutes",
        "events": [(0, "Beginning power ascension", 0.0),
                   (300, "50% power plateau", 0.0),
                   (600, "Resuming ascension to 100%", 0.0)],
        "cooling_factor": AlwaysOne(),
        "ext_rho":        RhoProfile([(0, 0.0)]),
        "az5_active":     AlwaysFalse(),
        "scram_t":        None,
    },
    "Fukushima Station Blackout": {
        "desc": "2011 Daiichi Units 1-3 – complete station blackout sequence (scaled)",
        "events": [
            (0,     "Magnitude 9.0 earthquake – automatic SCRAM triggered",      0.0),
            (2400,  "Tsunami strikes – station blackout, seawater cooling lost",  0.0),
            (7200,  "RCIC operating under battery power, struggling",              0.0),
            (10800, "RCIC fails on Unit 1 – core uncovery begins",                0.0),
            (14400, "Core damage progression accelerates, pressure relief opens", 0.0),
            (18000, "Significant hydrogen generation; building venting begins",   0.0),
        ],
        "cooling_factor": CoolingProfile([(0, 1.0), (2400, 0.40), (7200, 0.15),
                                          (10800, 0.05), (14400, 0.02)]),
        "ext_rho":        AlwaysZero(),
        "az5_active":     AlwaysFalse(),
        "scram_t":        0.0,
    },
    "Chernobyl Test (RBMK)": {
        "desc": "1986 Chernobyl Unit 4 – voltage regulation test, graphite tip effect",
        "events": [
            (0,    "25 Apr 01:00 – Full power operation, test preparation", 0.0),
            (300,  "ECCS isolated for test",                                 0.0),
            (600,  "Power reduction begins",                                 0.0),
            (900,  "Xenon poisoning – power falls to ~30 MWt",              0.0),
            (1200, "Power stabilized ~200 MWt (7% nominal)",                0.0),
            (1800, "Test begins – turbine rundown, reduced cooling",         0.0),
            (2000, "AZ-5 (SCRAM) button pressed – graphite tip surge",      0.0),
            (2015, "First explosion",                                        0.0),
        ],
        "cooling_factor": CoolingProfile([(0, 1.0), (1800, 0.55)]),
        "ext_rho":        RhoProfile([(0, 0.0), (900, -0.02), (1200, -0.005)]),
        "az5_active":     AZ5Profile(2000, 2015),
        "scram_t":        None,
    },
    "Three Mile Island (PWR LOCA)": {
        "desc": "1979 TMI-2 – stuck-open PORV, operator error disables ECCS",
        "events": [
            (0,   "Normal operation at 97%",                               0.0),
            (120, "Secondary feedwater pumps trip",                         0.0),
            (200, "PORV opens – small LOCA begins",                        0.0),
            (260, "PORV fails to reseat – stuck open",                     0.0),
            (300, "Operators misread instruments, disable ECCS",           0.0),
            (600, "Core uncovery begins – fuel damage imminent",           0.0),
        ],
        "cooling_factor": CoolingProfile([(0, 0.9), (120, 0.6), (200, 0.25),
                                          (300, 0.05), (600, 0.02)]),
        "ext_rho":        AlwaysZero(),
        "az5_active":     AlwaysFalse(),
        "scram_t":        200.0,
    },
    "Loss of Coolant Accident (Generic)": {
        "desc": "Rapid depressurisation with partial ECCS activation",
        "events": [
            (0,   "Normal operation",               0.0),
            (30,  "Large-break LOCA initiates",      0.0),
            (35,  "SCRAM on low pressure signal",    0.0),
            (60,  "ECCS high-pressure injection",    0.0),
            (120, "ECCS accumulators discharge",     0.0),
        ],
        "cooling_factor": CoolingProfile([(0, 1.0), (30, 0.1), (60, 0.5), (120, 0.8)]),
        "ext_rho":        AlwaysZero(),
        "az5_active":     AlwaysFalse(),
        "scram_t":        35.0,
    },
    "Anticipated Transient Without Scram": {
        "desc": "ATWS – loss of feedwater with SCRAM failure",
        "events": [
            (0,   "Normal operation", 0.0),
            (60,  "Loss of feedwater – SCRAM demanded but fails", 0.0),
            (120, "Operator initiates manual boration", 0.0),
        ],
        "cooling_factor": CoolingProfile([(0, 1.0), (60, 0.3)]),
        "ext_rho":        AlwaysZero(),
        "az5_active":     AlwaysFalse(),
        "scram_t":        None,
    },
}

# ──────────────────────────────────────────────────────────
#  Event Scheduler
# ──────────────────────────────────────────────────────────

def apply_scenario_events(state: ReactorState, scenario: dict,
                           event_log: list, prev_t: float) -> ReactorState:
    for (t_ev, msg, _) in scenario.get("events", []):
        if prev_t < t_ev <= state.t:
            event_log.append(f"t={t_ev:.0f}s │ {msg}")
    return state

# ──────────────────────────────────────────────────────────
#  Alarm checker
# ──────────────────────────────────────────────────────────

def check_alarms(state: ReactorState, active_alarms: set) -> List[str]:
    new_alarms = []
    vals = {
        'power':       state.power,
        'Tf':          state.Tf,
        'pressure':    state.pressure,
        'void':        state.void,
        'xenon':       state.xenon * 5e13,
        'h2_produced': state.h2_produced,
    }
    for name, param, thresh, level in ALARMS:
        key = (name, level)
        current_val = vals.get(param, 0.0)
        if current_val >= thresh:
            if key not in active_alarms:
                active_alarms.add(key)
                new_alarms.append(
                    f"{ALARM_COLORS.get(level, '⚪')} [{level}] {name}  "
                    f"(t={state.t:.1f}s, val={current_val:.3g})")
        else:
            active_alarms.discard(key)
    return new_alarms

# ──────────────────────────────────────────────────────────
#  Parallel workers (top-level → picklable)
# ──────────────────────────────────────────────────────────

def _mc_worker(args):
    reactor_type, scenario_name, duration, seed = args
    np.random.seed(seed)
    scenario = SCENARIOS[scenario_name]
    sc = SimConfig(
        void_factor    = float(np.random.uniform(0.85, 1.15)),
        doppler_factor = float(np.random.uniform(0.80, 1.20)),
        lambda_factor  = float(np.random.uniform(0.70, 1.30)),
        decay_factor   = float(np.random.uniform(0.90, 1.10)),
    )
    physics = ReactorPhysics(reactor_type, sc)
    state   = make_initial_state(physics, scenario)
    traj    = []
    t = 0.0
    dt = physics.dt
    while t < duration:
        ext_rho = scenario["ext_rho"](t)
        cooling = scenario["cooling_factor"](t)
        az5     = scenario["az5_active"](t)
        state   = physics.step(state, 0.0, ext_rho, cooling, az5)
        traj.append(state.power * 100.0)
        t += dt
    return traj


def _sensitivity_worker(args):
    reactor_type, scenario_name, setpoint, param_name, param_val, duration = args
    scenario = SCENARIOS[scenario_name]
    ref_cfg  = SimConfig()
    cfg      = SimConfig(**{**vars(ref_cfg), param_name: param_val})
    physics  = ReactorPhysics(reactor_type, cfg)
    state    = make_initial_state(physics, scenario)
    t = 0.0
    dt = physics.dt
    while t < duration:
        ext_rho = scenario["ext_rho"](t)
        cooling = scenario["cooling_factor"](t)
        az5     = scenario["az5_active"](t)
        state   = physics.step(state, 0.0, ext_rho, cooling, az5)
        t += dt
    return state.power, state.Tf

# ──────────────────────────────────────────────────────────
#  Initial state factory
# ──────────────────────────────────────────────────────────

def make_initial_state(physics: "ReactorPhysics", scenario: dict) -> "ReactorState":
    state = ReactorState(
        power    = 0.05,
        Tf       = physics.Tf0,
        Tc       = physics.Tc0,
        pressure = physics.P_nom,
        rod_pos  = 39.0,
        boron    = 800.0,
    )
    sc_scram_t = scenario.get("scram_t")
    if sc_scram_t is not None:
        state.scram_active = True
        state.scram_t      = sc_scram_t
    return state

# ──────────────────────────────────────────────────────────
#  Monte Carlo  (@st.cache_data for re-use)
# ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_monte_carlo(reactor_type: str, scenario_name: str,
                    num_runs: int = 50, duration: float = 300.0) -> dict:
    n_workers = min(multiprocessing.cpu_count(), num_runs, 8)
    args = [(reactor_type, scenario_name, duration, i) for i in range(num_runs)]
    results = []
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for traj in pool.map(_mc_worker, args,
                                  chunksize=max(1, num_runs // n_workers)):
                results.append(traj)
    except Exception:
        results = [_mc_worker(a) for a in args]
    min_len = min(len(r) for r in results)
    arr     = np.array([r[:min_len] for r in results])
    return {
        "time": np.arange(min_len) * 0.05,
        "mean": arr.mean(axis=0),
        "std":  arr.std(axis=0),
        "p5":   np.percentile(arr,  5, axis=0),
        "p95":  np.percentile(arr, 95, axis=0),
    }

# ──────────────────────────────────────────────────────────
#  Sensitivity Analysis  (@st.cache_data)
# ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_sensitivity(reactor_type: str, scenario_name: str,
                        setpoint: float) -> pd.DataFrame:
    ref_cfg  = SimConfig()
    duration = 300.0
    ref_p, ref_Tf = _sensitivity_worker(
        (reactor_type, scenario_name, setpoint, "void_factor", ref_cfg.void_factor, duration))
    params = [
        ("Void factor",      "void_factor",      0.10),
        ("Doppler factor",   "doppler_factor",   0.10),
        ("Λ factor",         "lambda_factor",    0.15),
        ("Decay factor",     "decay_factor",     0.10),
        ("Rod worth factor", "rod_worth_factor", 0.10),
    ]
    jobs = []
    for name, key, delta in params:
        base_val = getattr(ref_cfg, key)
        jobs.append((reactor_type, scenario_name, setpoint, key, base_val*(1+delta), duration))
        jobs.append((reactor_type, scenario_name, setpoint, key, base_val*(1-delta), duration))
    n_workers = min(multiprocessing.cpu_count(), len(jobs), 8)
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results_raw = list(pool.map(_sensitivity_worker, jobs))
    except Exception:
        results_raw = [_sensitivity_worker(j) for j in jobs]
    rows = []
    for i, (name, key, delta) in enumerate(params):
        ps_power, ps_Tf = results_raw[i * 2]
        ms_power, ms_Tf = results_raw[i * 2 + 1]
        s_power = ((ps_power*100 - ms_power*100) / max(ref_p*100, 1e-6)) / (2 * delta)
        s_Tf    = ((ps_Tf - ms_Tf) / max(ref_Tf, 1e-6)) / (2 * delta)
        rows.append({"Parameter": name, "S_power": s_power, "S_Tf": s_Tf})
    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────
#  AI Operator Assistant
# ──────────────────────────────────────────────────────────

def ai_operator_advice(state: ReactorState) -> str:
    lines = []
    rho  = state.rho
    Tf   = state.Tf
    xe   = state.xenon
    pwr  = state.power

    if state.scram_active:
        lines.append("🛑 SCRAM active — monitor decay heat; ensure ECCS is standing by.")
    if pwr > 1.15:
        lines.append("⚠️ Power >115%: insert control rods immediately or initiate SCRAM.")
    elif pwr > 1.05:
        lines.append("⚡ Power above setpoint: reduce rod position or add boron.")
    elif pwr < 0.50 and not state.scram_active:
        lines.append("📉 Power low: check xenon load — may need rod withdrawal.")

    if Tf > 1800.0:
        lines.append("🔥 Fuel temp critical! Activate ECCS and initiate SCRAM if not done.")
    elif Tf > 1200.0:
        lines.append("🌡️ Cladding failure risk — increase cooling or reduce power.")

    if rho > 0.002:
        lines.append("☢️ Positive reactivity: reactor is super-prompt — act immediately.")
    elif rho > 0.0:
        lines.append("📈 Reactivity slightly positive — monitor power trend carefully.")

    if xe > 1.5:
        lines.append("☣️ High Xenon load — avoid rapid power changes; xenon may trap you.")
    if state.void > 0.5:
        lines.append("💨 High void fraction — watch for positive void feedback (BWR/RBMK).")
    if state.pressure > 165.0:
        lines.append("🔩 High primary pressure — check PORV / relief valves.")
    if state.h2_produced > 0.05:
        lines.append("💥 H₂ accumulation risk — ensure containment hydrogen recombiners active.")
    if not lines:
        lines.append("✅ All parameters nominal. Continue monitoring.")
    return "\n\n".join(lines)

# ──────────────────────────────────────────────────────────
#  Live background MC uncertainty band  (5 quick runs)
# ──────────────────────────────────────────────────────────

def _run_mc_band_bg(reactor_type: str, scenario_name: str, duration: float,
                    result_holder: dict) -> None:
    """Run 5 MC runs in a daemon thread; store result in result_holder."""
    try:
        res = run_monte_carlo(reactor_type, scenario_name, num_runs=5, duration=duration)
        result_holder["data"] = res
    except Exception as e:
        result_holder["error"] = str(e)

# ──────────────────────────────────────────────────────────
#  GIF / MP4 Export
# ──────────────────────────────────────────────────────────

def export_animation(history: list, fmt: str = "gif") -> Optional[bytes]:
    if not IMAGEIO_OK or not PLOTLY_IO_OK:
        return None
    frames = []
    step   = max(1, len(history) // 60)   # at most 60 frames
    df_full = pd.DataFrame(history)
    for i in range(step, len(history)+1, step):
        sub = df_full.iloc[:i]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["t"], y=sub["power"]*100,
                                 line=dict(color="#00ff9d", width=2), name="Power"))
        if "Tf" in sub.columns:
            fig.add_trace(go.Scatter(x=sub["t"], y=sub["Tf"],
                                     line=dict(color="#ff9500"), name="Fuel T",
                                     yaxis="y2"))
        fig.update_layout(
            template="plotly_dark", height=350, width=700,
            yaxis=dict(title="Power [%]"),
            yaxis2=dict(title="Fuel T [°C]", overlaying="y", side="right"),
            margin=dict(t=30, b=30, l=40, r=40),
        )
        img = pio.to_image(fig, format="png", width=700, height=350)
        frames.append(imageio.v2.imread(io.BytesIO(img)))
    buf = io.BytesIO()
    if fmt == "gif":
        imageio.mimsave(buf, frames, format="GIF", fps=10)
    else:
        imageio.mimsave(buf, frames, format="MP4", fps=10)
    return buf.getvalue()

# ──────────────────────────────────────────────────────────
#  Cached Plotly figure builder
# ──────────────────────────────────────────────────────────

def _build_live_figure(df: pd.DataFrame, setpoint: float,
                       reactor_type: str, scenario_name: str, t: float,
                       mc_band: Optional[dict] = None) -> go.Figure:
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Power & Setpoint [% nominal]",
                        "Temperatures [°C]",
                        "Control Rods & Void [%]",
                        "Reactivity [Δk/k]",
                        "Xenon / Iodine [normalised]"))

    # Row 1 – Power
    if mc_band is not None:
        t_mc = mc_band["time"]
        mask = t_mc <= t
        if mask.any():
            fig.add_trace(go.Scatter(
                x=np.concatenate([t_mc[mask], t_mc[mask][::-1]]),
                y=np.concatenate([mc_band["mean"][mask]+mc_band["std"][mask],
                                  (mc_band["mean"][mask]-mc_band["std"][mask])[::-1]]),
                fill="toself", fillcolor="rgba(255,200,50,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name="±1σ MC band",
                showlegend=True), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["t"], y=df["power"]*100,
        name="Power", line=dict(color="#00ff9d", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["t"], y=[setpoint*100]*len(df),
        name="Setpoint", line=dict(dash="dash", color="#ffffff", width=1)),
        row=1, col=1)
    if "decay_heat" in df.columns:
        fig.add_trace(go.Scatter(x=df["t"], y=df["decay_heat"]*100,
            name="Decay Heat", line=dict(color="#ffdd57", dash="dot")),
            row=1, col=1)

    # Row 2 – Temperatures
    if "Tf" in df.columns:
        fig.add_trace(go.Scatter(x=df["t"], y=df["Tf"],
            name="Fuel T", line=dict(color="#ff9500", width=2)), row=2, col=1)
    if "Tc" in df.columns:
        fig.add_trace(go.Scatter(x=df["t"], y=df["Tc"],
            name="Coolant T", line=dict(color="#00b4ff", width=2)), row=2, col=1)

    # Row 3 – Rods & Void
    if "rod_pos" in df.columns:
        fig.add_trace(go.Scatter(x=df["t"], y=df["rod_pos"],
            name="Rod Pos [%]", line=dict(color="#ff69b4")), row=3, col=1)
    if "void" in df.columns:
        fig.add_trace(go.Scatter(x=df["t"], y=df["void"]*100,
            name="Void %", line=dict(color="#32cd32")), row=3, col=1)

    # Row 4 – Reactivity
    if "rho" in df.columns:
        rho_last  = float(df["rho"].iloc[-1])
        rho_color = "#ff4d4d" if rho_last > 0 else "#00ff9d"
        fig.add_trace(go.Scatter(x=df["t"], y=df["rho"],
            name="ρ (Δk/k)", line=dict(color=rho_color)), row=4, col=1)
        fig.add_hline(y=0, line=dict(color="white", dash="dash"), row=4, col=1)

    # Row 5 – Xenon/Iodine
    if "xenon" in df.columns:
        fig.add_trace(go.Scatter(x=df["t"], y=df["xenon"],
            name="Xe-135", line=dict(color="#c084fc")), row=5, col=1)
    if "iodine" in df.columns:
        fig.add_trace(go.Scatter(x=df["t"], y=df["iodine"],
            name="I-135", line=dict(color="#fb923c", dash="dot")), row=5, col=1)

    fig.update_layout(
        height=1050, template="plotly_dark",
        title=dict(text=f"{reactor_type} – {scenario_name}  │  t={t:.0f}s",
                   font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=10)),
        margin=dict(t=80, b=20),
    )
    return fig

# ──────────────────────────────────────────────────────────
#  Keyboard shortcut JS injection
# ──────────────────────────────────────────────────────────

_KB_JS = """
<script>
(function() {
  if (window._npp_kb_bound) return;
  window._npp_kb_bound = true;
  document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.code === 'Space') {
      e.preventDefault();
      const btns = Array.from(document.querySelectorAll('button'));
      const pb = btns.find(b => b.textContent.includes('Pause') || b.textContent.includes('Resume'));
      if (pb) pb.click();
    }
    if (e.code === 'KeyR') {
      const btns = Array.from(document.querySelectorAll('button'));
      const sb = btns.find(b => b.textContent.includes('Manual SCRAM'));
      if (sb) sb.click();
    }
  });
})();
</script>
"""

# ──────────────────────────────────────────────────────────
#  MAX_HISTORY
# ──────────────────────────────────────────────────────────

MAX_HISTORY = 12_000

# ──────────────────────────────────────────────────────────
#  Live Simulation
# ──────────────────────────────────────────────────────────

def run_live_simulation(reactor_type: str, scenario_name: str,
                        setpoint: float, max_t: float, speed_mult: int):
    _live_sim_fragment(reactor_type, scenario_name, setpoint, max_t, speed_mult)


@st.fragment(run_every=None)
def _live_sim_fragment(reactor_type: str, scenario_name: str,
                       setpoint: float, max_t: float, speed_mult: int):

    # Adaptive chunk size: more steps when stable, fewer when transient
    power_now = (ReactorState.from_dict(st.session_state.history[-1]).power
                 if st.session_state.history else 0.05)
    deviation = abs(power_now - setpoint)
    STEPS_PER_CHUNK = max(10, int(10 * (1 + deviation * 20)))
    if st.session_state.get("turbo_mode"):
        STEPS_PER_CHUNK = max(200, speed_mult * 20)

    scenario   = SCENARIOS[scenario_name]
    is_free    = scenario.get("free_play", False)
    sim_config = st.session_state.sim_config
    physics    = ReactorPhysics(reactor_type, sim_config)

    if st.session_state.history:
        state = ReactorState.from_dict(st.session_state.history[-1])
    else:
        state = make_initial_state(physics, scenario)
        st.session_state.history.append(state.to_dict())

    t = state.t

    if "pid" not in st.session_state:
        st.session_state.pid = {"integral": 0.0, "prev_e": 0.0}
    pid   = st.session_state.pid
    I_MAX = 50.0

    # ── UI placeholders ──────────────────────────────────
    status_container   = st.empty()
    progress_container = st.empty()
    chart_container    = st.empty()
    metrics_container  = st.empty()
    alarms_container   = st.empty()
    log_container      = st.empty()

    if "active_alarms" not in st.session_state:
        st.session_state.active_alarms = set()

    # ── Inject keyboard shortcuts ────────────────────────
    st.html(_KB_JS)

    # ── Real-time sync: adjust sleep so 1 sim-s ≈ 1 wall-s ─
    rt_sync = st.session_state.get("realtime_sync", False)

    # ── PAUSED ─────────────────────────────────────────────
    if st.session_state.paused:
        status_container.warning(
            f"⏸ Paused at t={t:.0f}s  |  Click **▶ Resume** to continue.")
        progress_container.progress(min(1.0, t / max_t))
        _render_current_state(chart_container, metrics_container, alarms_container,
                               log_container, state, setpoint,
                               st.session_state.history, reactor_type, scenario_name, t)
        return

    # ── COMPLETE ────────────────────────────────────────────
    if t >= max_t:
        status_container.success(f"✅ Complete — t={t:.0f}s  |  {state.core_damage}")
        progress_container.progress(1.0)
        _render_current_state(chart_container, metrics_container, alarms_container,
                               log_container, state, setpoint,
                               st.session_state.history, reactor_type, scenario_name, t)
        st.session_state.sim_active  = False
        st.session_state.replay_data = st.session_state.history.copy()
        st.session_state.turbo_mode  = False
        df = pd.DataFrame(st.session_state.history)
        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=False),
            f"npp_{reactor_type}_{scenario_name.replace(' ','_')}.csv",
        )
        st.balloons()
        return

    if "sim_start_wall" not in st.session_state:
        st.session_state.sim_start_wall = time.time()
    start_wall = st.session_state.sim_start_wall

    # ── Run physics chunk ───────────────────────────────────
    wall_chunk_start = time.time()
    steps_done = 0
    while steps_done < STEPS_PER_CHUNK and t < max_t:
        dt     = physics.dt
        prev_t = t

        # Free Play overrides from sliders
        if is_free:
            ext_rho = st.session_state.get("fp_ext_rho", 0.0)
            cooling = st.session_state.get("fp_cooling", 1.0)
        else:
            ext_rho = scenario["ext_rho"](t)
            cooling = scenario["cooling_factor"](t)
        az5 = scenario["az5_active"](t)

        # Operator actions
        if st.session_state.get("op_inject_boron"):
            state.boron = min(4000.0, state.boron + 200.0)
            st.session_state.event_log.append(
                f"t={t:.1f}s │ 🧪 OPERATOR: Boron +200 ppm → {state.boron:.0f} ppm")
            st.session_state.op_inject_boron = False

        if st.session_state.get("op_eccs"):
            state.eccs_active = True
            st.session_state.event_log.append(f"t={t:.1f}s │ 💧 OPERATOR: ECCS activated")
            st.session_state.op_eccs = False

        if st.session_state.get("op_manual_scram"):
            if not state.scram_active:
                state.scram_active = True
                state.scram_t      = t
                st.session_state.event_log.append(f"t={t:.1f}s │ 🛑 OPERATOR: Manual SCRAM")
            st.session_state.op_manual_scram = False

        # Auto-SCRAM
        if (state.power > 1.20 or state.Tf > 750.0) and not state.scram_active:
            state.scram_active = True
            state.scram_t      = t
            st.session_state.event_log.append(
                f"t={t:.1f}s │ ⚡ AUTO-SCRAM: P={state.power:.1%}  Tf={state.Tf:.0f}°C")

        # PID / rod control
        if state.scram_active:
            rod_speed = -40.0
        elif is_free:
            rod_speed = st.session_state.get("fp_rod_speed", 0.0)
        else:
            error           = setpoint - state.power
            pid["integral"] = float(np.clip(pid["integral"] + error * dt, -I_MAX, I_MAX))
            d_term          = (error - pid["prev_e"]) / dt
            rod_speed       = (sim_config.kp * error
                               + sim_config.ki * pid["integral"]
                               + sim_config.kd * d_term) * 12.0
            pid["prev_e"]   = error

        state = physics.step(state, rod_speed, ext_rho, cooling, az5)
        state = apply_scenario_events(state, scenario,
                                      st.session_state.event_log, prev_t)
        t     = state.t

        new_alarms = check_alarms(state, st.session_state.active_alarms)
        for a in new_alarms:
            st.session_state.event_log.append(a)
            # Play beep for new alarms
            st.session_state["_play_beep"] = True

        if len(st.session_state.history) >= MAX_HISTORY:
            st.session_state.history = st.session_state.history[::2]
        st.session_state.history.append(state.to_dict())

        if ("Meltdown" in state.core_damage or "Explosion" in state.core_damage):
            if not st.session_state.get("damage_alerted"):
                st.session_state.event_log.append(
                    f"t={t:.1f}s │ 🚨 SEVERE DAMAGE: {state.core_damage} — paused")
                st.session_state.damage_alerted = True
                st.session_state.paused = True
                break

        steps_done += 1

    st.session_state.pid = pid

    # ── Progress + status ───────────────────────────────────
    pct     = min(1.0, t / max_t)
    elapsed = time.time() - start_wall
    eta     = max(0.0, (max_t - t) / max(speed_mult, 1))
    progress_container.progress(pct)
    status_container.text(
        f"▶ {t:.0f} / {max_t:.0f}s  ({pct*100:.0f}%)  │  "
        f"Elapsed: {elapsed:.0f}s  │  ETA: {eta:.0f}s  │  {state.core_damage}")

    # ── Alarm beep ──────────────────────────────────────────
    if st.session_state.pop("_play_beep", False):
        st.audio(base64.b64decode(BEEP_B64), format="audio/wav", autoplay=True)

    # ── Render charts (skip in turbo mode) ─────────────────
    if not st.session_state.get("turbo_mode"):
        _render_current_state(chart_container, metrics_container, alarms_container,
                               log_container, state, setpoint,
                               st.session_state.history, reactor_type, scenario_name, t)

    # ── Real-time sync sleep ────────────────────────────────
    sim_seconds_this_chunk = steps_done * physics.dt
    wall_chunk = time.time() - wall_chunk_start
    if rt_sync:
        sleep_needed = sim_seconds_this_chunk - wall_chunk
        if sleep_needed > 0:
            time.sleep(min(sleep_needed, 2.0))
    else:
        time.sleep(max(0.01, 0.04 / max(speed_mult, 1)))

    if st.session_state.sim_active and not st.session_state.paused and t < max_t:
        st.rerun()


def _render_current_state(chart_container, metrics_container, alarms_container,
                           log_container, state: ReactorState, setpoint: float,
                           history: list, reactor_type: str, scenario_name: str,
                           t: float):
    df = pd.DataFrame(history)

    if len(df) > 5 and "t" in df.columns and "power" in df.columns:
        # Use cached figure if available; otherwise build fresh
        mc_band = st.session_state.get("mc_band_data")

        # Build or update the cached figure
        if "live_fig" not in st.session_state:
            st.session_state.live_fig = _build_live_figure(
                df, setpoint, reactor_type, scenario_name, t, mc_band)
        else:
            fig = st.session_state.live_fig
            # Incremental update instead of full rebuild
            try:
                new_t = df["t"].tolist()
                # Update all traces by name mapping
                trace_map = {tr.name: i for i, tr in enumerate(fig.data)}
                updates = {
                    "Power":     (new_t, (df["power"]*100).tolist()),
                    "Setpoint":  (new_t, [setpoint*100]*len(df)),
                    "Decay Heat":(new_t, (df["decay_heat"]*100).tolist() if "decay_heat" in df.columns else []),
                    "Fuel T":    (new_t, df["Tf"].tolist() if "Tf" in df.columns else []),
                    "Coolant T": (new_t, df["Tc"].tolist() if "Tc" in df.columns else []),
                    "Rod Pos [%]":(new_t, df["rod_pos"].tolist() if "rod_pos" in df.columns else []),
                    "Void %":   (new_t, (df["void"]*100).tolist() if "void" in df.columns else []),
                    "ρ (Δk/k)": (new_t, df["rho"].tolist() if "rho" in df.columns else []),
                    "Xe-135":   (new_t, df["xenon"].tolist() if "xenon" in df.columns else []),
                    "I-135":    (new_t, df["iodine"].tolist() if "iodine" in df.columns else []),
                }
                for name, (xs, ys) in updates.items():
                    if name in trace_map and ys:
                        fig.data[trace_map[name]].x = xs
                        fig.data[trace_map[name]].y = ys
                fig.update_layout(
                    title_text=f"{reactor_type} – {scenario_name}  │  t={t:.0f}s")
            except Exception:
                # Rebuild on any mismatch
                st.session_state.live_fig = _build_live_figure(
                    df, setpoint, reactor_type, scenario_name, t, mc_band)

        chart_container.plotly_chart(st.session_state.live_fig, use_container_width=True)

    # ── Metrics ─────────────────────────────────────────────
    with metrics_container.container():
        cols = st.columns(6)
        cols[0].metric("Power",      f"{state.power:.1%}",
                       f"{(state.power - setpoint)*100:+.1f} pp")
        cols[1].metric("Fuel T",     f"{state.Tf:.0f} °C")
        cols[2].metric("Coolant T",  f"{state.Tc:.0f} °C")
        cols[3].metric("Pressure",   f"{state.pressure:.1f} bar")
        cols[4].metric("Reactivity", f"{state.rho:+.4f} Δk/k")
        cols[5].metric("Status",     state.core_damage)
        cols2 = st.columns(4)
        cols2[0].metric("Xe-135",    f"{state.xenon:.2f}")
        cols2[1].metric("Boron",     f"{state.boron:.0f} ppm")
        cols2[2].metric("H₂ gen",    f"{state.h2_produced:.3f}")
        cols2[3].metric("ECCS",      "ACTIVE" if state.eccs_active else "Standby")

    # ── Active alarms ────────────────────────────────────────
    active = list(st.session_state.get("active_alarms", set()))
    if active:
        alarm_strs = [f"{ALARM_COLORS.get(lv,'⚪')} **{nm}** [{lv}]"
                      for nm, lv in active]
        alarms_container.error("  |  \n".join(alarm_strs))
    else:
        alarms_container.empty()

    # ── Event log ────────────────────────────────────────────
    with log_container.container():
        st.subheader("Event Log")
        for ev in st.session_state.event_log[-12:]:
            st.markdown(f"• {ev}")

# ──────────────────────────────────────────────────────────
#  Replay
# ──────────────────────────────────────────────────────────

def replay_simulation(replay_history: list):
    if not replay_history:
        st.warning("No replay data available.")
        return
    placeholder = st.empty()
    progress    = st.progress(0)
    n = len(replay_history)
    for i in range(0, n, max(1, n//200)):
        sub = pd.DataFrame(replay_history[:i+1])
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07)
        fig.add_trace(go.Scatter(x=sub["t"], y=sub["power"]*100, name="Power",
                                 line=dict(color="#00ff9d")), row=1, col=1)
        if "Tf" in sub.columns:
            fig.add_trace(go.Scatter(x=sub["t"], y=sub["Tf"],    name="Fuel T",
                                     line=dict(color="#ff9500")), row=2, col=1)
        if "rod_pos" in sub.columns:
            fig.add_trace(go.Scatter(x=sub["t"], y=sub["rod_pos"], name="Rods",
                                     line=dict(color="#ff69b4")), row=3, col=1)
        if "rho" in sub.columns:
            fig.add_trace(go.Scatter(x=sub["t"], y=sub["rho"],   name="ρ",
                                     line=dict(color="#c084fc")), row=4, col=1)
        fig.update_layout(height=700, template="plotly_dark", showlegend=True)
        placeholder.plotly_chart(fig, use_container_width=True)
        progress.progress(min(1.0, (i+1)/n))
        time.sleep(0.04)
    st.success("Replay finished.")

# ──────────────────────────────────────────────────────────
#  Comparison Mode
# ──────────────────────────────────────────────────────────

def run_comparison(reactor_type_a: str, scenario_a: str,
                   reactor_type_b: str, scenario_b: str, duration: float):
    results = {}
    for rt, sc_name in [(reactor_type_a, scenario_a), (reactor_type_b, scenario_b)]:
        sc      = SCENARIOS[sc_name]
        physics = ReactorPhysics(rt, SimConfig())
        state   = make_initial_state(physics, sc)
        hist    = []
        t = 0.0
        dt = physics.dt
        while t < duration:
            ext_rho = sc["ext_rho"](t)
            cooling = sc["cooling_factor"](t)
            az5     = sc["az5_active"](t)
            state   = physics.step(state, 0.0, ext_rho, cooling, az5)
            hist.append(state.to_dict())
            t += dt
        results[f"{rt} – {sc_name}"] = pd.DataFrame(hist)
    return results

# ──────────────────────────────────────────────────────────
#  Phase-Space Plot
# ──────────────────────────────────────────────────────────

def phase_space_plot(history: list) -> go.Figure:
    df  = pd.DataFrame(history)
    fig = go.Figure()
    if "rho" not in df.columns or "power" not in df.columns:
        return fig
    fig.add_trace(go.Scatter(
        x=df["rho"],
        y=df["power"] * 100,
        mode="lines+markers",
        marker=dict(size=3, color=df["t"], colorscale="Viridis",
                    colorbar=dict(title="Time [s]", thickness=12)),
        line=dict(width=1, color="rgba(255,255,255,0.3)"),
        name="Phase trajectory",
    ))
    fig.add_vline(x=0, line=dict(color="white", dash="dash", width=1))
    fig.update_layout(
        title="Phase-Space: Reactivity vs Power",
        xaxis_title="Reactivity ρ [Δk/k]",
        yaxis_title="Power [% nominal]",
        template="plotly_dark",
        height=500,
    )
    return fig

# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="NPP Simulator v11.0",
        layout="wide",
        page_icon="☢️",
    )
    st.title("☢️ Nuclear Power Plant Simulator – v11.0 Enhanced Edition")
    st.caption(
        "PWR · BWR · RBMK · SMR · CANDU  │  "
        "RK4 Kinetics · Coupled Xenon-ODE · Differential Void · "
        "ECCS · MC · Sensitivity · Phase-Space · AI Advisor · Free Play"
    )

    # ── Session state init ──────────────────────────────────
    defaults = {
        "sim_active":      False,
        "paused":          False,
        "history":         [],
        "event_log":       [],
        "replay_data":     None,
        "sim_config":      SimConfig(),
        "pid":             {"integral": 0.0, "prev_e": 0.0},
        "active_alarms":   set(),
        "op_inject_boron": False,
        "op_eccs":         False,
        "op_manual_scram": False,
        "damage_alerted":  False,
        "sim_start_wall":  None,
        "turbo_mode":      False,
        "realtime_sync":   False,
        "live_fig":        None,
        "mc_band_data":    None,
        "_play_beep":      False,
        "fp_ext_rho":      0.0,
        "fp_cooling":      1.0,
        "fp_rod_speed":    0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Simulator Controls")
        reactor_type  = st.selectbox("Reactor Type",  list(REACTOR_CONFIGS.keys()))
        scenario_name = st.selectbox("Scenario",       list(SCENARIOS.keys()))
        st.caption(SCENARIOS[scenario_name]["desc"])

        is_free_play = SCENARIOS[scenario_name].get("free_play", False)

        setpoint  = st.slider("Power Setpoint [%]",  5, 110, 100, step=5) / 100.0
        duration  = st.slider("Duration [s]",        60, 86400, 900, step=60)
        speed_x   = st.slider("Speed multiplier ×",   1, 50, 10)

        # Free Play controls
        if is_free_play:
            st.divider()
            st.subheader("🎮 Free Play Controls")
            fp_ext_rho = st.slider("External ρ [×10⁻³ Δk/k]", -30, 30, 0) * 1e-3
            fp_cooling  = st.slider("Cooling Factor",  0.0, 1.5, 1.0, step=0.05)
            fp_rod_spd  = st.slider("Rod Speed [%/s]", -20.0, 20.0, 0.0, step=0.5)
            st.session_state.fp_ext_rho  = fp_ext_rho
            st.session_state.fp_cooling  = fp_cooling
            st.session_state.fp_rod_speed = fp_rod_spd

        st.subheader("PID Tuning")
        st.session_state.sim_config.kp = st.slider("Kp", 0.5, 15.0,
                                                    st.session_state.sim_config.kp, 0.5)
        st.session_state.sim_config.ki = st.slider("Ki", 0.0, 1.5,
                                                    st.session_state.sim_config.ki, 0.05)
        st.session_state.sim_config.kd = st.slider("Kd", 0.0, 25.0,
                                                    st.session_state.sim_config.kd, 1.0)

        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("🚀 Start / Restart", type="primary"):
            st.session_state.sim_active     = True
            st.session_state.paused         = False
            st.session_state.history        = []
            st.session_state.event_log      = []
            st.session_state.active_alarms  = set()
            st.session_state.pid            = {"integral": 0.0, "prev_e": 0.0}
            st.session_state.damage_alerted = False
            st.session_state.sim_start_wall = time.time()
            st.session_state.live_fig       = None
            st.session_state.mc_band_data   = None
            # Kick off background MC band
            _holder: dict = {}
            th = threading.Thread(
                target=_run_mc_band_bg,
                args=(reactor_type, scenario_name, min(duration, 300.0), _holder),
                daemon=True)
            th.start()
            st.session_state["_mc_band_holder"] = _holder
            st.rerun()

        pause_label = "⏸ Pause" if not st.session_state.paused else "▶ Resume"
        if c2.button(pause_label):
            st.session_state.paused = not st.session_state.paused
            if not st.session_state.paused:
                st.session_state.sim_start_wall = time.time()
            st.rerun()

        if st.button("🛑 Abort"):
            st.session_state.sim_active     = False
            st.session_state.paused         = False
            st.session_state.damage_alerted = False
            st.session_state.turbo_mode     = False
            st.rerun()

        # Turbo button
        turbo_label = "⚡ Turbo OFF (100×)" if st.session_state.turbo_mode else "⚡ Turbo ON (100×)"
        if st.button(turbo_label):
            st.session_state.turbo_mode = not st.session_state.turbo_mode
            st.rerun()

        # Real-time sync toggle
        rt_label = "🕒 Real-Time Sync: ON" if st.session_state.realtime_sync else "🕒 Real-Time Sync: OFF"
        if st.button(rt_label):
            st.session_state.realtime_sync = not st.session_state.realtime_sync
            st.rerun()

        st.divider()
        st.subheader("🕹️ Operator Actions")
        st.caption("Space=Pause  R=SCRAM  (keyboard shortcuts active)")
        if st.button("💉 Emergency Boron (+200 ppm)"):
            st.session_state.op_inject_boron = True
        if st.button("💧 Activate ECCS"):
            st.session_state.op_eccs = True
        if st.button("🛑 Manual SCRAM"):
            st.session_state.op_manual_scram = True

        st.divider()
        st.subheader("🤖 AI Operator Assistant")
        if st.button("💡 What should I do?"):
            if st.session_state.history:
                s = ReactorState.from_dict(st.session_state.history[-1])
                advice = ai_operator_advice(s)
                st.info(advice)
            else:
                st.info("Start a simulation first.")

        st.divider()
        st.subheader("💾 Save / Load State")
        if st.button("💾 Save state to JSON"):
            if st.session_state.history:
                payload = json.dumps(st.session_state.history[-1], indent=2)
                st.download_button("⬇ Download state.json",
                                   payload, "npp_state.json", "application/json")
        uploaded = st.file_uploader("📂 Load state from JSON", type="json")
        if uploaded is not None:
            try:
                d = json.load(uploaded)
                st.session_state.history = [d]
                st.session_state.event_log.append("📂 State loaded from JSON")
                st.success("State loaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load: {e}")

        # Cache clearing
        if st.button("🗑 Clear MC / Sensitivity Cache"):
            run_monte_carlo.clear()
            compute_sensitivity.clear()
            st.success("Cache cleared.")

    # ── Poll background MC band ──────────────────────────────
    holder = st.session_state.get("_mc_band_holder", {})
    if "data" in holder and st.session_state.mc_band_data is None:
        st.session_state.mc_band_data = holder["data"]
        st.session_state.live_fig = None  # force rebuild to include band

    # ── Tabs ────────────────────────────────────────────────
    tabs_list = ["📡 Live", "🌀 Phase-Space", "🎲 Monte Carlo",
                 "📊 Sensitivity", "⚖️ Compare", "🎬 Export", "ℹ️ About"]
    if SAFETY_MODULES_AVAILABLE:
        tabs_list.append("🔐 Safety Design")

    tab_refs   = st.tabs(tabs_list)
    tab_live   = tab_refs[0]
    tab_phase  = tab_refs[1]
    tab_mc     = tab_refs[2]
    tab_sa     = tab_refs[3]
    tab_cmp    = tab_refs[4]
    tab_exp    = tab_refs[5]
    tab_info   = tab_refs[6]
    tab_safety = tab_refs[7] if SAFETY_MODULES_AVAILABLE else None

    # ── LIVE ───────────────────────────────────────────────
    with tab_live:
        if st.session_state.turbo_mode and st.session_state.sim_active:
            st.info("⚡ **Turbo mode active** — charts disabled during run for max speed.")
        if st.session_state.sim_active:
            run_live_simulation(reactor_type, scenario_name, setpoint, duration, speed_x)
        if st.session_state.replay_data and not st.session_state.sim_active:
            if st.button("▶ Replay Last Simulation"):
                replay_simulation(st.session_state.replay_data)

    # ── PHASE-SPACE ────────────────────────────────────────
    with tab_phase:
        st.subheader("Phase-Space: Reactivity vs Power")
        if st.session_state.history:
            fig = phase_space_plot(st.session_state.history)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Color encodes simulation time (dark→early, bright→late). "
                       "Stable reactors converge to ρ=0. Unstable excursions diverge right.")
        else:
            st.info("Run a simulation first to see the phase-space trajectory.")

    # ── MONTE CARLO ────────────────────────────────────────
    with tab_mc:
        st.subheader("Monte Carlo Uncertainty Analysis")
        mc_col1, mc_col2 = st.columns(2)
        num_runs    = mc_col1.slider("Number of runs",   10, 300, 50)
        mc_duration = mc_col2.slider("MC Duration [s]", 60, 900, 300, step=30)
        if st.button("▶ Run Monte Carlo Ensemble"):
            with st.spinner(f"Running {num_runs} simulations…"):
                mc = run_monte_carlo(reactor_type, scenario_name, num_runs, mc_duration)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mc["time"], y=mc["p95"],
                fill=None, mode="lines", line=dict(color="rgba(0,255,150,0)"), showlegend=False))
            fig.add_trace(go.Scatter(x=mc["time"], y=mc["p5"],
                fill="tonexty", line=dict(color="rgba(0,255,150,0)"),
                name="5–95 percentile band", fillcolor="rgba(0,200,120,0.15)"))
            fig.add_trace(go.Scatter(x=mc["time"], y=mc["mean"]+mc["std"],
                fill=None, mode="lines", line=dict(color="rgba(0,255,150,0)"), showlegend=False))
            fig.add_trace(go.Scatter(x=mc["time"], y=mc["mean"]-mc["std"],
                fill="tonexty", line=dict(color="rgba(0,255,150,0)"),
                name="±1σ band", fillcolor="rgba(0,255,150,0.2)"))
            fig.add_trace(go.Scatter(x=mc["time"], y=mc["mean"],
                name="Mean Power", line=dict(width=3, color="#00ff9d")))
            fig.update_layout(
                title=f"Power Uncertainty – {reactor_type} – {scenario_name}",
                xaxis_title="Time [s]", yaxis_title="Power [% nominal]",
                template="plotly_dark", height=550)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Sampled {num_runs} runs with ±10–20% uncertainty on "
                       "void, Doppler, neutron lifetime, and decay heat coefficients.")
        if st.button("🗑 Clear MC Cache "):
            run_monte_carlo.clear()
            st.success("Monte Carlo cache cleared.")

    # ── SENSITIVITY ────────────────────────────────────────
    with tab_sa:
        st.subheader("Local Sensitivity Analysis (OAT)")
        if not st.session_state.history:
            st.info("Run at least one simulation first.")
        else:
            if st.button("▶ Compute Sensitivities"):
                with st.spinner("Running perturbed cases…"):
                    sens_df = compute_sensitivity(reactor_type, scenario_name, setpoint)
                st.dataframe(sens_df.style.format(precision=4), use_container_width=True)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=sens_df["Parameter"], y=sens_df["S_power"],
                                     name="Power sensitivity",  marker_color="#00cc96"))
                fig.add_trace(go.Bar(x=sens_df["Parameter"], y=sens_df["S_Tf"],
                                     name="Fuel T sensitivity", marker_color="#ff6b6b"))
                fig.update_layout(barmode="group",
                                  title="Tornado Plot – Relative Sensitivity Index",
                                  template="plotly_dark", height=420,
                                  xaxis_title="Parameter", yaxis_title="Sensitivity [–]")
                st.plotly_chart(fig, use_container_width=True)
            if st.button("🗑 Clear Sensitivity Cache"):
                compute_sensitivity.clear()
                st.success("Sensitivity cache cleared.")

    # ── COMPARE ────────────────────────────────────────────
    with tab_cmp:
        st.subheader("Scenario / Reactor Comparison")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Case A**")
            rt_a = st.selectbox("Reactor A", list(REACTOR_CONFIGS.keys()), key="cmp_rt_a")
            sc_a = st.selectbox("Scenario A", list(SCENARIOS.keys()), key="cmp_sc_a")
        with cc2:
            st.markdown("**Case B**")
            rt_b = st.selectbox("Reactor B", list(REACTOR_CONFIGS.keys()),
                                 index=2, key="cmp_rt_b")
            sc_b = st.selectbox("Scenario B", list(SCENARIOS.keys()),
                                 index=2, key="cmp_sc_b")
        cmp_dur = st.slider("Comparison duration [s]", 60, 3600, 600,
                             step=60, key="cmp_dur")
        if st.button("▶ Run Comparison"):
            with st.spinner("Running comparison simulations…"):
                cmp_results = run_comparison(rt_a, sc_a, rt_b, sc_b, float(cmp_dur))
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=("Power [% nominal]",
                                                "Fuel Temperature [°C]",
                                                "Reactivity [Δk/k]"))
            colors = ["#00ff9d", "#ff6b6b"]
            for idx, (label, df) in enumerate(cmp_results.items()):
                col = colors[idx % 2]
                fig.add_trace(go.Scatter(x=df["t"], y=df["power"]*100,
                    name=label, line=dict(color=col, width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df["t"], y=df["Tf"],
                    name=f"{label} Tf", line=dict(color=col, width=2, dash="dot"),
                    showlegend=False), row=2, col=1)
                fig.add_trace(go.Scatter(x=df["t"], y=df["rho"],
                    name=f"{label} ρ", line=dict(color=col, width=2, dash="dash"),
                    showlegend=False), row=3, col=1)
            fig.update_layout(height=700, template="plotly_dark",
                              title="Side-by-Side Comparison")
            st.plotly_chart(fig, use_container_width=True)

    # ── EXPORT ─────────────────────────────────────────────
    with tab_exp:
        st.subheader("🎬 Animation Export (GIF / MP4)")
        if not st.session_state.history:
            st.info("Run a simulation first.")
        else:
            fmt = st.radio("Format", ["gif", "mp4"], horizontal=True)
            if st.button("▶ Generate Animation"):
                if not IMAGEIO_OK:
                    st.error("Install `imageio` and `imageio-ffmpeg` to enable export.")
                else:
                    with st.spinner("Rendering frames…"):
                        data = export_animation(st.session_state.history, fmt)
                    if data:
                        mime = "image/gif" if fmt == "gif" else "video/mp4"
                        st.download_button(
                            f"⬇ Download .{fmt}", data,
                            f"npp_sim.{fmt}", mime)
                    else:
                        st.error("Export failed – check imageio/plotly-orca installation.")

    # ── ABOUT ──────────────────────────────────────────────
    with tab_info:
        st.markdown("## About NPP Simulator v1.0: (c) 2026 spsingh37@gmail.com")
        st.markdown("""

**Reactor Notes:**

| Reactor | Key Feature |
|---------|-------------|
| PWR | Pressurized water; negative void; boron control |
| BWR | Boiling water; negative void (differential dynamics) |
| RBMK | Positive moderator + void → unstable at low power |
| SMR | Small Modular; strong negative feedbacks |
| CANDU | Heavy water; online refuelling; negative void |

> *Educational simulator only – not a certified nuclear engineering tool.*
        """)

    # ── SAFETY DESIGN ──────────────────────────────────────
    if SAFETY_MODULES_AVAILABLE and tab_safety is not None:
        with tab_safety:
            _render_safety()


if __name__ == "__main__":
    main()
