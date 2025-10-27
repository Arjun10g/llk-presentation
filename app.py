from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal, Tuple, Dict, Any
from math import tan, radians, copysign, isfinite
import numpy as np
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
import time
from statsmodels.formula.api import mixedlm
import statsmodels.api as sm
import math

app = Flask(__name__)
# ------------------------------------
# Utility: favicon route
# ------------------------------------
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

# ------------------------------------
# Helper Functions
# ------------------------------------
# ============================================================
# HELPER FUNCTIONS FOR MLM
# ============================================================
def safe_cholesky(V):
    """Try Cholesky decomposition; return None if matrix not PD."""
    try:
        return np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        return None


def simulate_boundary(J=8, n_per=6, beta=(0.5, 1.0), tau2=0.03, sigma2=1.0):
    """Simulate random-intercept multilevel data."""
    if J < 2 or n_per < 2:
        raise ValueError("Need at least 2 clusters and 2 obs per cluster.")
    N = J * n_per
    x = np.linspace(-1, 1, N)
    group = np.repeat(np.arange(J), n_per)
    b = np.random.normal(0, np.sqrt(tau2), size=J)
    y = beta[0] + beta[1] * x + b[group] + np.random.normal(0, np.sqrt(sigma2), N)
    X = np.column_stack((np.ones(N), x))
    Z = np.zeros((N, J))
    Z[np.arange(N), group] = 1
    return y, X, Z, J, N


def loglik_mlm_fast(y, X, Z, tau2, sigma2, J, eps=1e-8):
    """Compute marginal log-likelihood for random-intercept model."""
    n = len(y)
    if tau2 <= 0 or sigma2 <= 0:
        return np.nan
    G = np.eye(J) * tau2
    R = np.eye(n) * sigma2
    V = Z @ G @ Z.T + R + np.eye(n) * eps
    L = safe_cholesky(V)
    if L is None:
        return np.nan
    Vinv_y = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Vinv_X = np.linalg.solve(L.T, np.linalg.solve(L, X))
    XtVinvX = X.T @ Vinv_X
    XtVinvY = X.T @ Vinv_y
    try:
        beta_hat = np.linalg.solve(XtVinvX, XtVinvY)
    except np.linalg.LinAlgError:
        return np.nan
    resid = y - X @ beta_hat
    Vinv_resid = np.linalg.solve(L.T, np.linalg.solve(L, resid))
    quad = resid.T @ Vinv_resid
    logdetV = 2 * np.sum(np.log(np.diag(L)))
    ll = -0.5 * (logdetV + quad + n * np.log(2 * np.pi))
    return float(ll)


# ============================================================
# LIKELIHOOD SURFACE ROUTE (NO TIMEOUT, CACHED)
# ============================================================
@app.route('/compute_surface', methods=['POST'])
def compute_surface():
    try:
        params = request.json or {}
        J = int(params.get("J", 8))
        n_per = int(params.get("n_per", 6))
        tau_max = float(params.get("tauMax", 0.15))
        sigma_max = float(params.get("sigmaMax", 1.5))
        steps = int(params.get("steps", 25))

        # --- safety caps ---
        J = max(2, min(J, 12))
        n_per = max(2, min(n_per, 10))
        steps = max(5, min(steps, 30))

        # --- simulate data ---
        y, X, Z, J, n = simulate_boundary(J, n_per)

        tau_seq = np.linspace(1e-4, tau_max, steps)
        sig_seq = np.linspace(0.2, sigma_max, steps)
        logL = np.full((steps, steps), np.nan)

        # --- caching dict (simple, no rounding bug) ---
        cache: Dict[Tuple[float, float], float] = {}

        def cached_ll(t2, s2):
            key = (float(t2), float(s2))
            if key in cache:
                return cache[key]
            val = loglik_mlm_fast(y, X, Z, t2, s2, J)
            if not np.isfinite(val):
                val = np.nan
            cache[key] = val
            return val

        print(f"Computing likelihood surface: J={J}, n_per={n_per}, steps={steps}")
        for i, t2 in enumerate(tau_seq):
            for j, s2 in enumerate(sig_seq):
                if j == 0 and i % 5 == 0:
                    print(f" → Row {i}/{steps} (tau2={t2:.4f})")
                ll = cached_ll(t2, s2)
                logL[i, j] = ll
                

        if np.all(np.isnan(logL)):
            return jsonify({"error": "All log-likelihood values invalid (singular matrices)."}), 500

        # --- locate MLE ---
        max_idx = np.nanargmax(logL)
        mle_i, mle_j = np.unravel_index(max_idx, logL.shape)
        mle_tau, mle_sigma, mle_val = float(tau_seq[mle_i]), float(sig_seq[mle_j]), float(logL[mle_i, mle_j])

        # --- profiles (robust) ---
        tau_prof = [cached_ll(t2, mle_sigma) for t2 in tau_seq]
        sig_prof = [cached_ll(mle_tau, s2) for s2 in sig_seq]

        # --- clean NaNs for JSON ---
        def clean(arr):
            return [float(a) if np.isfinite(a) else None for a in arr]

        return jsonify({
            "tau": tau_seq.tolist(),
            "sigma": sig_seq.tolist(),
            "logLik": np.nan_to_num(logL, nan=np.nanmin(logL)).tolist(),
            "mle": {"tau": mle_tau, "sigma": mle_sigma, "value": mle_val},
            "tauProfile": clean(tau_prof),
            "sigmaProfile": clean(sig_prof)
        })

    except Exception as e:
        print("❌ compute_surface error:", e)
        return jsonify({"error": str(e)}), 500
#######
# Matrix anatomy route
#######


@app.route("/matrix_anatomy", methods=["POST"])
def matrix_anatomy():
    try:
        params = request.json or {}
        J = int(params.get("J", 4))
        n_per = int(params.get("n_per", 5))
        tau2_int = float(params.get("tau2", 0.05))
        tau2_slope = float(params.get("tau2_slope", 0.02))
        rho = float(params.get("rho", 0.3))
        sigma2 = float(params.get("sigma2", 1.0))
        structure = params.get("structure", "intercept")  # "intercept" or "slope"

        # --- simulate ---
        N = J * n_per
        x = np.linspace(-1, 1, N)
        group = np.repeat(np.arange(J), n_per)

        if structure == "intercept":
            G = np.eye(J) * tau2_int
            Z = np.zeros((N, J))
            Z[np.arange(N), group] = 1
        else:  # random slope
            # G per cluster: 2x2 covariance
            G_cluster = np.array([
                [tau2_int, rho * np.sqrt(tau2_int * tau2_slope)],
                [rho * np.sqrt(tau2_int * tau2_slope), tau2_slope]
            ])
            G = np.kron(np.eye(J), G_cluster)

            Z = np.zeros((N, 2 * J))
            for j in range(J):
                idx = group == j
                Z[idx, 2 * j] = 1           # intercept
                Z[idx, 2 * j + 1] = x[idx]  # slope

        R = np.eye(N) * sigma2
        V = Z @ G @ Z.T + R
        Vinv = np.linalg.inv(V)

        # --- variance partitioning ---
        var_between = np.mean(np.diag(Z @ G @ Z.T))
        var_within = np.mean(np.diag(R))
        total_var = var_between + var_within
        icc = var_between / total_var

        return jsonify({
            "G": G.tolist(),
            "R": R.tolist(),
            "V": V.tolist(),
            "Vinv": Vinv.tolist(),
            "icc": icc,
            "components": {
                "between": var_between,
                "within": var_within,
                "total": total_var
            },
            "structure": structure
        })
    except Exception as e:
        print("❌ Matrix anatomy error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/simulate_fxre_visual")
def simulate_fxre_visual():
    """Simulate grouped data with fixed and random effects for D3 animation."""
    J, n_per = 10, 25
    beta0, beta1 = 1.5, 1.2
    tau2, sigma2 = 1.0, 1.0

    rng = np.random.default_rng(123)
    group = np.repeat(np.arange(J), n_per)
    x = np.tile(np.linspace(-2, 2, n_per), J)
    eps = rng.normal(0, np.sqrt(sigma2), len(x))
    b = rng.normal(0, np.sqrt(tau2), J)

    y = beta0 + beta1 * x + b[group] + eps  # random-intercept model

    # Return structure for D3
    data = []
    for j in range(J):
        idx = (group == j)
        pts = [{"x": float(xi), "y": float(yi), "g": int(j)} for xi, yi in zip(x[idx], y[idx])]
        data.append({"group": int(j), "points": pts})

    return jsonify({"groups": data, "meta": {"J": J, "n_per": n_per}})

# ------------------------------------------------------------
# Constants & Tolerances
# ------------------------------------------------------------
# World coordinates (must match the D3 viewBox horizontally/vertically)
X_MIN_DEFAULT = -4.0
X_MAX_DEFAULT =  4.0
Y_MIN_DEFAULT = -2.2
Y_MAX_DEFAULT =  3.6

# Physics & numerical tolerances
EPS            = 1e-12       # tiny tolerance
VX_AT_REST     = 2e-3        # rest threshold for vx
VY_AT_REST     = 2e-3        # rest threshold for vy
Y_SURF_TOL     = 1e-2        # distance to surface for "touching"
FRICTION_DECAY = 0.985       # generic friction factor per step

# Step size for curve generation (server → client path sampling)
DEFAULT_STEP = 0.05

# ------------------------------------------------------------
# Configuration & State
# ------------------------------------------------------------
Mode = Literal["quadratic", "boundary"]

@dataclass
class VizConfig:
    """Global configuration that defines the surface and physics constants."""
    mode: Mode = "quadratic"
    a: float = 1.0                # curvature for quadratic (a > 0)
    angle_deg: float = 30.0       # walls angle for boundary mode (0–80)
    gravity: float = 0.05         # vertical acceleration per tick
    damping: float = 0.82         # bounce damping for vy on impact
    roll_factor: float = 0.02     # horizontal acceleration magnitude along -grad
    friction: float = FRICTION_DECAY  # friction decay factor per step
    x_min: float = X_MIN_DEFAULT
    x_max: float = X_MAX_DEFAULT
    y_min: float = Y_MIN_DEFAULT
    y_max: float = Y_MAX_DEFAULT

    def slope_k(self) -> float:
        """
        Return the positive slope k used for the boundary V (y = k|x|).
        We clamp the angle to a safe range to avoid absurd slopes.
        """
        ang = max(0.0, min(80.0, float(self.angle_deg)))
        return tan(radians(ang))

    def validate(self) -> None:
        """
        Validate that parameters are finite and within safe ranges.
        Raises ValueError if invalid.
        """
        if self.mode not in ("quadratic", "boundary"):
            raise ValueError("mode must be 'quadratic' or 'boundary'")
        if not isfinite(self.a) or self.a <= 0:
            raise ValueError("quadratic curvature 'a' must be finite and > 0")
        if not isfinite(self.gravity) or self.gravity <= 0:
            raise ValueError("gravity must be finite and > 0")
        if not (0.0 < self.damping < 1.0):
            raise ValueError("damping must be in (0,1)")
        if not (0.0 < self.roll_factor < 1.0):
            raise ValueError("roll_factor must be in (0,1)")
        if not (0.9 <= self.friction <= 1.0):
            raise ValueError("friction must be in [0.9, 1.0]")
        if not (self.x_min < self.x_max):
            raise ValueError("x_min must be < x_max")
        if not (self.y_min < self.y_max):
            raise ValueError("y_min must be < y_max")


@dataclass
class SimState:
    """
    Simulation state for the marble.
    x, y : position in world coordinates
    vx, vy : velocities
    """
    x: float = -3.5
    y: float = -1.0
    vx: float = 0.0
    vy: float = 0.0

    def clamp_x(self, cfg: VizConfig) -> None:
        """Clamp x so we never leave drawable world bounds."""
        self.x = max(cfg.x_min, min(cfg.x_max, self.x))


# Global instances (simple in-memory “session”)
CFG = VizConfig()
ST  = SimState()

# ------------------------------------------------------------
# Surface & Gradient Functions
# ------------------------------------------------------------
def surface_y(cfg: VizConfig, x: float) -> float:
    """
    Return floor/surface height for a given x based on cfg.mode.
    - quadratic: y = a*x^2
    - boundary : y = k*|x|
    """
    if cfg.mode == "quadratic":
        return cfg.a * x * x
    # boundary mode
    return cfg.slope_k() * abs(x)


def surface_grad(cfg: VizConfig, x: float) -> float:
    """
    Return dy/dx at x for the active surface:
    - quadratic: d/dx (a*x^2) = 2*a*x
    - boundary : subgradient of |x| → ±k, 0 at x=0
    """
    if cfg.mode == "quadratic":
        return 2.0 * cfg.a * x
    k = cfg.slope_k()
    if x > 0:  return k
    if x < 0:  return -k
    return 0.0


def vertex(cfg: VizConfig) -> Tuple[float, float]:
    """
    The minimum point for both surfaces:
    - quadratic y=a*x^2 has vertex at (0,0)
    - boundary V y=k|x| has vertex at (0,0)
    """
    return (0.0, 0.0)


def world_bounds(cfg: VizConfig) -> Tuple[float, float, float, float]:
    """Return world bounds (x_min, x_max, y_min, y_max)."""
    return cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max


# ------------------------------------------------------------
# Curve Generation (for front-end path drawing)
# ------------------------------------------------------------
def generate_curve_points(cfg: VizConfig,
                          xmin: float,
                          xmax: float,
                          step: float) -> Dict[str, Any]:
    """
    Sample the active surface on [xmin, xmax] with spacing `step`.
    Returns dict with x[], y[], grad[] and vertex (x0,y0).
    """
    xmin, xmax = float(xmin), float(xmax)
    step = max(EPS, float(step))
    if xmin >= xmax:
        xmin, xmax = sorted((xmin, xmax))

    xs = np.arange(xmin, xmax + step, step, dtype=float)
    ys = np.array([surface_y(cfg, float(xx)) for xx in xs], dtype=float)
    gs = np.array([surface_grad(cfg, float(xx)) for xx in xs], dtype=float)

    vx0, vy0 = vertex(cfg)
    return dict(
        mode=cfg.mode,
        a=cfg.a,
        angle_deg=cfg.angle_deg,
        x=xs.tolist(),
        y=ys.tolist(),
        grad=gs.tolist(),
        vertex=[vx0, vy0]
    )


# ------------------------------------------------------------
# Collision & Constraint Logic
# ------------------------------------------------------------
def project_to_surface_if_below(cfg: VizConfig, st: SimState) -> None:
    """
    If the marble has penetrated below the surface (y > surface_y),
    snap it back to the surface and flip vy with damping.
    """
    fy = surface_y(cfg, st.x)
    if st.y > fy:
        st.y  = fy
        st.vy = -cfg.damping * st.vy
        st.vx *= 0.9  # horizontal impulse loss on impact


def enforce_boundary_walls(cfg: VizConfig, st: SimState) -> None:
    """
    Boundary mode: V-shaped floor y = k|x|.
    If the marble penetrates below the V (y > k|x|), snap to the wall and bounce.
    """
    if cfg.mode != "boundary":
        return

    k = cfg.slope_k()
    required_y = k * abs(st.x)        # floor height at current x
    if st.y > required_y:             # penetrated (below floor, since y grows downward)
        st.y  = required_y
        st.vy = -cfg.damping * st.vy  # bounce
        st.vx *= 0.9                  # lose a bit of horizontal speed on impact



# ------------------------------------------------------------
# Physics Integrator (semi-implicit Euler)
# ------------------------------------------------------------
def apply_gravity(cfg: VizConfig, st: SimState) -> None:
    """vy += g ; y += vy   (semi-implicit Euler update)"""
    st.vy += cfg.gravity
    st.y  += st.vy


def apply_surface_rolling(cfg: VizConfig, st: SimState) -> None:
    """
    Horizontal acceleration proportional to negative gradient:
      ax = -roll_factor * (dy/dx)
    Then semi-implicit Euler:
      vx += ax ; x += vx
    """
    grad = surface_grad(cfg, st.x)
    ax   = -cfg.roll_factor * grad
    st.vx += ax
    st.vx *= 0.98  # viscous damping separate from global friction
    st.x  += st.vx
    st.clamp_x(cfg)


def apply_global_friction(cfg: VizConfig, st: SimState) -> None:
    """Generic drag applied to both velocity components every step."""
    st.vx *= cfg.friction
    st.vy *= cfg.friction


def at_rest(cfg: VizConfig, st: SimState) -> bool:
    """Heuristic: near-zero velocities and on the surface."""
    fy = surface_y(cfg, st.x)
    return (abs(st.vx) < VX_AT_REST and
            abs(st.vy) < VY_AT_REST and
            abs(st.y - fy) < Y_SURF_TOL)


def physics_step(cfg: VizConfig, st: SimState) -> None:
    """
    One simulation tick: gravity → rolling → collision → boundary walls → friction.
    Order matters:
      1) integrate free-fall
      2) roll along gradient (x dynamics)
      3) snap to surface if penetrated (bounce)
      4) enforce V walls (boundary mode only)
      5) apply global friction
    """
    apply_gravity(cfg, st)
    apply_surface_rolling(cfg, st)
    project_to_surface_if_below(cfg, st)
    enforce_boundary_walls(cfg, st)
    apply_global_friction(cfg, st)


# ------------------------------------------------------------
# Serialization Helpers
# ------------------------------------------------------------
def state_json(cfg: VizConfig, st: SimState) -> Dict[str, Any]:
    """Pack current config and state as JSON-friendly dict."""
    return dict(
        config=asdict(cfg),
        state=asdict(st),
        vertex=vertex(cfg),
        bounds=dict(x_min=cfg.x_min, x_max=cfg.x_max, y_min=cfg.y_min, y_max=cfg.y_max),
    )


# ------------------------------------------------------------
# REST API
# ------------------------------------------------------------
@app.get("/mlmviz/curve")
def api_curve():
    """
    Sample the surface for drawing.
    Query:
      mode       = 'quadratic' | 'boundary' (default: cfg.mode)
      a          = curvature > 0             (quadratic only)
      angle_deg  = 0..80                     (boundary only)
      xmin, xmax = sampling range (defaults to cfg bounds)
      step       = sampling step (default 0.05)
    """
    # clone current config so queries don’t mutate global until /config is called
    cfg = VizConfig(**asdict(CFG))
    mode = request.args.get("mode")
    if mode in ("quadratic", "boundary"):
        cfg.mode = mode

    if "a" in request.args:
        cfg.a = max(EPS, float(request.args.get("a", cfg.a)))

    if "angle_deg" in request.args:
        cfg.angle_deg = float(request.args.get("angle_deg", cfg.angle_deg))

    xmin = float(request.args.get("xmin", cfg.x_min))
    xmax = float(request.args.get("xmax", cfg.x_max))
    step = float(request.args.get("step", DEFAULT_STEP))

    # guard & validate
    cfg.validate()
    data = generate_curve_points(cfg, xmin, xmax, step)
    return jsonify(data)


@app.get("/mlmviz/state")
def api_state():
    """Return current (global) configuration + state."""
    return jsonify(state_json(CFG, ST))


@app.get("/mlmviz/state/reset")
def api_state_reset():
    """
    Reset the simulation state to a canonical starting point:
      - start left of origin, slightly above the surface
      - zero velocities
    """
    ST.vx = ST.vy = 0.0
    ST.x  = -3.5
    # start just above the floor to avoid initial bounce explosion
    ST.y  = surface_y(CFG, ST.x) - 0.2
    return jsonify(state_json(CFG, ST))


@app.post("/mlmviz/state/config")
def api_state_config():
    """
    Update global configuration (mode, a, angle, physics params).
    Body (JSON) any subset of:
      { "mode": "quadratic"|"boundary",
        "a": <float>,
        "angle_deg": <float>,
        "gravity": <float>,
        "damping": <0..1>,
        "roll_factor": <0..1>,
        "friction": <0.9..1>,
        "x_min": <float>, "x_max": <float>, "y_min": <float>, "y_max": <float> }
    Returns updated config + state.
    """
    data = request.get_json(silent=True) or {}

    if "mode" in data:
        CFG.mode = str(data["mode"])
    if "a" in data:
        CFG.a = float(data["a"])
    if "angle_deg" in data:
        CFG.angle_deg = float(data["angle_deg"])
    if "gravity" in data:
        CFG.gravity = float(data["gravity"])
    if "damping" in data:
        CFG.damping = float(data["damping"])
    if "roll_factor" in data:
        CFG.roll_factor = float(data["roll_factor"])
    if "friction" in data:
        CFG.friction = float(data["friction"])
    if "x_min" in data:
        CFG.x_min = float(data["x_min"])
    if "x_max" in data:
        CFG.x_max = float(data["x_max"])
    if "y_min" in data:
        CFG.y_min = float(data["y_min"])
    if "y_max" in data:
        CFG.y_max = float(data["y_max"])

    # Validate new configuration
    CFG.validate()
    # Clamp state if current x is now outside bounds
    ST.clamp_x(CFG)
    # Ensure y is not below the surface after config changes
    fy = surface_y(CFG, ST.x)
    if ST.y > fy:
        ST.y = fy

    return jsonify(state_json(CFG, ST))


@app.post("/mlmviz/state/step")
def api_state_step():
    """
    Advance physics by n steps and return updated state.
    Body JSON:
      { "n": <int>, "return_trace": <bool> }
    If return_trace true, we include arrays of (x,y) positions for each step.
    """
    data = request.get_json(silent=True) or {}
    n = int(data.get("n", 1))
    return_trace = bool(data.get("return_trace", False))
    n = max(1, min(2000, n))  # guard runaway requests

    xs, ys = [], []
    for _ in range(n):
        physics_step(CFG, ST)
        if return_trace:
            xs.append(ST.x)
            ys.append(ST.y)
        if at_rest(CFG, ST):
            break

    resp = state_json(CFG, ST)
    if return_trace:
        resp["trace"] = {"x": xs, "y": ys}
    resp["at_rest"] = at_rest(CFG, ST)
    return jsonify(resp)

@app.route("/simulate_randomslope_full", methods=["POST"])
def simulate_randomslope_full():
    data = request.get_json()

    n = int(data.get("n", 50))
    tau00 = float(data.get("tau00", 1.0))
    tau11 = float(data.get("tau11", 0.5))
    tau01 = float(data.get("tau01", 0.3))
    sigma2 = float(data.get("sigma2", 0.2))

    # Covariate vector
    x = np.linspace(-2, 2, n)

    # Construct G0 and Z_j
    G0 = np.array([[tau00, tau01],
                   [tau01, tau11]])
    Zj = np.column_stack((np.ones(n), x))

    # Compute marginal covariance
    Vj = Zj @ G0 @ Zj.T + sigma2 * np.eye(n)

    # Extract diagonal and correlation matrix
    diagV = np.diag(Vj)
    corrV = Vj / np.sqrt(np.outer(diagV, diagV))

    return jsonify({
        "x": x.tolist(),
        "diagV": diagV.tolist(),
        "covV": Vj.tolist(),
        "corrV": corrV.tolist()
    })

@app.route("/simulate_shrinkage_dynamics", methods=["POST"])
def simulate_shrinkage_dynamics():
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    import pandas as pd

    params = request.get_json() or {}
    J = int(params.get("J", 8))
    n_per = int(params.get("n_per", 10))
    tau2_true = float(params.get("tau2", 1.0))
    sigma2_true = float(params.get("sigma2", 1.0))
    reml = bool(params.get("reml", True))
    steps = int(params.get("steps", 25))

    rng = np.random.default_rng(42)
    group = np.repeat(np.arange(J), n_per)
    x = np.tile(np.linspace(-1, 1, n_per), J)
    b0 = rng.normal(0, np.sqrt(tau2_true), J)
    y = 2.0 + b0[group] + rng.normal(0, np.sqrt(sigma2_true), J * n_per)
    df = pd.DataFrame({"y": y, "x": x, "group": group})

    # Create ML and REML fits
    model = mixedlm("y ~ 1", df, groups=df["group"])
    fit_ml = model.fit(reml=False)
    fit_reml = model.fit(reml=True)

    def extract_betas(fit):
        re = fit.random_effects
        return np.array([float(re[j][0]) for j in range(J)])

    blup_ml = extract_betas(fit_ml)
    blup_reml = extract_betas(fit_reml)

    # Generate shrinkage frames across τ² evolution
    tau_seq = np.linspace(0.01, tau2_true * 2, steps)
    frames = []
    for t2 in tau_seq:
        shrink = t2 / (t2 + sigma2_true / n_per)
        blup_dynamic = shrink * b0
        frames.append({
            "tau2": float(t2),
            "shrink_ratio": float(shrink),
            "blup_dynamic": blup_dynamic.tolist(),
        })

    return jsonify({
        "group": list(range(J)),
        "true_b": b0.tolist(),
        "blup_ml": blup_ml.tolist(),
        "blup_reml": blup_reml.tolist(),
        "frames": frames,
        "meta": {
            "tau2": tau2_true,
            "sigma2": sigma2_true,
            "J": J,
            "n_per": n_per,
            "reml": reml
        }
    })

def safe_num(x):
    """Convert NaN/Inf to None for valid JSON."""
    if isinstance(x, (float, int)):
        if math.isnan(x) or math.isinf(x):
            return None
        return float(x)
    return x

def sanitize(obj):
    """Recursively sanitize dicts/lists of numbers for JSON."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    else:
        return safe_num(obj)

# ======================================================
# Quick ML vs REML Comparison Endpoint
# ======================================================
@app.route("/mlreml_quick", methods=["POST"])
def mlreml_quick():
    """
    Simulate a multilevel dataset with optional random slopes,
    fit ML and REML using statsmodels.mixedlm,
    and return variance + fixed effect estimates.
    """
    try:
        # ---------- Parse user inputs ----------
        p = request.get_json(silent=True) or {}
        J       = int(p.get("J", 8))              # clusters
        n_per   = int(p.get("n_per", 10))         # observations per cluster
        tau2_int   = float(p.get("tau2_int", 1.0))  # intercept variance
        tau2_slope = float(p.get("tau2_slope", 0.5)) # slope variance
        rho     = float(p.get("rho", 0.0))        # intercept–slope correlation
        sigma2  = float(p.get("sigma2", 1.0))     # residual variance
        beta0   = float(p.get("beta0", 2.0))
        beta1   = float(p.get("beta1", 0.5))
        slope   = bool(p.get("random_slope", False))
        seed    = int(p.get("seed", 123))

        rng = np.random.default_rng(seed)

        # ---------- Simulate data ----------
        group = np.repeat(np.arange(J), n_per)
        # x slightly jittered per group for realism
        x = np.concatenate([
            np.linspace(-1, 1, n_per) + rng.normal(0, 0.05, n_per)
            for _ in range(J)
        ])

        if slope:
            # Full covariance matrix G0
            cov01 = rho * np.sqrt(tau2_int * tau2_slope)
            G0 = np.array([[tau2_int, cov01],
                           [cov01, tau2_slope]])

            # Simulate random effects (b0j, b1j)
            b = rng.multivariate_normal(np.zeros(2), G0, size=J)
            b0, b1 = b[:, 0], b[:, 1]

            # Generate response
            y = (beta0 + beta1 * x +
                 b0[group] + b1[group] * x +
                 rng.normal(0, np.sqrt(sigma2), J * n_per))

            df = pd.DataFrame({"y": y, "x": x, "group": group})
            formula = "y ~ x"
            re_formula = "~x"

        else:
            # Random intercept only
            b0 = rng.normal(0, np.sqrt(tau2_int), J)
            y = beta0 + beta1 * x + b0[group] + rng.normal(0, np.sqrt(sigma2), J * n_per)
            df = pd.DataFrame({"y": y, "x": x, "group": group})
            formula = "y ~ x"
            re_formula = "1"

        # ---------- Fit ML and REML ----------
        model = mixedlm(formula, df, groups=df["group"],
                        re_formula=(None if re_formula == "1" else re_formula))

        result_ml = model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
        result_reml = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)

        # ---------- Extract results ----------
        def extract_variances(res, use_slope):
            cov_re = np.atleast_2d(res.cov_re)
            if use_slope and cov_re.shape[0] >= 2:
                tau2_int_est   = float(cov_re[0, 0])
                tau2_slope_est = float(cov_re[1, 1])
                tau01_est      = float(cov_re[0, 1])
            else:
                tau2_int_est   = float(cov_re[0, 0])
                tau2_slope_est = None
                tau01_est      = None
            sigma2_hat = float(res.scale)
            return dict(
                tau2_intercept=tau2_int_est,
                tau2_slope=tau2_slope_est,
                tau01=tau01_est,
                sigma2=sigma2_hat
            )

        def extract_fixed(res):
            fe = res.fe_params
            se = res.bse_fe
            return {
                "coef": {k: float(fe[k]) for k in fe.index},
                "se":   {k: float(se[k]) for k in se.index}
            }

        out = {
            "inputs": {
                "J": J,
                "n_per": n_per,
                "tau2_int_true": tau2_int,
                "tau2_slope_true": tau2_slope if slope else None,
                "rho_true": rho if slope else None,
                "sigma2_true": sigma2,
                "beta0_true": beta0,
                "beta1_true": beta1,
                "random_slope": slope
            },
            "ml": {
                "logLik": float(result_ml.llf),
                "aic": safe_num(getattr(result_ml, "aic", None)),
                "bic": safe_num(getattr(result_ml, "bic", None)),
                "var": extract_variances(result_ml, slope),
                "fixed": extract_fixed(result_ml)
            },
            "reml": {
                "logLik": float(result_reml.llf),
                "aic": safe_num(getattr(result_reml, "aic", None)),
                "bic": safe_num(getattr(result_reml, "bic", None)),
                "var": extract_variances(result_reml, slope),
                "fixed": extract_fixed(result_reml)
            }
        }

        return jsonify(sanitize(out))

    except Exception as e:
        print("❌ mlreml_quick error:", e)
        return jsonify({"error": str(e)}), 500

# ------------------------------------
# Serve frontend
# ------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


