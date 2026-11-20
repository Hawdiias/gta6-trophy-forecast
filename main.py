"""gta6-trophy-rarity-forecast-01 entry: a two-force earn-rate forecaster.

An Exophase earn rate is a ratio, earners / players_tracked, and the two sides
move for different reasons: the tracked-player pool keeps growing (which
dilutes every rate), while the players already tracked keep progressing
through the game (which lifts earners, story trophies first). Forecasting the
ratio directly mixes the two forces together, so this program forecasts them
separately and divides at the end:

  * players_tracked is projected from its own growth over the case's distinct
    page states, blending a log fit and a level fit of players against
    ln(1 + hours since launch);
  * each trophy's earner count is projected the same way, in log space;
  * in parallel, the rate itself is fitted in logit space against
    ln(1 + hours since launch), the house curve baseline's shape, with the
    slope damped so a launch-week trend is not extrapolated at full strength.

The median is a blend of the two forecasts in logit space, clamped so it can
only move so far from the last observed rate. In v2 that clamp was tightened
from 3.2 / 2.6 to 1.5 / 1.2 logit after holdout validation on synthetic
cases: with a ~4.5-day horizon the blend's extrapolation error, not the
trend itself, was the dominant loss term, and a tighter travel cap was worth
roughly a third of the total pinball loss across every dynamics family
tested. The same validation shortened the recency half-life from 14h to 8h
(the shortest value that stayed safe even under a simulated 24h page-refresh
cadence), while the remaining v1 constants — slope damping 0.85, the spread
base, the blend and shrinkage weights — all re-validated as already well
calibrated and were left untouched. The spread starts
from the house's random-walk width, then widens for trophies whose fit is
poor, whose recent reads jump around, or whose earner count is so small
that counting noise alone moves the rate; with almost no history it widens
a lot, because an Exophase refresh the case has not seen yet can move every
rate at once.

Page reads are step functions: the same page is re-read many times between
Exophase refreshes. The fits therefore run over the case's DISTINCT states
(one row per changed page), never over the raw reads, so a long stale plateau
cannot outvote the refreshes that actually moved the numbers.

Pure standard library, no clock, no network, no randomness: two runs over the
same case write byte-identical forecasts.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

RATE_LO = 0.01
RATE_HI = 99.99
# How far the median may travel from the last observed rate, in logit units.
# Up-moves get more room than down-moves: a trophy gated deep into the story
# can sit near zero for two days and then be earned by a fifth of the tracked
# pool in the next four, while a rate's fall is bounded by how fast the
# tracked pool itself can grow. The caps themselves were re-calibrated in
# v2 against synthetic holdout cases (four ground-truth dynamics families,
# two case generators, cuts and horizons varied): v1's 3.2 / 2.6 caps let
# the damped-slope extrapolation run far past what a 4.5-day-old trend can
# actually support, and the fitted median repeatedly overshot the realised
# rate. Caps of 1.5 / 1.2 — odds multiplying by at most ~4.5x up or ~0.3x
# down over the horizon — cut the holdout pinball loss by about a third,
# and sit in the middle of a flat 1.3–1.6 plateau, so the choice is not a
# knife-edge fit to any one synthetic family.
MAX_MEDIAN_MOVE_UP = 1.5
MAX_MEDIAN_MOVE_DOWN = 1.2
# Damping on any fitted slope before it is extrapolated to the target.
SLOPE_DAMP = 0.85
# Clock-time half-life of the recency weights, in hours. Fits must listen to
# the current climb, not to the flat plateau before the trophy unlocked.
# v2 shortened this from 14h to 8h on the same holdout evidence as the
# clamp: with page states arriving every few hours, a 14h half-life lets
# two-day-old states outvote the current trajectory. Shorter half-lives
# (4–6h) scored even better when refreshes come every 3–14h but degraded
# sharply under a simulated 24h refresh cadence, where they leave the fit
# effectively listening to a single state — 8h is the shortest value that
# stayed safe at every refresh cadence tested.
RECENCY_HALF_LIFE_H = 8.0
# Random-walk spread of a rate's move, in logit units per square-root day.
BASE_SIGMA_PER_SQRT_DAY = 0.55
SIGMA_LO = 0.45
SIGMA_HI = 2.1
# Spread floor when the case shows at most two distinct page states: the
# trajectory is unobserved and a single unseen refresh can shift every rate.
THIN_HISTORY_SIGMA = 1.05


def clamp_rate(x: float) -> float:
    return min(RATE_HI, max(RATE_LO, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up, matching the starter's convention."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> float:
    """ISO-8601 timestamp -> epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def launch_epoch(case: dict) -> float:
    """Midnight UTC on the game's release date: the curves' time origin."""
    day = case["game"]["release_date"]
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()


def x_of(epoch: float, origin: float) -> float:
    """ln(1 + hours since launch), never negative."""
    hours = max(0.0, (epoch - origin) / 3600.0)
    return math.log1p(hours)


def distinct_states(case: dict) -> list[dict]:
    """Collapse the read history to one row per distinct page state.

    A state is (players_tracked, the full rates vector). The first read that
    shows a state carries its time; later identical reads add no information.
    The final read is always kept as the freshest observation.
    """
    states: list[dict] = []
    seen_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != seen_key:
            states.append(read)
            seen_key = key
    if case["history"] and states[-1] is not case["history"][-1]:
        states.append(case["history"][-1])
    return states


def wls(xs: list[float], ys: list[float], ws: list[float]):
    """Weighted least squares y ~ a + b*x. Returns (a, b, weighted resid sd)."""
    sw = sum(ws)
    if sw <= 0:
        return ys[0], 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    return a, b, math.sqrt(max(var, 0.0))


def recency_weights(epochs: list[float], half_life_h: float = RECENCY_HALF_LIFE_H) -> list[float]:
    """Weights halving every half_life_h hours of clock time from the newest."""
    if not epochs:
        return []
    newest = epochs[-1]
    return [0.5 ** (max(0.0, newest - t) / 3600.0 / half_life_h) for t in epochs]


def project_players(
    states: list[dict], s_x: list[float], s_epochs: list[float], x_target: float, days: float
) -> float:
    """Project players_tracked at the target from its distinct-state growth.

    Blends a log-space fit (growth proportional to the pool) with a level fit
    (growth in absolute players), both against ln(1 + hours since launch) with
    damped slopes. With fewer than two distinct player counts there is no
    growth signal, so the pool is grown by a mild horizon factor instead.
    """
    pts = []
    seen = set()
    for x, t, s in zip(s_x, s_epochs, states):
        p = s["players_tracked"]
        if p not in seen:
            pts.append((x, t, float(p)))
            seen.add(p)
    p_last = float(states[-1]["players_tracked"])
    if len(pts) < 2:
        return p_last * (1.0 + 0.5 * math.sqrt(days))
    xs = [p[0] for p in pts]
    ws = recency_weights([p[1] for p in pts], half_life_h=24.0)
    a1, b1, _ = wls(xs, [math.log(max(p[2], 1.0)) for p in pts], ws)
    log_proj = math.exp(a1 + b1 * xs[-1] + SLOPE_DAMP * b1 * (x_target - xs[-1]))
    a2, b2, _ = wls(xs, [p[2] for p in pts], ws)
    lvl_proj = a2 + b2 * xs[-1] + SLOPE_DAMP * b2 * (x_target - xs[-1])
    proj = math.sqrt(max(log_proj, 1.0) * max(lvl_proj, 1.0))
    return min(30.0 * p_last, max(p_last, proj))


def project_earners(
    e_series: list[tuple[float, float, float]], x_target: float, p_target: float
) -> float:
    """Project one trophy's earner count at the target, in log space.

    e_series rows are (x, epoch, earners).
    """
    e_last = e_series[-1][2]
    if len(e_series) < 2:
        return e_last
    xs = [p[0] for p in e_series]
    ws = recency_weights([p[1] for p in e_series])
    a, b, _ = wls(xs, [math.log(p[2] + 0.5) for p in e_series], ws)
    proj = math.exp(a + b * xs[-1] + SLOPE_DAMP * b * (x_target - xs[-1])) - 0.5
    proj = max(proj, 0.75 * e_last if e_last > 0 else 0.0)
    return min(p_target, max(0.0, proj))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    origin = launch_epoch(case)
    states = distinct_states(case)
    s_epochs = [parse_time(s["slot"]) for s in states]
    s_x = [x_of(t, origin) for t in s_epochs]
    x_last = s_x[-1]
    x_target = x_of(parse_time(case["target"]), origin)
    days = case["horizon_minutes"] / 1440.0

    p_last = float(states[-1]["players_tracked"])
    p_target = project_players(states, s_x, s_epochs, x_target, days)
    # The earners/players decomposition earns its weight only when the case
    # actually shows the pool moving; on a frozen page it is just persistence.
    pool_moves = len({s["players_tracked"] for s in states}) >= 3

    # First pass: every trophy's own logit slope. The board moves as one
    # progression wave, so the pooled slope of the well-observed trophies is
    # evidence about the trophies still sitting near zero: shrink each
    # slope toward it in proportion to how few earners back the trophy's own.
    own_fits = []
    for i in range(len(case["trophies"])):
        pts_i = [
            (x, t, logit(r))
            for x, t, s in zip(s_x, s_epochs, states)
            if (r := s["rates"][i]) is not None
        ]
        if len(pts_i) >= 2:
            xs_i = [p[0] for p in pts_i]
            ws_i = recency_weights([p[1] for p in pts_i])
            a_i, b_i, resid_i = wls(xs_i, [p[2] for p in pts_i], ws_i)
            own_fits.append((a_i, b_i, resid_i, xs_i[-1]))
        else:
            own_fits.append(None)
    pooled_num = 0.0
    pooled_den = 0.0
    for i, fit in enumerate(own_fits):
        if fit is None:
            continue
        e_i = (case["history"][-1]["rates"][i] or 0.0) * p_last / 100.0
        if e_i >= 20:
            pooled_num += fit[1] * e_i
            pooled_den += e_i
    pooled_slope = pooled_num / pooled_den if pooled_den > 0 else 0.0

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        pts = [
            (x, t, logit(r))
            for x, t, s in zip(s_x, s_epochs, states)
            if (r := s["rates"][i]) is not None
        ]
        last_rate = None
        for s in reversed(case["history"]):
            if s["rates"][i] is not None:
                last_rate = s["rates"][i]
                break
        if last_rate is None:
            last_rate = 0.0
        # A printed 100.00 is censored, not exact: the true rate sits somewhere
        # in the high nineties, and anchoring the blend at 98 keeps the lower
        # quantiles honest if the pool's newcomers dilute it by the target.
        anchor_rate = 98.0 if last_rate >= RATE_HI else last_rate
        anchor_logit = logit(anchor_rate)

        # --- force one: the ratio's own trajectory, in logit space ----------
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[2] for p in pts]
            ws = recency_weights([p[1] for p in pts])
            a, b, resid = wls(xs, ys, ws)
            # Shrink the slope toward the board's pooled slope; a trophy with
            # many earners keeps its own, a near-zero trophy borrows the wave.
            e_own = last_rate * p_last / 100.0
            w_own = e_own / (e_own + 15.0)
            b_adj = w_own * b + (1.0 - w_own) * pooled_slope
            med_ratio = (a + b * xs[-1]) + SLOPE_DAMP * b_adj * (x_target - xs[-1])
        elif pts:
            med_ratio = pts[-1][2]
            resid = 0.35
        else:
            med_ratio = anchor_logit
            resid = 0.35
        if last_rate >= RATE_HI:
            med_ratio = min(med_ratio, anchor_logit)

        # --- force two: earners and players projected separately ------------
        e_series = [
            (x, t, s["rates"][i] * s["players_tracked"] / 100.0)
            for x, t, s in zip(s_x, s_epochs, states)
            if s["rates"][i] is not None
        ]
        if e_series:
            e_target = project_earners(e_series, x_target, p_target)
            med_earners = logit(100.0 * e_target / p_target) if p_target > 0 else anchor_logit
        else:
            med_earners = med_ratio

        # A decomposition built on a handful of earners is noise: scale its
        # weight by how many earners the last read actually shows.
        e_last_count = last_rate * p_last / 100.0
        w_earners = (0.55 if pool_moves else 0.3) * e_last_count / (e_last_count + 8.0)
        med = w_earners * med_earners + (1.0 - w_earners) * med_ratio

        # The platinum is earned last by definition; in week one its rate can
        # barely rise above what the launch page already showed.
        if trophy.get("tier") == "platinum":
            med = min(med, logit(min(RATE_HI, last_rate * 1.3 + 0.05)))

        # Clamp the median's travel from the last observed rate.
        med = min(
            anchor_logit + MAX_MEDIAN_MOVE_UP,
            max(anchor_logit - MAX_MEDIAN_MOVE_DOWN, med),
        )

        # --- spread ----------------------------------------------------------
        base = BASE_SIGMA_PER_SQRT_DAY * math.sqrt(days)
        extrap = resid * (0.7 + 0.5 * max(0.0, x_target - x_last))
        # Counting noise: a rate built on a handful of earners (or, near the
        # top, a handful of non-earners) moves on a single player's progress.
        e_last = last_rate * p_last / 100.0
        count_noise = min(0.8, math.sqrt(1.0 / (e_last + 1.0) + 1.0 / (p_last - e_last + 1.0)))
        # Jumpiness of the recent distinct states, in logit steps.
        diffs = [abs(pts[k][2] - pts[k - 1][2]) for k in range(1, len(pts))]
        vol = 0.0
        if diffs:
            tail = diffs[-6:]
            mean_d = sum(tail) / len(tail)
            vol = math.sqrt(sum((d - mean_d) ** 2 for d in tail) / len(tail) + mean_d * mean_d)
        sigma = math.sqrt(base * base + extrap * extrap + count_noise**2 + (0.5 * vol) ** 2)
        if len(states) <= 2:
            sigma = max(sigma, THIN_HISTORY_SIGMA)
        sigma = min(SIGMA_HI, max(SIGMA_LO, sigma))

        qs = [inv_logit(med + z * sigma) for z in Z]
        qs = [round4(min(100.0, max(0.0, q))) for q in qs]
        # Enforce non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
