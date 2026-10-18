"""Leak-free feature engineering for hourly -> quarter-hourly price reconstruction.

The same entry point, :func:`build_features`, is used for training and for the
frozen inference pipeline:

* training:  ``build_features(derive_hourly(train15), train15, target_df=train15)``
* inference: ``build_features(test_hourly, history_15min)``

Availability rule enforced by construction:

* Everything derived from the hourly curve uses only the 24 hourly prices of the
  evaluated date D (plus H(23), H(24) of D-1, read from the quarter-hour history).
* Everything derived from quarter-hour prices uses history dates strictly before
  D, addressed by calendar day (``D - k``). A training history that contains D
  itself, or later dates, therefore yields exactly the same features as the
  jury's ``history_15min.csv``.
* The quarter-hour prices of D are only used to attach the training target.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DATE, ZONE, HOUR, QUARTER = "delivery_date", "market_id", "hour_index", "quarter_index"
PRICE, HOURLY, TARGET = "price_eur_mwh", "hourly_price_eur_mwh", "target_spread"
PRED = "predicted_price_eur_mwh"
HOUR_KEYS = [DATE, ZONE, HOUR]
QH_KEYS = [DATE, ZONE, HOUR, QUARTER]

ZONES = ["ZONE_0", "ZONE_1", "ZONE_2", "ZONE_3"]
HOURS = np.arange(1, 25)
QUARTERS = np.arange(1, 5)
MARKET_TZ = "Europe/Berlin"  # CET/CEST

SPLITS = {
    "train": ("2025-10-01", "2026-06-30"),
    "validation": ("2026-08-01", "2026-09-30"),
    "test_diagnostic": ("2026-07-01", "2026-07-31"),
}

LAG_DAYS = (1, 2, 3, 7, 14)
ROLL_WINDOWS = (3, 7, 14)
WEEKDAY_OCCURRENCES = 4
SLOPE_WINDOW = 28
SLOPE_MIN_OBS = 7
SLOPE_RIDGE = 1.0  # (EUR/MWh)^2 added to var(slope); shrinks beta towards 0 on flat windows
EPS = 1e-5
MAX_LOOKBACK_DAYS = max(max(LAG_DAYS), 7 * WEEKDAY_OCCURRENCES, SLOPE_WINDOW, max(ROLL_WINDOWS))

CATEGORIES = {
    ZONE: ZONES,
    HOUR: list(range(1, 25)),
    QUARTER: list(range(1, 5)),
    "day_of_week": list(range(7)),
    "month": list(range(1, 13)),
}

FEATURE_GROUPS = {
    "categorical": [ZONE, HOUR, QUARTER, "day_of_week", "month"],
    "calendar": [
        "is_weekend", "day_of_year", "week_of_year", "quarter_of_day",
        "is_daytime", "is_nighttime", "is_morning_ramp", "is_solar_peak", "is_evening_peak",
        "hour_sin", "hour_cos", "quarter_sin", "quarter_cos",
        "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    ],
    "hourly_geometry": [
        HOURLY, "diff_prev_1h", "diff_prev_2h", "diff_next_1h", "diff_next_2h",
        "slope_centered", "curvature_1h", "is_first_hour", "is_last_hour", "prev_day_tail_available",
    ],
    "quarter_interactions": ["q_centered", "linear_interp_proxy", "q_x_diff_prev_1h", "q_x_diff_next_1h"],
    "daily_context": [
        "daily_mean", "daily_std", "daily_min", "daily_max", "daily_skew", "daily_range",
        "h_minus_daily_mean", "h_minmax_pos", "h_zscore", "is_nonpositive_price",
    ],
    "cross_zone": [
        "sys_mean_hourly", "sys_std_hourly", "sys_mean_slope", "sys_std_slope",
        "h_spread_vs_system", "slope_spread_vs_system",
    ],
    "history_lags": [f"spread_lag_{k}d" for k in LAG_DAYS],
    "history_rolling": [
        f"spread_roll{w}d_{stat}" for w in ROLL_WINDOWS for stat in ("mean", "median", "std")
    ] + [f"spread_dow{WEEKDAY_OCCURRENCES}_mean"],
    "slope_response": [f"slope_beta_{SLOPE_WINDOW}d", f"slope_alpha_{SLOPE_WINDOW}d", "slope_response_pred"],
    "boundary": ["prev_day_last_qh_price", "prev_day_last_qh_spread", "prev_day_last_qh_gap"],
    "history_meta": ["history_days_available"],
}
FEATURE_COLS = [c for cols in FEATURE_GROUPS.values() for c in cols]
CATEGORICAL_COLS = FEATURE_GROUPS["categorical"]
NON_FEATURE_COLS = ["timestamp", "unix_timestamp", "split", PRICE, TARGET, DATE]
INT8_COLS = [
    "is_weekend", "is_daytime", "is_nighttime", "is_morning_ramp", "is_solar_peak", "is_evening_peak",
    "is_first_hour", "is_last_hour", "prev_day_tail_available", "is_nonpositive_price",
]
INT16_COLS = ["day_of_year", "week_of_year", "quarter_of_day"]
FLOAT64_COLS = [HOURLY, PRICE, TARGET]


# ---------------------------------------------------------------------------
# Stage 1: schema, target and baseline
# ---------------------------------------------------------------------------

def _normalise_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE] = pd.to_datetime(out[DATE]).dt.normalize()
    out[ZONE] = out[ZONE].astype(str)
    out[HOUR] = out[HOUR].astype("int64")
    if QUARTER in out:
        out[QUARTER] = out[QUARTER].astype("int64")
    return out


def _missing_calendar_dates(dates: pd.Series) -> list[str]:
    unique = pd.DatetimeIndex(dates.unique()).sort_values()
    if unique.empty:
        return []
    full = pd.date_range(unique[0], unique[-1], freq="D")
    return [d.strftime("%Y-%m-%d") for d in full.difference(unique)]


def validate_15min(df: pd.DataFrame, require_complete_days: bool = True) -> dict:
    """Schema checks for train_prices_15min.csv / history_15min.csv."""
    if df.duplicated(QH_KEYS).any():
        raise ValueError("duplicate (delivery_date, market_id, hour_index, quarter_index) keys")
    if not df[HOUR].between(1, 24).all() or not df[QUARTER].between(1, 4).all():
        raise ValueError("hour_index must be 1..24 and quarter_index 1..4")
    if not np.isfinite(df[PRICE].to_numpy(float)).all():
        raise ValueError("non-finite price_eur_mwh")
    unknown = set(df[ZONE]) - set(ZONES)
    if unknown:
        raise ValueError(f"unknown market_id values: {sorted(unknown)}")
    per_day = df.groupby([DATE, ZONE]).size()
    incomplete = per_day[per_day != 96]
    if require_complete_days and len(incomplete):
        raise ValueError(f"{len(incomplete)} (date, zone) groups do not have 96 quarter-hours")
    return {
        "rows": int(len(df)),
        "zones": sorted(df[ZONE].unique().tolist()),
        "n_dates": int(df[DATE].nunique()),
        "first_date": df[DATE].min().strftime("%Y-%m-%d") if len(df) else None,
        "last_date": df[DATE].max().strftime("%Y-%m-%d") if len(df) else None,
        "rows_per_zone": {z: int(n) for z, n in df.groupby(ZONE).size().items()},
        "incomplete_days": int(len(incomplete)),
        "missing_calendar_dates": _missing_calendar_dates(df[DATE]),
    }


def validate_hourly(df: pd.DataFrame) -> None:
    """Schema checks for test_hourly.csv-shaped input (24 hourly prices per date and zone)."""
    if df.duplicated(HOUR_KEYS).any():
        raise ValueError("duplicate (delivery_date, market_id, hour_index) keys")
    if not np.isfinite(df[HOURLY].to_numpy(float)).all():
        raise ValueError("non-finite hourly_price_eur_mwh")
    unknown = set(df[ZONE]) - set(ZONES)
    if unknown:
        raise ValueError(f"unknown market_id values: {sorted(unknown)}")
    per_day = df.groupby([DATE, ZONE])[HOUR].agg(["size", "min", "max"])
    if not ((per_day["size"] == 24) & (per_day["min"] == 1) & (per_day["max"] == 24)).all():
        raise ValueError("every (delivery_date, market_id) needs hour_index 1..24")


def load_15min(path: str | Path) -> pd.DataFrame:
    df = _normalise_keys(pd.read_csv(path, dtype={ZONE: str}))
    validate_15min(df)
    return df.sort_values(QH_KEYS, ignore_index=True)


def load_hourly(path: str | Path) -> pd.DataFrame:
    df = _normalise_keys(pd.read_csv(path, dtype={ZONE: str}))
    validate_hourly(df)
    return df.sort_values(HOUR_KEYS, ignore_index=True)


def derive_hourly(df15: pd.DataFrame) -> pd.DataFrame:
    """Reference hourly price = arithmetic mean of the hour's four quarter-hour prices."""
    counts = df15.groupby(HOUR_KEYS)[QUARTER].nunique()
    if (counts != 4).any():
        raise ValueError(f"{int((counts != 4).sum())} hours do not have four quarter-hours")
    hourly = df15.groupby(HOUR_KEYS, as_index=False, sort=True)[PRICE].mean()
    return hourly.rename(columns={PRICE: HOURLY})


def add_target(df15: pd.DataFrame, hourly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the hourly reference and the intra-hour residual target_spread = price - hourly."""
    hourly = derive_hourly(df15) if hourly is None else hourly
    out = df15.merge(hourly[HOUR_KEYS + [HOURLY]], on=HOUR_KEYS, how="left", validate="many_to_one")
    out[TARGET] = out[PRICE] - out[HOURLY]
    return out


def reconstruct_prices(df: pd.DataFrame, y_hat) -> pd.Series:
    """pred = hourly + (y_hat - mean_q(y_hat)): the four predictions average exactly to the hourly price."""
    y = pd.Series(np.asarray(y_hat, dtype="float64"), index=df.index)
    keys = [df[DATE], df[ZONE].astype(str), df[HOUR].astype("int64")]
    centred = y - y.groupby(keys).transform("mean")
    return df[HOURLY].astype("float64") + centred


def check_hourly_constraint(
    pred: pd.DataFrame, hourly: pd.DataFrame, pred_col: str = PRED, tol: float = 1e-4
) -> dict:
    """Organiser-style eligibility check: four finite predictions per hour whose mean matches H(h)."""
    p = _normalise_keys(pred[QH_KEYS + [pred_col]])
    h = _normalise_keys(hourly[HOUR_KEYS + [HOURLY]])
    quarter_sets = p.groupby(HOUR_KEYS)[QUARTER].agg(["size", "nunique", "min", "max"])
    bad_sets = ~((quarter_sets["size"] == 4) & (quarter_sets["nunique"] == 4)
                 & (quarter_sets["min"] == 1) & (quarter_sets["max"] == 4))
    means = p.groupby(HOUR_KEYS)[pred_col].mean().rename("pred_mean").reset_index()
    m = h.merge(means, on=HOUR_KEYS, how="left")
    dev = (m["pred_mean"] - m[HOURLY]).abs()
    result = {
        "n_hours": int(len(h)),
        "missing_hours": int(m["pred_mean"].isna().sum()),
        "duplicate_rows": int(p.duplicated(QH_KEYS).sum()),
        "all_finite": bool(np.isfinite(p[pred_col].to_numpy(float)).all()),
        "bad_quarter_sets": int(bad_sets.sum()),
        "max_abs_deviation": float(dev.max()),
        "n_out_of_tolerance": int((dev > tol).sum()),
    }
    result["compliant"] = (
        result["missing_hours"] == 0 and result["duplicate_rows"] == 0 and result["all_finite"]
        and result["bad_quarter_sets"] == 0 and result["n_out_of_tolerance"] == 0
    )
    return result


def assign_split(dates: pd.Series, splits: dict = SPLITS) -> pd.Series:
    out = pd.Series("outside", index=dates.index, dtype=object)
    for name, (start, end) in splits.items():
        out[dates.between(pd.Timestamp(start), pd.Timestamp(end))] = name
    return out.astype(pd.CategoricalDtype(list(splits) + ["outside"]))


def official_mae(df: pd.DataFrame, pred_col: str, obs_col: str = PRICE) -> tuple[pd.Series, float]:
    """MAE per (zone, day), averaged over days per zone, then equally over zones."""
    err = (df[pred_col] - df[obs_col]).abs()
    per_day = err.groupby([df[ZONE].astype(str), df[DATE]]).mean()
    per_zone = per_day.groupby(level=0).mean()
    return per_zone, float(per_zone.mean())


def mae_table(df: pd.DataFrame, pred_col: str, split_col: str = "split") -> pd.DataFrame:
    rows = {}
    for split in df[split_col].astype(str).unique():
        part = df[df[split_col].astype(str) == split]
        per_zone, mean = official_mae(part, pred_col)
        rows[split] = {**per_zone.to_dict(), "equal_zone_mean": mean}
    order = [s for s in list(SPLITS) + ["outside"] if s in rows]
    return pd.DataFrame(rows).T.loc[order]


# ---------------------------------------------------------------------------
# Stage 2: hourly-curve features for day D
# ---------------------------------------------------------------------------

def _hourly_wide(hourly: pd.DataFrame) -> pd.DataFrame:
    """(delivery_date, market_id) x hour_index matrix of hourly prices."""
    w = hourly.pivot(index=[DATE, ZONE], columns=HOUR, values=HOURLY)
    return w.reindex(columns=HOURS).sort_index()


def _previous_day_tail(index: pd.MultiIndex, history_hourly: pd.DataFrame | None) -> np.ndarray:
    """H(23), H(24) of calendar day D-1 per (D, zone), read from the history; NaN when absent."""
    if history_hourly is None or history_hourly.empty:
        return np.full((len(index), 2), np.nan)
    tail = _hourly_wide(history_hourly)[[23, 24]]
    prev = pd.MultiIndex.from_arrays(
        [index.get_level_values(DATE) - pd.Timedelta(days=1), index.get_level_values(ZONE)],
        names=[DATE, ZONE],
    )
    return tail.reindex(prev).to_numpy(dtype="float64")


def hourly_curve_geometry(df_hourly: pd.DataFrame, history_hourly: pd.DataFrame | None) -> pd.DataFrame:
    """Slopes, curvature and daily context per (date, zone, hour).

    The curve is extended as [H_{D-1}(23), H_{D-1}(24), H_D(1..24), NaN, NaN]: D-1 comes from
    the quarter-hour history (identical in training and test), D+1 is never available.
    ``slope_centered`` is the central difference where both neighbours exist and otherwise the
    one-sided difference, so it is defined at h=24 and at h=1 when D-1 is missing.
    """
    w = _hourly_wide(df_hourly)
    if w.isna().any().any():
        raise ValueError("each (date, zone) needs 24 hourly prices")
    n = len(w)
    tail = _previous_day_tail(w.index, history_hourly)
    H = w.to_numpy(dtype="float64")
    ext = np.concatenate([tail, H, np.full((n, 2), np.nan)], axis=1)
    h_m2, h_m1, h_p1, h_p2 = ext[:, 0:24], ext[:, 1:25], ext[:, 3:27], ext[:, 4:28]

    diff_prev_1h = H - h_m1
    diff_next_1h = h_p1 - H
    central = 0.5 * (h_p1 - h_m1)
    slope = np.where(np.isnan(central), np.where(np.isnan(diff_next_1h), diff_prev_1h, diff_next_1h), central)

    d_mean = H.mean(axis=1, keepdims=True)
    d_std = H.std(axis=1, ddof=0, keepdims=True)
    d_min = H.min(axis=1, keepdims=True)
    d_max = H.max(axis=1, keepdims=True)
    d_skew = w.skew(axis=1).to_numpy(dtype="float64")[:, None]
    ones = np.ones((1, 24))

    out = pd.DataFrame({
        DATE: np.repeat(w.index.get_level_values(DATE), 24),
        ZONE: np.repeat(w.index.get_level_values(ZONE), 24),
        HOUR: np.tile(HOURS, n),
        HOURLY: H.ravel(),
        "diff_prev_1h": diff_prev_1h.ravel(),
        "diff_prev_2h": (H - h_m2).ravel(),
        "diff_next_1h": diff_next_1h.ravel(),
        "diff_next_2h": (h_p2 - H).ravel(),
        "slope_centered": slope.ravel(),
        "curvature_1h": (h_p1 - 2.0 * H + h_m1).ravel(),
        "prev_day_tail_available": np.repeat(~np.isnan(tail[:, 1]), 24),
        "daily_mean": (d_mean * ones).ravel(),
        "daily_std": (d_std * ones).ravel(),
        "daily_min": (d_min * ones).ravel(),
        "daily_max": (d_max * ones).ravel(),
        "daily_skew": (d_skew * ones).ravel(),
        "daily_range": ((d_max - d_min) * ones).ravel(),
        "h_minus_daily_mean": (H - d_mean).ravel(),
        "h_minmax_pos": ((H - d_min) / (d_max - d_min + EPS)).ravel(),
        "h_zscore": ((H - d_mean) / (d_std + EPS)).ravel(),
        "is_nonpositive_price": (H <= 0).ravel(),
    })
    out["is_first_hour"] = out[HOUR] == 1
    out["is_last_hour"] = out[HOUR] == 24
    return out


def add_cross_zone_features(geo: pd.DataFrame) -> pd.DataFrame:
    """Simultaneous system context across the zones present at (date, hour) on day D."""
    g = geo.groupby([DATE, HOUR], sort=False)
    stats = pd.DataFrame({
        "sys_mean_hourly": g[HOURLY].mean(),
        "sys_std_hourly": g[HOURLY].std(ddof=0),
        "sys_mean_slope": g["slope_centered"].mean(),
        "sys_std_slope": g["slope_centered"].std(ddof=0),
    }).reset_index()
    out = geo.merge(stats, on=[DATE, HOUR], how="left", validate="many_to_one")
    out["h_spread_vs_system"] = out[HOURLY] - out["sys_mean_hourly"]
    out["slope_spread_vs_system"] = out["slope_centered"] - out["sys_mean_slope"]
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Delivery-time calendar on CET/CEST market time; hour_index 1 = 00:00-01:00 local."""
    out = df.copy()
    d, h, q = out[DATE], out[HOUR], out[QUARTER]
    out["timestamp"] = d + pd.to_timedelta((h - 1) * 60 + (q - 1) * 15, unit="min")
    utc = out["timestamp"].dt.tz_localize(MARKET_TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    out["unix_timestamp"] = (
        (utc.dt.tz_localize(None) - pd.Timestamp("1970-01-01")).dt.total_seconds().round().astype("Int64")
    )
    dow = d.dt.dayofweek
    doy = d.dt.dayofyear
    out["day_of_week"] = dow
    out["is_weekend"] = dow >= 5
    out["day_of_year"] = doy
    out["month"] = d.dt.month
    out["week_of_year"] = d.dt.isocalendar().week.astype("int64").to_numpy()
    out["quarter_of_day"] = (h - 1) * 4 + q
    out["is_daytime"] = h.between(7, 20)
    out["is_nighttime"] = ~out["is_daytime"]
    out["is_morning_ramp"] = h.between(6, 10)
    out["is_solar_peak"] = h.between(11, 15)
    out["is_evening_peak"] = h.between(17, 21)
    for name, value, period in (
        ("hour", h - 1, 24.0), ("quarter", q - 1, 4.0), ("dow", dow, 7.0), ("doy", doy - 1, 365.25)
    ):
        angle = 2.0 * np.pi * value / period
        out[f"{name}_sin"] = np.sin(angle)
        out[f"{name}_cos"] = np.cos(angle)
    return out


def add_quarter_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """A linear ramp with slope s per hour puts quarter q at s * (q - 2.5) / 4 from the hourly mean."""
    out = df.copy()
    out["q_centered"] = out[QUARTER] - 2.5
    out["linear_interp_proxy"] = out["q_centered"] / 4.0 * out["slope_centered"]
    out["q_x_diff_prev_1h"] = out["q_centered"] * out["diff_prev_1h"]
    out["q_x_diff_next_1h"] = out["q_centered"] * out["diff_next_1h"]
    return out


# ---------------------------------------------------------------------------
# Stage 3: historical quarter-hour features (calendar days strictly before D)
# ---------------------------------------------------------------------------

def history_panel(df_history_15min: pd.DataFrame) -> pd.DataFrame:
    """Per history day: price, intra-hour spread and that day's own hourly slope_centered.

    Incomplete (date, zone) days are dropped because their hourly curve is undefined.
    """
    hist = _normalise_keys(df_history_15min[QH_KEYS + [PRICE]])
    validate_15min(hist, require_complete_days=False)
    complete = hist.groupby([DATE, ZONE])[PRICE].transform("size") == 96
    if (~complete).any():
        warnings.warn(f"dropping {int((~complete).sum())} history rows from incomplete days")
    hist = hist[complete].sort_values(QH_KEYS, ignore_index=True)
    if hist.empty:
        return hist.assign(**{HOURLY: [], TARGET: [], "slope_centered": []})
    hourly = derive_hourly(hist)
    hist = add_target(hist, hourly)
    geo = hourly_curve_geometry(hourly, hourly)[HOUR_KEYS + ["slope_centered"]]
    return hist.merge(geo, on=HOUR_KEYS, how="left", validate="many_to_one")


def _shift_rows(a: np.ndarray, k: int) -> np.ndarray:
    """Row t of the result holds row t-k of ``a`` (one row = one calendar day)."""
    out = np.full_like(a, np.nan)
    out[k:] = a[: a.shape[0] - k]
    return out


def _lag_stack(a: np.ndarray, lags) -> np.ndarray:
    return np.stack([_shift_rows(a, k) for k in lags])


def _nan_stats(stack: np.ndarray) -> dict[str, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n = np.sum(~np.isnan(stack), axis=0)
        std = np.nanstd(stack, axis=0, ddof=1)
        return {
            "mean": np.nanmean(stack, axis=0),
            "median": np.nanmedian(stack, axis=0),
            "std": np.where(n >= 2, std, np.nan),
        }


def _slope_response(spread: np.ndarray, slope: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ridge-shrunk OLS of spread on slope_centered over the trailing SLOPE_WINDOW calendar days."""
    lags = range(1, SLOPE_WINDOW + 1)
    y, x = _lag_stack(spread, lags), _lag_stack(slope, lags)
    valid = ~np.isnan(y) & ~np.isnan(x)
    n = valid.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x0, y0 = np.where(valid, x, 0.0), np.where(valid, y, 0.0)
        x_mean, y_mean = x0.sum(axis=0) / n, y0.sum(axis=0) / n
        dx = np.where(valid, x - x_mean, 0.0)
        dy = np.where(valid, y - y_mean, 0.0)
        cov = (dx * dy).sum(axis=0) / n
        var = (dx * dx).sum(axis=0) / n
        beta = cov / (var + SLOPE_RIDGE)
        alpha = y_mean - beta * x_mean
    enough = n >= SLOPE_MIN_OBS
    return np.where(enough, beta, np.nan), np.where(enough, alpha, np.nan)


def historical_features(rows: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """History-derived features for target rows (DATE, ZONE, HOUR, QUARTER, slope_centered, HOURLY).

    All statistics are computed on a calendar-day grid and read through explicit day shifts
    (k >= 1), so the value for date D only ever touches history dates D-k < D. Missing dates
    (the DST gaps, or gaps in the jury history) stay NaN; nothing is shifted to the previous
    available row.
    """
    target_dates = pd.DatetimeIndex(rows[DATE].unique()).sort_values()
    grid = pd.date_range(target_dates[0] - pd.Timedelta(days=MAX_LOOKBACK_DAYS), target_dates[-1], freq="D")
    cols = pd.MultiIndex.from_product([ZONES, HOURS, QUARTERS], names=[ZONE, HOUR, QUARTER])

    def grid_matrix(value: str) -> np.ndarray:
        if panel.empty:
            return np.full((len(grid), len(cols)), np.nan)
        wide = panel.pivot(index=DATE, columns=[ZONE, HOUR, QUARTER], values=value)
        return wide.reindex(index=grid, columns=cols).to_numpy(dtype="float64")

    spread, price, slope = grid_matrix(TARGET), grid_matrix(PRICE), grid_matrix("slope_centered")

    feats: dict[str, np.ndarray] = {}
    for k in LAG_DAYS:
        feats[f"spread_lag_{k}d"] = _shift_rows(spread, k)
    for w in ROLL_WINDOWS:
        for stat, value in _nan_stats(_lag_stack(spread, range(1, w + 1))).items():
            feats[f"spread_roll{w}d_{stat}"] = value
    weekday_lags = [7 * i for i in range(1, WEEKDAY_OCCURRENCES + 1)]
    feats[f"spread_dow{WEEKDAY_OCCURRENCES}_mean"] = _nan_stats(_lag_stack(spread, weekday_lags))["mean"]
    beta, alpha = _slope_response(spread, slope)
    feats[f"slope_beta_{SLOPE_WINDOW}d"] = beta
    feats[f"slope_alpha_{SLOPE_WINDOW}d"] = alpha

    row_pos = grid.get_indexer(pd.DatetimeIndex(rows[DATE]))
    col_pos = cols.get_indexer(pd.MultiIndex.from_arrays(
        [rows[ZONE].astype(str), rows[HOUR].astype("int64"), rows[QUARTER].astype("int64")]
    ))
    if (row_pos < 0).any() or (col_pos < 0).any():
        raise ValueError("target rows fall outside the history grid")
    out = pd.DataFrame({name: m[row_pos, col_pos] for name, m in feats.items()}, index=rows.index)
    out["slope_response_pred"] = (
        out[f"slope_alpha_{SLOPE_WINDOW}d"] + out[f"slope_beta_{SLOPE_WINDOW}d"] * rows["slope_centered"]
    )

    last_col = cols.get_indexer(pd.MultiIndex.from_arrays(
        [rows[ZONE].astype(str), np.full(len(rows), 24), np.full(len(rows), 4)]
    ))
    first_hour = rows[HOUR].to_numpy() == 1
    prev_price = _shift_rows(price, 1)[row_pos, last_col]
    prev_spread = _shift_rows(spread, 1)[row_pos, last_col]
    out["prev_day_last_qh_price"] = np.where(first_hour, prev_price, np.nan)
    out["prev_day_last_qh_spread"] = np.where(first_hour, prev_spread, np.nan)
    out["prev_day_last_qh_gap"] = out["prev_day_last_qh_price"] - rows[HOURLY].to_numpy()

    hist_dates = np.sort(panel[DATE].unique()) if not panel.empty else np.array([], dtype="datetime64[ns]")
    out["history_days_available"] = np.searchsorted(hist_dates, rows[DATE].to_numpy(), side="left")
    return out


# ---------------------------------------------------------------------------
# Stage 4: assembly, dtypes, metadata
# ---------------------------------------------------------------------------

def apply_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed category lists, so category codes are identical in training and test."""
    out = df.copy()
    for col, cats in CATEGORIES.items():
        if col in out:
            values = out[col].astype(str) if col == ZONE else out[col].astype("int64")
            out[col] = pd.Categorical(values, categories=cats)
    return out


def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = apply_categories(df)
    for col in INT8_COLS:
        out[col] = out[col].astype("int8")
    for col in INT16_COLS:
        out[col] = out[col].astype("int16")
    out["history_days_available"] = out["history_days_available"].astype("int32")
    for col in out.columns:
        if col in FLOAT64_COLS:
            out[col] = out[col].astype("float64")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].astype("float32")
    return out


def build_features(
    df_hourly: pd.DataFrame,
    df_history_15min: pd.DataFrame | None,
    target_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (delivery_date, market_id, hour_index, quarter_index) of ``df_hourly``.

    df_hourly:        test_hourly.csv schema (delivery_date, market_id, hour_index, hourly_price_eur_mwh)
    df_history_15min: history_15min.csv schema; only dates strictly before each target date are used
    target_df:        training only; quarter-hour prices used to attach price_eur_mwh and target_spread
    """
    hourly = _normalise_keys(df_hourly[HOUR_KEYS + [HOURLY]])
    validate_hourly(hourly)
    if df_history_15min is None:
        df_history_15min = pd.DataFrame(columns=QH_KEYS + [PRICE])
    panel = history_panel(df_history_15min)
    history_hourly = (
        panel.drop_duplicates(HOUR_KEYS)[HOUR_KEYS + [HOURLY]] if not panel.empty else None
    )

    geo = add_cross_zone_features(hourly_curve_geometry(hourly, history_hourly))
    rows = geo.merge(pd.DataFrame({QUARTER: QUARTERS}), how="cross")
    rows = rows.sort_values(QH_KEYS, ignore_index=True)
    rows = add_quarter_interactions(add_calendar_features(rows))
    rows = pd.concat([rows, historical_features(rows, panel)], axis=1)

    if target_df is not None:
        obs = _normalise_keys(target_df[QH_KEYS + [PRICE]])
        rows = rows.merge(obs, on=QH_KEYS, how="left", validate="one_to_one")
        rows[TARGET] = rows[PRICE] - rows[HOURLY]
        hour_mean = rows.groupby(HOUR_KEYS)[TARGET].transform("mean").abs().max()
        if not hour_mean < 1e-6:
            raise ValueError(f"target hourly means differ from hourly input (max |mean spread| = {hour_mean})")
    rows["split"] = assign_split(rows[DATE])

    ordered = [DATE, "timestamp", "unix_timestamp", "split"] + FEATURE_COLS
    ordered += [c for c in (PRICE, TARGET) if c in rows]
    return cast_dtypes(rows[ordered])


def feature_metadata(df: pd.DataFrame, extra: dict | None = None) -> dict:
    forbidden = set(NON_FEATURE_COLS) & set(FEATURE_COLS)
    if forbidden:
        raise ValueError(f"non-feature columns listed as features: {sorted(forbidden)}")
    meta = {
        "target_col": TARGET,
        "observed_price_col": PRICE,
        "hourly_price_col": HOURLY,
        "id_cols": QH_KEYS,
        "sort_cols": ["unix_timestamp", ZONE],
        "split_col": "split",
        "excluded_from_features": NON_FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "categories": CATEGORIES,
        "feature_cols": FEATURE_COLS,
        "feature_groups": FEATURE_GROUPS,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "splits": SPLITS,
        "rows": int(len(df)),
        "rows_per_split": {str(k): int(v) for k, v in df["split"].value_counts(sort=False).items()},
        "params": {
            "lag_days": list(LAG_DAYS),
            "roll_windows_days": list(ROLL_WINDOWS),
            "weekday_occurrences": WEEKDAY_OCCURRENCES,
            "slope_window_days": SLOPE_WINDOW,
            "slope_min_obs": SLOPE_MIN_OBS,
            "slope_ridge": SLOPE_RIDGE,
            "market_tz": MARKET_TZ,
        },
        "reconstruction": "predicted_price = hourly_price_eur_mwh + (y_hat - mean over the hour's 4 quarters of y_hat)",
    }
    if extra:
        meta.update(extra)
    return meta


def save_enriched(df: pd.DataFrame, meta: dict, out_dir: str | Path, name: str = "train_enriched") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / f"{name}.parquet"
    meta_path = out_dir / "feature_metadata.json"
    df.to_parquet(parquet, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"parquet": str(parquet), "metadata": str(meta_path)}


def load_enriched(path: str | Path) -> pd.DataFrame:
    """Read the enriched Parquet; integer-coded categoricals are restored from CATEGORIES."""
    return apply_categories(pd.read_parquet(path))


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------

def _compare_frames(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> dict:
    a = a.sort_values(QH_KEYS, ignore_index=True)
    b = b.sort_values(QH_KEYS, ignore_index=True)
    if len(a) != len(b):
        return {"rows_match": False, "max_abs_diff": np.inf, "nan_mismatch": -1}
    max_diff, nan_mismatch = 0.0, 0
    for c in cols:
        if isinstance(a[c].dtype, pd.CategoricalDtype):
            nan_mismatch += int((a[c].astype(str) != b[c].astype(str)).sum())
            continue
        x, y = a[c].to_numpy(dtype="float64"), b[c].to_numpy(dtype="float64")
        nan_mismatch += int((np.isnan(x) != np.isnan(y)).sum())
        both = ~np.isnan(x) & ~np.isnan(y)
        if both.any():
            max_diff = max(max_diff, float(np.abs(x[both] - y[both]).max()))
    return {"rows_match": True, "max_abs_diff": max_diff, "nan_mismatch": nan_mismatch}


def truncation_invariance_check(df15: pd.DataFrame, full: pd.DataFrame, dates) -> pd.DataFrame:
    """Rebuild each date D exactly as the jury would supply it (hourly curve of D + history < D)."""
    records = []
    for d in pd.to_datetime(list(dates)):
        day = df15[df15[DATE] == d]
        test_like = build_features(derive_hourly(day), df15[df15[DATE] < d])
        res = _compare_frames(full[full[DATE] == d], test_like, FEATURE_COLS)
        records.append({"date": d.date(), **res})
    return pd.DataFrame(records)


def perturbation_check(df15: pd.DataFrame, full: pd.DataFrame, date, delta: float = 50.0) -> dict:
    """Add a zero-mean (+d, -d, +d, -d) pattern to D's quarter-hours in the history: D's features must not move."""
    d = pd.Timestamp(date)
    pert = df15.copy()
    mask = pert[DATE] == d
    pert.loc[mask, PRICE] += np.where(pert.loc[mask, QUARTER] % 2 == 1, delta, -delta)
    rebuilt = build_features(derive_hourly(df15[mask]), pert)
    return {"date": str(d.date()), "delta": delta, **_compare_frames(full[full[DATE] == d], rebuilt, FEATURE_COLS)}
