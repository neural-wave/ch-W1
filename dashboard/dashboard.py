"""PREVIEW of the quarter-hour price dashboard with an all-zones view (see aet/dashboard.py).

Adds "All" to the zone switcher: the official score (mean of the per-zone MAEs), a per-zone table and
one price panel per zone. Zone prices are never averaged: the zones are different markets.

Run from the repository root, on its own or as the Model Results page of dashboard/app.py:
    python -m pip install -r requirements.txt
    PYTHONPATH=. python -m streamlit run dashboard/dashboard.py --server.address localhost
    # official CSV elsewhere:  ... --server.address localhost -- --train path/to/train_prices_15min.csv
    # your own test days:      ... --server.address localhost -- --history history_15min.csv \
    #                              --truth truth_15min.csv [--hourly test_hourly.csv] [--predictions predictions.csv]

Shows the frozen model trained through 31 Aug 2026 (artifacts_september_assessment, committed) on 1-30 Sep 2026.
Each date is replayed with quarter-hours strictly before it. Nothing is trained, tuned or written to disk;
the risk reference is the frozen heuristic of `aet.volatility` (training slice through 31 Jul 2026).

With --truth, the dashboard instead replays the supplied test days with the submission model (artifacts_official,
or --model): history from --history, hourly prices from --hourly (or the truth's hourly means) and, if given,
the reconstruction written by `aet.infer` (--predictions) as the model line.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from aet.cube import HourlyCube, QuarterCube
from aet.history import HistoryStore
from intervals import band, chance_within, load_table   # dashboard/intervals.py
from aet.pipeline import ShapePipeline
from aet.schema import read_hourly, read_quarter_hourly
from aet.volatility import ShapeRisk, risk_inputs

ROOT = Path(__file__).resolve().parent.parent          # repository root (parent of dashboard/)
ASSESSMENT_MODEL = str(ROOT / "artifacts_september_assessment")   # trained through 2026-08-31; absolute path
OFFICIAL_MODEL = str(ROOT / "artifacts_official")                # trained through 2026-09-30: for later test days
INTERVALS = str(Path(__file__).parent / "september_80.json")   # dashboard/intervals.py, calibrated on Jan-Aug folds
FIRST_DAY, LAST_DAY = "2026-09-01", "2026-09-30"
RISK_TRAINING_END = "2026-07-31"                      # same slice as `aet.volatility diagnose`
TRAIN_CSV_CANDIDATES = (                              # anchored to the repository root, not the current directory
    str(ROOT / "data" / "train_prices_15min.csv"),
    str(ROOT.parent / "data" / "train_prices_15min.csv"),
    str(ROOT.parent / "Data" / "train_prices_15min.csv"),
)

MODEL_LABEL = "Model Price Prediction"   # display name of the final reconstructed price (chart line)
RISK_NOTE = "Risk labels are heuristic indicators of difficult conditions, not anomaly probabilities."
FLAT_COMPARISON_HELP = (
    "**Flat baseline:** repeat the supplied hourly average price for all four 15-minute prices. "
    "For example, an hourly price of 100 €/MWh gives [100, 100, 100, 100].\n\n"
    "**Lower than flat** measures the reduction in prediction error (MAE: the average absolute "
    "difference from the actual 15-minute prices).\n\n"
    "**Example:** flat MAE = 10 €/MWh and model MAE = 5 €/MWh means **50% lower error**.\n\n"
    "**Formula:** `(1 − method MAE / flat MAE) × 100`. A negative reduction means higher error.\n\n"
    "The comparison uses the selected dates and zone. In the All view, the overall comparison "
    "uses equally weighted zone MAEs; each table row compares its own zone.\n\n"
    "The percentage describes prediction error, not a reduction in electricity prices or "
    "the probability that a prediction is correct."
)
ALL = "All"
# What the charts can show, toggled in the bar above the charts.
SHOW_OPTIONS = {"hourly": "Hourly price", "actual": "Actual 15-min price", "baseline": "Smooth curve",
                "model": MODEL_LABEL, "band50": "50% band", "band90": "90% band", "risk": "High-risk hours"}
DEFAULT_SHOW = ("hourly", "actual", "model", "band90")   # smooth curve, 50% band, risk: optional
BANDS = (("range_lo", "range_hi", "90% band", 0.13, "band90"), ("mid_lo", "mid_hi", "50% band", 0.28, "band50"))


def zone_name(zone: str) -> str:
    """ZONE_0 -> Zone 0 for display; data keeps the official market_id."""
    return zone.replace("ZONE_", "Zone ")


def visible_bands(show: dict):
    return [(lo, hi, name, alpha) for lo, hi, name, alpha, key in BANDS if show[key]]
PERIODS = ["Day", "Week", "Month", "Custom"]
WITHIN = 5.0                             # EUR/MWh: "chance the actual price lands within ± this of the prediction"

# Basis components of the learned correction, in model order, with plain-language meaning.
COMPONENTS = ("slope", "curvature", "zigzag")
PART_LABELS = {
    "smooth": "Smooth curve shape",
    "slope": "Slope: rising or falling inside the hour",
    "curvature": "Curvature: edges vs middle",
    "zigzag": "Zigzag: alternating detail",
    "rules": "Rules: connected zones aligned, quarters keep the hour's sign",
}
SHORT_LABELS = {"smooth": "Smooth", "slope": "Slope", "curvature": "Curve", "zigzag": "Zigzag", "rules": "Rules"}
RISK_REASONS = {   # the strongest of the three risk signals, in plain words
    "hourly_slope": "the hourly price changes sharply around this hour",
    "daily_std": "prices swing a lot during this day",
    "historical_residual_magnitude": "this hour has been hard to predict in recent days",
}

# Validated with the dataviz palette checker (adjacent and all-pairs, light and dark).
# Hourly price red, smooth curve blue, final model green; learned parts yellow / violet / magenta;
# the fixed post-processing rules use the neutral secondary ink (not a categorical hue).
PALETTES = {
    "light": dict(hourly="#e34948", smooth="#2a78d6", model="#008300", slope="#eda100", curvature="#6250d6",
                  zigzag="#e87ba4", rules="#52514e", ink="#0b0b0b", ink2="#52514e", grid="#e1e0d9", axis="#c3c2b7",
                  surface="#fcfcfb"),
    "dark": dict(hourly="#e66767", smooth="#3987e5", model="#008300", slope="#c98500", curvature="#9085e9",
                 zigzag="#d55181", rules="#c3c2b7", ink="#f0efec", ink2="#c3c2b7", grid="#2c2c2a", axis="#383835",
                 surface="#1a1a19"),
}
RISK_COLOURS = {"high": "#fab219"}   # status palette: warning (red is taken by the hourly price)
BETTER, WORSE = "#16c45a", "#ef4444"  # bright arrow icons for "vs flat"; the words beside them carry the meaning
RISK_BADGE = {"low": "🟢 Low risk", "medium": "🟡 Medium risk", "high": "⚠️ High risk", "unknown": "⚪ Unknown"}
FONT = dict(base=18, small=17)       # chart text sizes (px); page text is scaled in main()


# ------------------------------------------------------------------ compatibility helpers
def _streamlit_version() -> tuple[int, int]:
    major, minor = st.__version__.split(".")[:2]
    return int(major), int(minor)


# `width="stretch"` replaced `use_container_width` in newer Streamlit; requirements allow >= 1.40.
STRETCH = {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}


def colours() -> dict:
    theme = getattr(getattr(st, "context", None), "theme", None)
    return PALETTES["dark" if getattr(theme, "type", None) == "dark" else "light"]


# ------------------------------------------------------------------ data
def cli() -> argparse.Namespace:
    """Options after `--` on the streamlit command line (all optional)."""
    parser = argparse.ArgumentParser(add_help=False)
    for option in ("--train", "--model", "--history", "--truth", "--hourly", "--predictions"):
        parser.add_argument(option)
    return parser.parse_known_args()[0]


def custom_mode() -> bool:
    """True when the user supplied their own test days (--truth) instead of the September replay."""
    return cli().truth is not None


def model_dir() -> str:
    return cli().model or (OFFICIAL_MODEL if custom_mode() else ASSESSMENT_MODEL)


def train_csv_path() -> Path:
    """--train after `--` on the streamlit command line, then $AET_TRAIN_CSV, then the usual locations."""
    explicit = cli().train or os.environ.get("AET_TRAIN_CSV")
    for candidate in ([explicit] if explicit else TRAIN_CSV_CANDIDATES):
        if Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(explicit or " or ".join(TRAIN_CSV_CANDIDATES))


def require_model() -> None:
    """Stop with a readable message if the model folder is missing (see dashboard/README.md)."""
    if not Path(model_dir(), "manifest.json").is_file():
        st.error(f"Model not found in `{model_dir()}`. Pass `-- --model PATH`, or regenerate the September "
                 "hold-out model with the commands in dashboard/README.md (trained through 2026-08-31).")
        st.stop()


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quarter and hour rows of the September replay or, with --truth, of the user's own test days."""
    try:
        path = str(train_csv_path())
    except FileNotFoundError as exc:
        st.error(f"Official training CSV not found: {exc}. Start the app with `-- --train PATH`.")
        st.stop()
    require_model()
    if not custom_mode():
        return load_assessment(path)
    args = cli()
    if not args.history:
        st.error("`--truth` also needs `--history history_15min.csv` (the 15-minute prices before the test days).")
        st.stop()
    try:
        return load_custom(path, model_dir(), args.history, args.truth, args.hourly, args.predictions)
    except (OSError, ValueError) as exc:
        st.error(f"Could not load your test data: {exc}")
        st.stop()


def decompose_day(pipe: ShapePipeline, history: QuarterCube, day, H: np.ndarray, final: np.ndarray | None = None):
    """H [zone, 24] -> (store, baseline [zone,24,4], {component: [zone,24,4]}, final [zone,24,4]).

    final = baseline + residual_scale * sum_j coefficient_j * basis_j, then the pipeline's fixed
    post-processing (zone coupling, partial tie, sign floor/ceiling) and exact hourly-mean projection.
    The components are the learned correction split per basis vector plus "rules", the remainder
    added by post-processing, so baseline + all components = final. A supplied `final` (e.g. the
    predictions written by `aet.infer`) replaces the recomputed one.
    """
    H1 = H[None]
    base = pipe.baseline(H1)
    store = HistoryStore.build(history.before(day), pipe.config.baseline, pipe.basis)
    X = pipe._features(np.array([day]), H1, base, store)[pipe.model.feature_names]
    coefficients = np.column_stack([booster.predict(X) for booster in pipe.model.boosters])
    scale = pipe.config.residual_scale
    parts = {name: (scale * coefficients[:, [j]] * pipe.basis[:, j][None, :]).reshape(H.shape + (4,))
             for j, name in enumerate(COMPONENTS)}
    if final is None:
        final, _ = pipe.predict_day(day, H, history)
    parts["rules"] = final - base[0] - sum(parts.values())
    return store, base[0], parts, final


@st.cache_data(show_spinner="Replaying the 30 September days (about a minute the first time)...")
def load_assessment(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per quarter (prices and parts) and one row per hour (risk), for every zone and date."""
    prices = pd.read_csv(path, dtype={"delivery_date": str})
    pipe, stored = ShapePipeline.load(ASSESSMENT_MODEL)
    cube = QuarterCube.from_long(prices, pipe.zones)
    history = stored.merge(cube)                     # predict_day only reads dates before each target
    days = cube.dates[(cube.dates >= np.datetime64(FIRST_DAY)) & (cube.dates <= np.datetime64(LAST_DAY))]
    actual = cube.values[np.searchsorted(cube.dates, days)]
    return replay(pipe, history, fit_risk(prices, pipe), days, actual, actual.mean(axis=-1))


@st.cache_data(show_spinner="Replaying your test days with the frozen model...")
def load_custom(train: str, model: str, history_path: str, truth_path: str, hourly_path: str | None,
                predictions_path: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The user's own test days. History is merged over the model's stored prices exactly as in `aet.infer`;
    the truth is used only to score and draw the actual prices, never as history."""
    pipe, stored = ShapePipeline.load(model)
    history = stored.merge(QuarterCube.from_long(read_quarter_hourly(history_path, strict=False), pipe.zones))
    truth = QuarterCube.from_long(read_quarter_hourly(truth_path), pipe.zones)
    days, actual = truth.dates, truth.values
    if not len(days) or not np.isfinite(actual).all():
        raise ValueError(f"{truth_path}: every test day needs all 96 quarter-hours for zones {', '.join(pipe.zones)}")

    def on_test_days(cube, name: str) -> np.ndarray:
        missing = np.setdiff1d(days, cube.dates)
        if len(missing):
            raise ValueError(f"{name}: no rows for the test days {', '.join(map(str, missing))}")
        values = cube.values[np.searchsorted(cube.dates, days)]
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: incomplete rows for the test days in {truth_path}")
        return values

    H = actual.mean(axis=-1)
    if hourly_path:
        H = on_test_days(HourlyCube.from_long(read_hourly(hourly_path), pipe.zones), hourly_path)
    predicted = None
    if predictions_path:
        pred = pd.read_csv(predictions_path)
        pred["delivery_date"] = pd.to_datetime(pred["delivery_date"])
        predicted = on_test_days(QuarterCube.from_long(pred, pipe.zones, "predicted_price_eur_mwh"), predictions_path)
    prices = pd.read_csv(train, dtype={"delivery_date": str})
    return replay(pipe, history, fit_risk(prices, pipe), days, actual, H, predicted)


def fit_risk(prices: pd.DataFrame, pipe: ShapePipeline) -> ShapeRisk:
    """The frozen risk heuristic: reference scales from the training slice through RISK_TRAINING_END."""
    return ShapeRisk.fit(QuarterCube.from_long(prices[prices.delivery_date <= RISK_TRAINING_END], pipe.zones),
                         pipe.config)


def replay(pipe: ShapePipeline, history: QuarterCube, risk: ShapeRisk, days: np.ndarray, actual: np.ndarray,
           hourly: np.ndarray, predicted: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """actual [day, zone, 24, 4], hourly [day, zone, 24], optional predicted [day, zone, 24, 4] per day in `days`."""
    # Uncertainty: the calibrated error distribution per zone x hour x quarter gives a 90% band (outer),
    # a 50% band (inner) and the chance of landing within ±WITHIN of the prediction. NaN if not calibrated.
    empty = np.full((len(pipe.zones), 24, 4), np.nan)
    unc = dict(lo90=empty, hi90=empty, lo50=empty, hi50=empty, chance=empty)
    if Path(INTERVALS).is_file() and "percentiles" in (table := load_table(INTERVALS)):
        (unc["lo90"], unc["hi90"]), (unc["lo50"], unc["hi50"]) = (band(table["percentiles"], lv) for lv in (0.9, 0.5))
        unc["chance"] = chance_within(table["percentiles"], WITHIN)
    zone_idx, hour_idx, quarter_idx = np.meshgrid(np.arange(len(pipe.zones)), np.arange(24), np.arange(4), indexing="ij")
    quarter_frames, hour_frames = [], []
    for i, day in enumerate(days):
        H = hourly[i]
        store, base, parts, final = decompose_day(pipe, history, day, H, None if predicted is None else predicted[i])
        start = pd.Timestamp(str(day))
        quarter_frames.append(pd.DataFrame({
            "delivery_date": str(day),
            "market_id": np.asarray(pipe.zones)[zone_idx.ravel()],
            "hour_index": hour_idx.ravel() + 1,
            "quarter_index": quarter_idx.ravel() + 1,
            "timestamp": start + pd.to_timedelta(hour_idx.ravel() * 60 + quarter_idx.ravel() * 15, unit="min"),
            "hourly": np.repeat(H[..., None], 4, axis=-1).ravel(),
            "actual": actual[i].ravel(),
            "baseline": base.ravel(),
            "model": final.ravel(),
            "range_lo": (final + unc["lo90"]).ravel(),    # 90% band
            "range_hi": (final + unc["hi90"]).ravel(),
            "mid_lo": (final + unc["lo50"]).ravel(),      # 50% band
            "mid_hi": (final + unc["hi50"]).ravel(),
            "chance": unc["chance"].ravel(),             # P(|actual - prediction| <= WITHIN)
            "smooth": (base - H[..., None]).ravel(),
            **{name: part.ravel() for name, part in parts.items()},
        }))
        hours = risk.predict(risk_inputs(np.array([day]), H[None], store, pipe.config))
        hour_frames.append(hours.assign(delivery_date=str(day))[
            ["delivery_date", "market_id", "hour_index", "risk_label", "dominant_signal", "limited_history"]])
    return pd.concat(quarter_frames, ignore_index=True), pd.concat(hour_frames, ignore_index=True)


def zone_mae(rows: pd.DataFrame, column: str) -> float:
    """Official metric for one zone: MAE per day, then the mean over days."""
    return float((rows[column] - rows.actual).abs().groupby(rows.delivery_date).mean().mean())


def hour_label(hour: int) -> str:
    return f"{hour - 1:02d}:00–{hour % 24:02d}:00" if hour < 24 else "23:00–24:00"


# ------------------------------------------------------------------ charts
def _style(fig: go.Figure, c: dict, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=60, r=20, t=50, b=40), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color=c["ink2"], size=FONT["base"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=c["ink"], size=FONT["base"])),
        hoverlabel=dict(font=dict(color=c["ink"], size=FONT["base"])),
    )
    axis = dict(gridcolor=c["grid"], linecolor=c["axis"], zeroline=False, automargin=True,
                tickfont=dict(size=FONT["small"]), title_font=dict(size=FONT["base"]))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def risk_runs(hours: pd.DataFrame, labels=("high",)) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Merge consecutive flagged hours into bands: fewer shapes, same picture."""
    runs, current = [], None
    for row in hours.sort_values(["delivery_date", "hour_index"]).itertuples():
        start = pd.Timestamp(row.delivery_date) + pd.Timedelta(hours=row.hour_index - 1)
        label = row.risk_label if row.risk_label in labels else None
        if current and label == current[2] and start == current[1]:
            current = (current[0], start + pd.Timedelta(hours=1), label)
            continue
        if current:
            runs.append(current)
        current = (start, start + pd.Timedelta(hours=1), label) if label else None
    if current:
        runs.append(current)
    return runs


def rgba(hex_colour: str, alpha: float) -> str:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def has_range(rows: pd.DataFrame) -> bool:
    return bool(rows.range_lo.notna().any())


def price_chart(rows: pd.DataFrame, hours: pd.DataFrame, show: dict, show_risk: bool, c: dict) -> go.Figure:
    fig = go.Figure()
    custom = np.column_stack([rows.delivery_date, rows.hour_index])
    if has_range(rows):
        # Fan chart drawn first so every line sits on top: each band is an upper edge (no line), then the
        # lower edge filled up to it. Outer 90% band light, inner 50% band darker.
        for lo, hi, name, alpha in visible_bands(show):
            fig.add_scatter(x=rows.timestamp, y=rows[hi], line=dict(width=0), showlegend=False,
                            hoverinfo="skip", name=f"{name} high")
            fig.add_scatter(x=rows.timestamp, y=rows[lo], line=dict(width=0), fill="tonexty",
                            fillcolor=rgba(c["model"], alpha), name=name,
                            customdata=np.column_stack([rows[lo], rows[hi]]),
                            hovertemplate=f"{name}: %{{customdata[0]:.2f}} – %{{customdata[1]:.2f}} €/MWh<extra></extra>")
    if show["hourly"]:
        fig.add_scatter(x=rows.timestamp, y=rows.hourly, name="Hourly price (input)", line=dict(color=c["hourly"], width=2),
                        line_shape="hv", customdata=custom, hovertemplate="%{y:.2f} €/MWh")
    if show["baseline"]:
        fig.add_scatter(x=rows.timestamp, y=rows.baseline, name="Smooth curve (baseline)",
                        line=dict(color=c["smooth"], width=2, dash="dash"), customdata=custom, hovertemplate="%{y:.2f} €/MWh")
    if show["model"]:
        fig.add_scatter(x=rows.timestamp, y=rows.model, name=MODEL_LABEL, line=dict(color=c["model"], width=2.5),
                        customdata=custom, hovertemplate="%{y:.2f} €/MWh")
    if show["actual"]:
        fig.add_scatter(x=rows.timestamp, y=rows.actual, name="Actual 15-min price", mode="markers",
                        marker=dict(color=c["ink"], size=8 if rows.delivery_date.nunique() <= 2 else 5,
                                    line=dict(color=c["surface"], width=1)),
                        customdata=custom, hovertemplate="%{y:.2f} €/MWh")
    if show_risk:
        for x0, x1, label in risk_runs(hours):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=RISK_COLOURS[label], opacity=0.2, line_width=0, layer="below")
        # September is more volatile than the reference period, so most hours are at least medium:
        # shading only high-risk hours keeps the chart readable (medium shows in the hour badge).
        fig.add_scatter(x=[None], y=[None], mode="markers", name="High-risk hour", hoverinfo="skip",
                        marker=dict(symbol="square", size=12, color=RISK_COLOURS["high"], opacity=0.5))
    # Fixed axes: computed from every series, so ticking layers on or off never rescales the chart.
    values = rows[["hourly", "baseline", "model", "actual", "range_lo", "range_hi"]].to_numpy()
    values = values[np.isfinite(values)]
    pad = 0.05 * (values.max() - values.min()) or 1.0
    first = pd.Timestamp(rows.delivery_date.min())
    last = pd.Timestamp(rows.delivery_date.max()) + pd.Timedelta(days=1)
    # "closest" hover: with "x unified", Streamlit receives an empty selection when a point is clicked.
    fig.update_layout(hovermode="closest", yaxis_title="€/MWh", clickmode="event+select",
                      yaxis_range=[values.min() - pad, values.max() + pad], xaxis_range=[first, last])
    fig = _style(fig, c, 540)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def zones_chart(rows: pd.DataFrame, hours: pd.DataFrame, show: dict, show_risk: bool, c: dict) -> go.Figure:
    """Small multiples: one price panel per zone on a shared time axis. Prices are never averaged across zones."""
    zones = sorted(rows.market_id.unique())
    fig = make_subplots(rows=len(zones), cols=1, shared_xaxes=True, vertical_spacing=0.045, subplot_titles=[zone_name(z) for z in zones])
    for k, zone in enumerate(zones, start=1):
        z, cell, first = rows[rows.market_id == zone], dict(row=k, col=1), k == 1
        custom = np.column_stack([z.delivery_date, z.hour_index, z.market_id])   # a click also picks the zone
        if has_range(z):
            for lo, hi, name, alpha in visible_bands(show):
                fig.add_scatter(x=z.timestamp, y=z[hi], line=dict(width=0), showlegend=False, hoverinfo="skip",
                                legendgroup=name, **cell)
                fig.add_scatter(x=z.timestamp, y=z[lo], line=dict(width=0), fill="tonexty", name=name,
                                fillcolor=rgba(c["model"], alpha), legendgroup=name, showlegend=first,
                                hoverinfo="skip", **cell)
        if show["hourly"]:
            fig.add_scatter(x=z.timestamp, y=z.hourly, name="Hourly price (input)", line=dict(color=c["hourly"], width=1.5),
                            line_shape="hv", legendgroup="hourly", showlegend=first, customdata=custom,
                            hovertemplate="%{y:.2f} €/MWh", **cell)
        if show["baseline"]:
            fig.add_scatter(x=z.timestamp, y=z.baseline, name="Smooth curve (baseline)", legendgroup="baseline",
                            line=dict(color=c["smooth"], width=1.5, dash="dash"), showlegend=first, customdata=custom,
                            hovertemplate="%{y:.2f} €/MWh", **cell)
        if show["model"]:
            fig.add_scatter(x=z.timestamp, y=z.model, name=MODEL_LABEL, line=dict(color=c["model"], width=2),
                            legendgroup="model", showlegend=first, customdata=custom, hovertemplate="%{y:.2f} €/MWh", **cell)
        if show["actual"]:
            fig.add_scatter(x=z.timestamp, y=z.actual, name="Actual 15-min price", mode="markers",
                            marker=dict(color=c["ink"], size=5 if z.delivery_date.nunique() <= 2 else 3),
                            legendgroup="actual", showlegend=first, customdata=custom, hovertemplate="%{y:.2f} €/MWh", **cell)
        if show_risk:
            for x0, x1, label in risk_runs(hours[hours.market_id == zone]):
                fig.add_vrect(x0=x0, x1=x1, fillcolor=RISK_COLOURS[label], opacity=0.2, line_width=0, layer="below", **cell)
    if show_risk:
        fig.add_scatter(x=[None], y=[None], mode="markers", name="High-risk hour", hoverinfo="skip",
                        marker=dict(symbol="square", size=12, color=RISK_COLOURS["high"], opacity=0.5), row=1, col=1)
    first_day = pd.Timestamp(rows.delivery_date.min())
    fig.update_xaxes(range=[first_day, pd.Timestamp(rows.delivery_date.max()) + pd.Timedelta(days=1)])
    fig.update_layout(hovermode="closest", clickmode="event+select")
    fig = _style(fig, c, 210 * len(zones) + 90)
    fig.update_annotations(font=dict(color=c["ink"], size=FONT["base"]), x=0, xanchor="left")
    fig.update_yaxes(title_text="€/MWh", title_font=dict(size=FONT["small"] - 2))
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return fig


def decomposition_chart(hour_rows: pd.DataFrame, c: dict) -> go.Figure:
    """One small bar chart per quarter: hourly price -> + each part -> prediction, next to the actual price.

    Each bar starts where the previous one ended. Quarters in a 2 x 2 grid leave room for level step names
    and a value label on every bar; parts under 0.5 €/MWh show a muted "0". Prediction and actual sit side
    by side instead of on top of each other.
    """
    steps = ["smooth", *COMPONENTS, "rules"]
    x_steps = list(range(len(steps)))                  # numeric x so prediction and actual can sit side by side
    x_pred, x_actual = len(steps) - 0.15, len(steps) + 0.35
    ticks = dict(tickvals=[*x_steps, len(steps) + 0.1], ticktext=[*(SHORT_LABELS[s] for s in steps), "Result"])
    quarters = list(hour_rows.sort_values("quarter_index").itertuples())
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True, horizontal_spacing=0.04, vertical_spacing=0.12,
                        subplot_titles=[f"Q{q}" for q in range(1, 5)])
    for q, row in enumerate(quarters):
        cell = dict(row=q // 2 + 1, col=q % 2 + 1)
        level = row.hourly
        for x, step in zip(x_steps, steps):
            delta = getattr(row, step)
            small = abs(delta) < 0.5
            fig.add_bar(x=[x], y=[delta], base=[level], name=PART_LABELS[step], width=0.6, showlegend=False,
                        marker=dict(color=c[step], cornerradius=3),
                        text=["0" if small else f"{delta:+.1f}"], textposition="outside", constraintext="none",
                        cliponaxis=False, textangle=0,
                        textfont=dict(color=c["ink2"] if small else c["ink"], size=FONT["small"] - 2),
                        hovertemplate=f"{PART_LABELS[step]}: %{{y:+.2f}} €/MWh<extra></extra>", **cell)
            level += delta
        whisker = {}
        if np.isfinite(row.range_lo):   # 90% band as a whisker on the prediction
            whisker = dict(error_y=dict(type="data", symmetric=False, array=[row.range_hi - row.model],
                                        arrayminus=[row.model - row.range_lo], color=c["model"], thickness=2, width=6))
        fig.add_scatter(x=[x_pred], y=[row.model], mode="markers", name=MODEL_LABEL, legendgroup="model",
                        showlegend=q == 0, marker=dict(symbol="diamond", size=14, color=c["model"]),
                        customdata=[[row.range_lo, row.range_hi]],
                        hovertemplate=f"{MODEL_LABEL}: %{{y:.2f}} €/MWh"
                                      + ("<br>90% band: %{customdata[0]:.2f} – %{customdata[1]:.2f}"
                                         if whisker else "") + "<extra></extra>", **whisker, **cell)
        fig.add_scatter(x=[x_actual], y=[row.actual], mode="markers", name="Actual 15-min price", legendgroup="actual",
                        showlegend=q == 0, marker=dict(symbol="circle", size=11, color=c["ink"],
                                                       line=dict(color=c["surface"], width=2)),
                        hovertemplate="Actual: %{y:.2f} €/MWh<extra></extra>", **cell)
    # Legend entry for the hourly price line (the line itself is a shape, which has no legend).
    fig.add_scatter(x=[None], y=[None], mode="lines", name="Hourly price (input)",
                    line=dict(color=c["hourly"], width=2), row=1, col=1)
    fig.update_annotations(font=dict(color=c["ink"], size=FONT["base"]))   # quarter titles
    for q, row in enumerate(quarters):
        # Added after the traces: plotly skips shapes on subplots that are still empty.
        fig.add_hline(y=row.hourly, line=dict(color=c["hourly"], width=2), layer="below",
                      row=q // 2 + 1, col=q % 2 + 1)
    values = np.r_[hour_rows.hourly, hour_rows.model, hour_rows.actual, hour_rows.baseline,
                   hour_rows.range_lo, hour_rows.range_hi]
    values = values[np.isfinite(values)]
    pad = max(4.0, 0.12 * (values.max() - values.min()))   # room for the labels above and below the bars
    fig.update_yaxes(range=[values.min() - pad, values.max() + pad])
    fig.update_yaxes(title="€/MWh", col=1)
    fig.update_xaxes(range=[-0.6, len(steps) + 0.75], showgrid=False, tickangle=0, **ticks)
    fig.update_layout(barmode="overlay")
    fig = _style(fig, c, 720)
    fig.update_xaxes(tickfont=dict(size=FONT["small"] - 2))
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0))
    return fig

# ------------------------------------------------------------------ page
def _shift_day(step: int, days: list) -> None:
    """Previous / next day that has data (test days need not be consecutive)."""
    later = [d for d in days if d > st.session_state.day]
    earlier = [d for d in days if d < st.session_state.day]
    if step > 0 and later:
        st.session_state.day = later[0]
    elif step < 0 and earlier:
        st.session_state.day = earlier[-1]


def weeks(first, last) -> dict:
    """Consecutive 7-day blocks from the first day: label -> (start, end)."""
    blocks, start = {}, first
    while start <= last:
        end = min(start + pd.Timedelta(days=6).to_pytimedelta(), last)
        blocks[f"{start.day}–{end.day}" if start.month == end.month else f"{start:%d %b}–{end:%d %b}"] = (start, end)
        start = end + pd.Timedelta(days=1).to_pytimedelta()
    return blocks


def control_bar(zones: list[str], dates: list[str]) -> tuple[str, pd.Timestamp, pd.Timestamp, dict]:
    """Zone, period and what to show, in one bordered box above the charts."""
    days = [pd.Timestamp(d).date() for d in dates]
    first, last = days[0], days[-1]
    week_blocks = weeks(first, last)
    for key, value in (("day", first), ("zone", ALL), ("period", PERIODS[0]), ("week", next(iter(week_blocks))),
                       ("custom_range", (first, min(first + pd.Timedelta(days=6).to_pytimedelta(), last))),
                       ("show", list(DEFAULT_SHOW))):
        st.session_state.setdefault(key, value)
    with st.container(border=True):
        zone_col, period_col, pick_col = st.columns([1.5, 1.05, 1.45], gap="small", vertical_alignment="bottom")
        zone = zone_col.segmented_control("Zone", [ALL, *zones], key="zone",
                                          format_func=zone_name) or ALL
        period = period_col.segmented_control("Period", PERIODS, key="period") or PERIODS[0]
        with pick_col:
            if period == "Day":
                back, day, forward = st.columns([1, 5, 1], vertical_alignment="bottom")
                back.button(":material/chevron_left:", key="prev_day", help="Previous day", on_click=_shift_day,
                            args=(-1, days), disabled=st.session_state.day <= first, **STRETCH)
                day.date_input("Day", min_value=first, max_value=last, key="day", format="DD/MM/YYYY")
                forward.button(":material/chevron_right:", key="next_day", help="Next day", on_click=_shift_day,
                               args=(1, days), disabled=st.session_state.day >= last, **STRETCH)
                start = end = st.session_state.day
            elif period == "Week":
                week = st.segmented_control("Week", list(week_blocks), key="week") \
                    or next(iter(week_blocks))
                start, end = week_blocks[week]
            elif period == "Month":
                st.markdown(f"**{first:%d %b} – {last:%d %b %Y}** · all {len(days)} supplied test days" if custom_mode()
                            else "**1 – 30 September 2026** · the whole held-out month")
                start, end = first, last
            else:
                picked = st.date_input("Date range", min_value=first, max_value=last, key="custom_range", format="DD/MM/YYYY")
                picked = picked if isinstance(picked, (tuple, list)) else (picked,)
                start, end = (picked[0], picked[-1]) if picked else (first, first)
        picked = st.pills("Show on the charts", list(SHOW_OPTIONS), selection_mode="multi",
                          format_func=SHOW_OPTIONS.get, key="show")
    show = {key: key in (picked or []) for key in SHOW_OPTIONS}
    return zone, pd.Timestamp(start), pd.Timestamp(end), show


def kpi_tile(c: dict, label: str, value: str, unit: str = "", note: str = "", hero: bool = False) -> str:
    """One stat tile as HTML: label, value (+ unit), optional note. Text uses ink tokens, never series colours."""
    note_html = f"<div class='kpi-note' style='color:{c['ink2']}'>{note}</div>" if note else ""
    unit_html = f"<span class='kpi-unit' style='color:{c['ink2']}'>{unit}</span>" if unit else ""
    return (f"<div class='kpi{' kpi-hero' if hero else ''}' style='border-color:{c['grid']}'>"
            f"<div class='kpi-label' style='color:{c['ink2']}'>{label}</div>"
            f"<div class='kpi-value' style='color:{c['ink']}'>{value}{unit_html}</div>{note_html}</div>")


def vs_flat(value: float, flat: float) -> str:
    """Signed change vs the flat baseline: arrow icon + words, so direction is never colour alone."""
    change = (value / flat - 1) * 100
    arrow, colour, word = ("▼", BETTER, "lower") if change <= 0 else ("▲", WORSE, "higher")
    return (f"<span class='kpi-arrow' style='color:{colour}'>{arrow}</span> "
            f"<b>{abs(change):.0f}%</b> {word} than flat")


def flat_comparison_info() -> None:
    """Explain the shared baseline comparison in an accessible native info popover."""
    with st.popover("About ‘lower than flat’", icon=":material/info:",
                    help="Explain the hourly baseline and percentage error reduction."):
        st.markdown(FLAT_COMPARISON_HELP)


def score_cards(rows: pd.DataFrame, hours: pd.DataFrame, c: dict) -> None:
    """Stacked stat tiles: the model's error leads (the one hero figure), the references follow."""
    flat, smooth, model = (zone_mae(rows, col) for col in ("hourly", "baseline", "model"))
    high = int((hours.risk_label == "high").sum())
    st.markdown("<div class='kpi-heading'>Average error per 15-minute price <span>· lower is better</span></div>",
                unsafe_allow_html=True)
    flat_comparison_info()
    st.markdown(
        kpi_tile(c, "Model prediction error", f"{model:.2f}", " €/MWh", vs_flat(model, flat), hero=True)
        + kpi_tile(c, "Smooth curve error", f"{smooth:.2f}", " €/MWh", vs_flat(smooth, flat))
        + kpi_tile(c, "Hourly baseline error", f"{flat:.2f}", " €/MWh", "flat: the hourly price repeated 4×")
        + range_tile(rows, c)
        + kpi_tile(c, "High-risk hours", f"{high}", f" of {len(hours)}", "heuristic flag, not a probability"),
        unsafe_allow_html=True)


def share_inside(rows: pd.DataFrame, lo: str, hi: str) -> float:
    return float(((rows.actual >= rows[lo]) & (rows.actual <= rows[hi])).mean() * 100)


def official_mae(rows: pd.DataFrame, column: str) -> float:
    """The competition score: MAE per day and zone -> mean over days per zone -> simple mean over zones."""
    return float(np.mean([zone_mae(rows[rows.market_id == z], column) for z in sorted(rows.market_id.unique())]))


def score_cards_all(rows: pd.DataFrame, hours: pd.DataFrame, c: dict) -> None:
    flat, smooth, model = (official_mae(rows, col) for col in ("hourly", "baseline", "model"))
    high = int((hours.risk_label == "high").sum())
    st.markdown("<div class='kpi-heading'>Official score: mean of the zone errors <span>· lower is better</span></div>",
                unsafe_allow_html=True)
    flat_comparison_info()
    st.markdown(
        kpi_tile(c, "Model prediction error · all zones", f"{model:.2f}", " €/MWh", vs_flat(model, flat), hero=True)
        + kpi_tile(c, "Smooth curve error", f"{smooth:.2f}", " €/MWh", vs_flat(smooth, flat))
        + kpi_tile(c, "Hourly baseline error", f"{flat:.2f}", " €/MWh", "flat: the hourly price repeated 4×")
        + range_tile(rows, c)
        + kpi_tile(c, "High-risk hours", f"{high}", f" of {len(hours)}", "zone-hours · heuristic flag, not a probability"),
        unsafe_allow_html=True)


def zone_table(rows: pd.DataFrame) -> None:
    """One row per zone: the three errors, the gain over flat and the uncertainty check."""
    records = []
    for zone in sorted(rows.market_id.unique()):
        r = rows[rows.market_id == zone]
        flat, model = zone_mae(r, "hourly"), zone_mae(r, "model")
        records.append({
            "Zone": zone_name(zone), "Hourly baseline error": flat, "Smooth curve error": zone_mae(r, "baseline"),
            "Model error": model, "Lower than flat": (1 - model / flat) * 100,
            "In 90% band": share_inside(r, "range_lo", "range_hi") if has_range(r) else np.nan,
            f"Within ±{WITHIN:.0f}: historical": r.chance.mean() * 100,
            f"Within ±{WITHIN:.0f}: observed": ((r.actual - r.model).abs() <= WITHIN).mean() * 100,
        })
    money = st.column_config.NumberColumn(format="%.2f €/MWh")
    share = st.column_config.NumberColumn(format="%.0f%%")
    st.dataframe(pd.DataFrame(records), hide_index=True, **STRETCH, column_config={
        "Hourly baseline error": money, "Smooth curve error": money, "Model error": money,
        "Lower than flat": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100,
                                                          help=FLAT_COMPARISON_HELP),
        "In 90% band": share, f"Within ±{WITHIN:.0f}: historical": share, f"Within ±{WITHIN:.0f}: observed": share,
    })


def range_tile(rows: pd.DataFrame, c: dict) -> str:
    """Uncertainty from historical errors (interval width, share within ±5) next to what was observed in the selection."""
    if not has_range(rows):
        return kpi_tile(c, "Uncertainty", "–", "", "run <code>dashboard/intervals.py</code> to calibrate")
    # The ranges are asymmetric (e.g. −33 to +4 €/MWh), so show the width and the typical bounds, never "±".
    width = float((rows.range_hi - rows.range_lo).mean())
    below, above = float((rows.range_lo - rows.model).mean()), float((rows.range_hi - rows.model).mean())
    historical = float(rows.chance.mean() * 100)
    observed = float(((rows.actual - rows.model).abs() <= WITHIN).mean() * 100)
    return (kpi_tile(c, "Average 90% interval width", f"{width:.1f}", " €/MWh",
                     f"typically {below:+.1f} to {above:+.1f} €/MWh around the prediction · actual prices inside: "
                     f"<b>{share_inside(rows, 'range_lo', 'range_hi'):.0f}%</b> (90% band), "
                     f"<b>{share_inside(rows, 'mid_lo', 'mid_hi'):.0f}%</b> (50% band)")
            + kpi_tile(c, f"Historical share within ±{WITHIN:.0f} €/MWh", f"{historical:.0f}%", "",
                       f"from Jan–Aug errors for the same zone, hour and quarter · observed share in the selected "
                       f"period: <b>{observed:.0f}%</b>"))


def consume_click(event, zone: str) -> bool:
    """A newly clicked chart point (date, hour[, zone]) becomes the hour to explain. True if it was new."""
    points = list(getattr(getattr(event, "selection", None), "points", []) or [])
    clicked = next((p.get("customdata") for p in points if p.get("customdata") is not None), None)
    if clicked is None or tuple(clicked) == st.session_state.get("last_click"):
        return False
    st.session_state.last_click = tuple(clicked)
    st.session_state.explain_date, st.session_state.explain_hour = str(clicked[0]), int(clicked[1])
    st.session_state.explain_zone = str(clicked[2]) if len(clicked) > 2 else zone
    return True


def pick_hour(rows: pd.DataFrame, dates: list[str]) -> tuple[str, int]:
    """Hour explained below: a clicked chart point wins, otherwise keep or default to the largest correction."""
    if st.session_state.get("explain_date") not in dates:
        learned = rows[list(COMPONENTS)].abs().sum(axis=1).groupby([rows.delivery_date, rows.hour_index]).mean()
        best_date, best_hour = learned.idxmax()
        st.session_state.explain_date, st.session_state.explain_hour = str(best_date), int(best_hour)
    date = st.selectbox("Day", dates, key="explain_date")
    hour = st.select_slider("Hour", options=list(range(1, 25)), key="explain_hour", format_func=hour_label)
    return date, hour


def inject_style(c: dict) -> None:
    """Shared page styling (also used by the Data Story page)."""
    # Streamlit sizes text in rem, so a larger root font scales the whole page.
    # Also hide the toolbar, menu and "File change: Rerun / Always rerun" prompt (fallback for older Streamlit).
    st.markdown(
        "<style>html { font-size: 18px; }"
        "[data-testid='stMainMenu'], [data-testid='stAppDeployButton'], [data-testid='stToolbarActions'],"
        "[data-testid='stStatusWidget'] { display: none !important; }"
        ".block-container, [data-testid='stMainBlockContainer'] { padding-top: 3.5rem; max-width: 1500px; }"
        # Page header
        f".hero-title {{ font-size: 2.4rem; font-weight: 700; color: {c['ink']}; line-height: 1.15; margin: 0; }}"
        f".hero-sub {{ font-size: 1.25rem; color: {c['ink2']}; margin: 0.35rem 0 0.2rem; }}"
        f".hero-meta {{ font-size: 0.9rem; color: {c['ink2']}; opacity: 0.85; margin-bottom: 0.6rem; }}"
        # Stat tiles
        f".kpi-heading {{ font-size: 0.95rem; font-weight: 600; color: {c['ink']}; margin: 0.2rem 0 0.6rem; }}"
        f".kpi-heading span {{ font-weight: 400; color: {c['ink2']}; }}"
        f".kpi {{ border: 1px solid {c['grid']}; border-radius: 12px; padding: 0.75rem 1rem; margin-bottom: 0.7rem; }}"
        f".kpi-hero {{ border: 2px solid {c['model']}; }}"
        f".kpi-label {{ font-size: 0.85rem; font-weight: 600; color: {c['ink2']}; }}"
        f".kpi-value {{ font-size: 1.9rem; font-weight: 600; color: {c['ink']}; line-height: 1.2; }}"
        ".kpi-hero .kpi-value { font-size: 3rem; }"
        f".kpi-unit {{ font-size: 0.95rem; font-weight: 400; color: {c['ink2']}; margin-left: 0.15rem; }}"
        ".kpi-note { font-size: 0.95rem; margin-top: 0.2rem; }"
        ".reliability .kpi-value { margin: 0.25rem 0; font-size: 2.5rem; }"
        ".reliability-summary { font-size: 0.9rem; line-height: 1.45; margin-top: 0.2rem; }"
        f".reliability-note {{ font-size: 0.78rem; line-height: 1.5; color: {c['ink2']}; margin-top: 0.45rem; }}"
        f".reliability-result {{ border-top: 1px solid {c['grid']}; margin-top: 0.85rem; padding-top: 0.75rem; }}"
        ".kpi-arrow { font-size: 1.35rem; line-height: 1; vertical-align: -0.12rem; }"
        f".section-sub {{ font-size: 1.05rem; color: {c['ink2']}; margin: -0.4rem 0 1.2rem; }}"
        "</style>", unsafe_allow_html=True)


def main(standalone: bool = False) -> None:
    st.set_option("client.toolbarMode", "minimal")   # no Deploy button or developer menu
    st.set_page_config(page_title="Quarter-hour prices", page_icon="⚡", layout="wide")
    quarters, hours_all = load_results()
    c = colours()

    inject_style(c)
    st.markdown("<div class='hero-title'>Quarter-hour price reconstruction</div>"
                "<div class='hero-sub'>From 24 hourly prices to 96 quarter-hour prices.</div>",
                unsafe_allow_html=True)
    if custom_mode():
        source = "predictions from your file" if cli().predictions else "predictions recomputed by the dashboard"
        st.markdown(f"<div class='hero-meta'>Your test data · {quarters.delivery_date.nunique()} days · model "
                    f"<code>{Path(model_dir()).name}</code> · {source}</div>", unsafe_allow_html=True)

    zones = sorted(quarters.market_id.unique())
    zone, start, end, show = control_bar(zones, sorted(quarters.delivery_date.unique()))
    show_risk = show["risk"]
    in_period = lambda frame: frame[frame.delivery_date.between(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}")]
    all_rows, all_hours = in_period(quarters), in_period(hours_all)
    if all_rows.empty:
        st.warning("No test data for the selected dates. Pick another day or period.")
        st.stop()
    period = f"{start:%d %b}" + (f" – {end:%d %b %Y}" if end > start else f" {start:%Y}")

    # Row 1: the price chart(s) on the left, the scores on the right.
    left, right = st.columns([3, 1], gap="large")
    if zone == ALL:
        with right:
            score_cards_all(all_rows, all_hours, c)
        with left:
            st.subheader(f"All zones · {period}")
            event = st.plotly_chart(zones_chart(all_rows, all_hours, show, show_risk, c), key="price_chart",
                                    on_select="rerun", selection_mode="points", **STRETCH)
            st.caption("One panel per zone: prices are never averaged across zones (they are different markets). "
                       "Click a point to see how that prediction was built.")
        st.subheader("Per zone")
        zone_table(all_rows)
    else:
        rows, hours = all_rows[all_rows.market_id == zone], all_hours[all_hours.market_id == zone]
        with right:
            score_cards(rows, hours, c)
        with left:
            st.subheader(f"{zone_name(zone)} · {period}")
            event = st.plotly_chart(price_chart(rows, hours, show, show_risk, c), key="price_chart",
                                    on_select="rerun", selection_mode="points", **STRETCH)
            st.caption("Tip: click a point to see how that prediction was built.")

    if standalone:   # run on its own: no other pages to link to
        return
    import nav   # imported late: nav imports this module
    if consume_click(event, zone):
        st.switch_page(nav.UNDERSTANDING)
    st.divider()
    st.page_link(nav.UNDERSTANDING, label="Understanding the Model: why it predicts what it does, hour by hour",
                 icon=":material/arrow_forward:")



if __name__ == "__main__":
    main(standalone=True)
