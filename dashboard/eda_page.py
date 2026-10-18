"""Data Story page: what one year of 15-minute prices taught us, told chapter by chapter.

Everything is computed from the official training CSV only (descriptive, full year including
September). Nothing here is fitted or used by the model; it explains why the model looks the way
it does. Opened through the multipage app:

    PYTHONPATH=. python -m streamlit run dashboard/app.py --server.address localhost
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import FONT, STRETCH, _style, colours, inject_style, kpi_tile, rgba, train_csv_path, zone_name

# Zone colours: categorical slots blue / orange / aqua / violet, validated with the dataviz checker
# (light and dark). Green and red stay reserved for the model and the hourly price on the results page.
ZONE_COLOURS = {
    "light": {"ZONE_0": "#2a78d6", "ZONE_1": "#eb6834", "ZONE_2": "#1baf7a", "ZONE_3": "#6250d6"},
    "dark": {"ZONE_0": "#3987e5", "ZONE_1": "#d95926", "ZONE_2": "#199e70", "ZONE_3": "#9085e9"},
}
SEASONS = ["Autumn", "Winter", "Spring", "Summer"]
SEASON_OF = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
SEASON_COLOURS = {"Autumn": "#eb6834", "Winter": "#2a78d6", "Spring": "#1baf7a", "Summer": "#eda100"}
UP, DOWN = "#eb6834", "#2a78d6"            # price climbs / drops inside the hour (diverging pair)
OFFSETS = np.array([-0.375, -0.125, 0.125, 0.375])
HOURS = [f"{h:02d}h" for h in range(24)]


def zone_colours() -> dict:
    return ZONE_COLOURS["dark" if colours()["surface"] == "#1a1a19" else "light"]


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Reading one year of 15-minute prices...")
def load_story(path: str) -> dict:
    df = pd.read_csv(path, dtype={"delivery_date": str})
    df["date"] = pd.to_datetime(df.delivery_date)
    w = (df.pivot_table(index=["date", "market_id", "hour_index"], columns="quarter_index", values="price_eur_mwh")
           .reset_index().rename(columns={1: "q1", 2: "q2", 3: "q3", 4: "q4", "market_id": "zone"}))
    w = w.sort_values(["zone", "date", "hour_index"]).reset_index(drop=True)
    Q = w[["q1", "q2", "q3", "q4"]].to_numpy()
    w["hourly"] = Q.mean(axis=1)
    w["hour"] = w.hour_index - 1
    w["season"] = w.date.dt.month.map(SEASON_OF)
    w["move"] = np.abs(Q - w.hourly.to_numpy()[:, None]).mean(axis=1)          # mean |15-min - hourly|
    w["tilt"] = w.q4 - w.q1
    day = w.groupby(["zone", "date"]).hourly
    prev, nxt = day.shift(1), day.shift(-1)
    w["change"] = w.hourly - prev                                               # visible hour-to-hour change
    w["jump"] = w.q1 - w.groupby(["zone", "date"]).q4.shift(1)                  # hidden boundary jump
    w["trend"] = np.where(prev.notna() & nxt.notna(), (nxt - prev) / 2,
                          np.where(nxt.notna(), nxt - w.hourly, w.hourly - prev))
    zones = sorted(w.zone.unique())

    stats = []
    for z in zones:
        p, r = df[df.market_id == z].price_eur_mwh, w[w.zone == z]
        resid = (r[["q1", "q2", "q3", "q4"]].to_numpy() - r.hourly.to_numpy()[:, None]).ravel()
        stats.append({"zone": z, "mean": p.mean(), "std": p.std(), "min": p.min(), "max": p.max(),
                      "negative": (p < 0).mean() * 100, "flat_mae": np.abs(resid).mean(),
                      "inside_share": resid.var() / p.var() * 100})
    stats = pd.DataFrame(stats).set_index("zone")

    daily = (w.groupby(["zone", "date"]).agg(avg=("hourly", "mean"), low=("hourly", "min"), high=("hourly", "max"),
                                             move=("move", "mean")).reset_index())
    daily["range"] = daily.high - daily.low
    for col in ("avg", "range", "move"):
        daily[f"{col}_7d"] = daily.groupby("zone")[col].transform(lambda s: s.rolling(7, min_periods=4, center=True).mean())
    range_corr = {z: daily[daily.zone == z][["range", "move"]].corr(method="spearman").iloc[0, 1] for z in zones}
    level_corr = {z: daily[daily.zone == z][["avg", "move"]].corr(method="spearman").iloc[0, 1] for z in zones}

    profile = w.groupby(["zone", "hour"]).hourly.median().unstack(0)
    season_profile = {"All": w.groupby(["season", "hour"]).hourly.median().unstack(0)}
    season_profile.update({z: w[w.zone == z].groupby(["season", "hour"]).hourly.median().unstack(0) for z in zones})
    df["season"], df["hour"] = df.date.dt.month.map(SEASON_OF), df.hour_index - 1
    negative = {"All": (df.assign(neg=df.price_eur_mwh < 0).groupby(["season", "hour"]).neg.mean().unstack(1) * 100)}
    negative.update({z: (df[df.market_id == z].assign(neg=lambda d: d.price_eur_mwh < 0)
                         .groupby(["season", "hour"]).neg.mean().unstack(1) * 100) for z in zones})
    season_move = w.groupby(["season", "zone"]).move.mean().unstack(1)

    change_box = w.groupby("hour").change.quantile([.05, .25, .5, .75, .95]).unstack(1)
    hidden = w.groupby("hour").agg(visible=("change", lambda s: s.abs().median()),
                                   jump=("jump", lambda s: s.abs().median()), move=("move", "median"))
    df["month"] = df.date.dt.month
    spikes = df[df.price_eur_mwh > 400]
    floors = df[df.price_eur_mwh < -400]
    extremes = {"by_hour": pd.DataFrame({"Spikes (> 400)": spikes.groupby("hour").size(),
                                         "Floor (< −400)": floors.groupby("hour").size()}).reindex(range(24)).fillna(0),
                "by_zone": pd.DataFrame({"Spikes (> 400)": spikes.groupby("market_id").size(),
                                         "Floor (< −400)": floors.groupby("market_id").size()}).reindex(zones).fillna(0),
                "spike_months": spikes.groupby(spikes.date.dt.strftime("%b")).size().sort_values(ascending=False),
                "floor_days": sorted(floors.delivery_date.unique())}

    tilt = w.groupby("hour").tilt.agg(["mean", "median"])
    tilt["consistency"] = w.groupby("hour").apply(
        lambda g: (np.sign(g.tilt) == np.sign(g.tilt.mean())).mean() * 100, include_groups=False)
    tilt_zone = w.groupby(["zone", "hour"]).tilt.mean().unstack(0)
    resid = (w[["q1", "q2", "q3", "q4"]].to_numpy() - w.hourly.to_numpy()[:, None])
    quarter_err = pd.DataFrame(np.abs(resid), columns=["Q1", "Q2", "Q3", "Q4"]).groupby(w.zone.to_numpy()).mean()

    ks = np.round(np.arange(0, 3.01, 0.1), 2)
    slope_rule, corr = {}, {}
    for z in zones:
        r = w[w.zone == z].dropna(subset=["trend"])
        R = r[["q1", "q2", "q3", "q4"]].to_numpy() - r.hourly.to_numpy()[:, None]
        T = r.trend.to_numpy()[:, None] * OFFSETS
        slope_rule[z] = [float(np.abs(R - k * T).mean()) for k in ks]
        corr[z] = float(np.corrcoef(r.trend, r.tilt)[0, 1])
    sample = w.dropna(subset=["trend"]).sample(min(40000, len(w)), random_state=0)[["zone", "trend", "tilt"]]

    return dict(zones=zones, stats=stats, daily=daily, range_corr=range_corr, level_corr=level_corr, profile=profile,
                season_profile=season_profile, negative=negative, season_move=season_move, change_box=change_box,
                hidden=hidden, extremes=extremes, tilt=tilt, tilt_zone=tilt_zone, quarter_err=quarter_err, ks=ks,
                slope_rule=slope_rule, corr=corr, sample=sample, first=w.date.min(), last=w.date.max(),
                days=int(w.date.nunique()))


# ------------------------------------------------------------------ building blocks
def chapter(number: int, title: str, lead: str) -> None:
    st.markdown(f"<div class='chapter-num'>Chapter {number}</div>", unsafe_allow_html=True)
    st.subheader(title)
    st.markdown(f"<div class='section-sub'>{lead}</div>", unsafe_allow_html=True)


def takeaway(text: str) -> None:
    st.markdown(f"<div class='takeaway'><b>For the model:</b> {text}</div>", unsafe_allow_html=True)


def chart(fig: go.Figure, c: dict, height: int, title: str) -> None:
    """Chart with its title as page text above it, so the title never collides with the legend."""
    st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
    st.plotly_chart(_style(fig, c, height), **STRETCH)


def hour_axis(fig: go.Figure) -> None:
    fig.update_xaxes(tickmode="array", tickvals=list(range(0, 24, 3)), ticktext=HOURS[::3], title_text="hour of the day")


def zone_lines(frame: pd.DataFrame, c: dict, zc: dict, height: int, y_title: str) -> go.Figure:
    """One line per zone over the hours, with a direct label at the line end (identity is never colour alone)."""
    fig = go.Figure()
    for z in frame.columns:
        fig.add_scatter(x=frame.index, y=frame[z], name=zone_name(z), line=dict(color=zc[z], width=2.5),
                        mode="lines+markers", marker=dict(size=6), hovertemplate="%{y:.1f} €/MWh")
        fig.add_annotation(x=frame.index[-1], y=frame[z].iloc[-1], text=zone_name(z), showarrow=False,
                           xanchor="left", xshift=8, font=dict(color=c["ink"], size=FONT["small"]))
    fig.update_layout(hovermode="x unified", yaxis_title=y_title)
    fig = _style(fig, c, height)
    fig.update_layout(margin=dict(l=10, r=80, t=50, b=10))
    return fig


# ------------------------------------------------------------------ chapters
def ch_zones(d: dict, c: dict, zc: dict) -> None:
    chapter(1, "Four zones, four different markets",
            "Same continent, same rules, very different personalities. Each zone has its own price level and its "
            "own appetite for risk.")
    character = {"ZONE_0": "Spiky: the full −500 to +747 range and the fattest tails",
                 "ZONE_1": "Pricier and the hardest to predict inside the hour",
                 "ZONE_2": "Expensive but calm: never negative, half the volatility",
                 "ZONE_3": "Cheap, with the most negative prices"}
    cols = st.columns(len(d["zones"]))
    for col, z in zip(cols, d["zones"]):
        s = d["stats"].loc[z]
        col.markdown(
            f"<div class='zone-card' style='border-top: 5px solid {zc[z]}'>"
            + kpi_tile(c, zone_name(z), f"{s['mean']:.0f}", " €/MWh average",
                       f"volatility (std) <b>{s['std']:.0f}</b> · negative <b>{s['negative']:.1f}%</b> of quarters"
                       f"<br>range <b>{s['min']:.0f}</b> to <b>{s['max']:.0f}</b><br><i>{character.get(z, '')}</i>")
            + "</div>", unsafe_allow_html=True)
    inside = d["stats"].inside_share
    takeaway(f"the hourly price already explains {100 - inside.max():.0f}–{100 - inside.min():.0f}% of the price "
             f"variation. The challenge is decided in the last {inside.min():.0f}–{inside.max():.0f}%: the movement "
             "inside the hour. Each zone needs its own strength.")


def ch_rhythm(d: dict, c: dict, zc: dict) -> None:
    chapter(2, "The daily rhythm",
            "Median hourly price at each hour of the day. All four zones follow the same daily pattern; "
            "what differs is their price level.")
    fig = zone_lines(d["profile"], c, zc, 460, "median price (€/MWh)")
    hour_axis(fig)
    st.plotly_chart(fig, **STRETCH)
    takeaway("prices peak in the morning (07–08h) and are most expensive in the evening (19–20h). Where the "
             "curve is steep, the 15-minute prices move the most.")


def ch_seasons(d: dict, c: dict, zc: dict) -> None:
    chapter(3, "Seasons reshape the day",
            "The shape of the day changes with the season, and negative prices appear in spring and summer.")
    st.session_state.setdefault("eda_zone", "All")
    pick = st.segmented_control("Zone", ["All", *d["zones"]], key="eda_zone",
                                format_func=lambda z: z if z == "All" else zone_name(z)) or "All"
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        prof = d["season_profile"][pick]
        fig = go.Figure()
        for s in SEASONS:
            fig.add_scatter(x=prof.index, y=prof[s], name=s, line=dict(color=SEASON_COLOURS[s], width=2.5),
                            mode="lines", hovertemplate="%{y:.1f} €/MWh")
        fig.update_layout(hovermode="x unified", yaxis_title="median price (€/MWh)")
        hour_axis(fig)
        chart(fig, c, 420, "Price by hour, per season")
    with right:
        neg = d["negative"][pick].reindex(SEASONS)
        fig = go.Figure(go.Heatmap(z=neg.to_numpy(), x=HOURS, y=SEASONS, colorscale=[[0, c["surface"]], [1, "#184f95"]],
                                   zmin=0, colorbar=dict(title="%", thickness=12),
                                   hovertemplate="%{y} · %{x}: %{z:.1f}% negative<extra></extra>"))
        fig.update_xaxes(tickvals=HOURS[::3])
        chart(fig, c, 420, "Share of negative 15-min prices")
    m = d["season_move"].mean(axis=1)
    takeaway(f"winter moves {100 * (1 - m['Winter'] / m.drop('Winter').mean()):.0f}% less inside the hour than the "
             "other seasons, which behave almost alike. A simple winter vs rest-of-year signal is enough; negative "
             "prices are a spring and summer midday event.")


def ch_trends(d: dict, c: dict, zc: dict) -> None:
    chapter(4, "The year, day by day",
            "7-day trends of the daily average price and of the daily swing (priciest minus cheapest hour).")
    st.session_state.setdefault("eda_trend", "Daily average")
    view = st.segmented_control("Show", ["Daily average", "Daily swing", "Movement inside the hour"],
                                key="eda_trend") or "Daily average"
    col = {"Daily average": "avg_7d", "Daily swing": "range_7d", "Movement inside the hour": "move_7d"}[view]
    fig = go.Figure()
    for z in d["zones"]:
        dz = d["daily"][d["daily"].zone == z]
        fig.add_scatter(x=dz.date, y=dz[col], name=zone_name(z), line=dict(color=zc[z], width=2),
                        hovertemplate="%{y:.1f} €/MWh")
    fig.update_layout(hovermode="x unified", yaxis_title="€/MWh (7-day average)")
    st.plotly_chart(_style(fig, c, 420), **STRETCH)
    rc, lc = d["range_corr"], d["level_corr"]
    takeaway(f"days with a wide swing also move more inside the hour (rank correlation {min(rc.values()):.2f}–"
             f"{max(rc.values()):.2f}), far more than the price level does ({min(lc.values()):.2f}–{max(lc.values()):.2f}). "
             "The day's swing is known from the 24 hourly prices, so it is a usable feature.")


def ch_spikes(d: dict, c: dict, zc: dict) -> None:
    chapter(5, "Hour-to-hour jumps and price spikes",
            "How much the price changes from one hour to the next, and when the extreme prices happen.")
    left, right = st.columns(2, gap="large")
    with left:
        b = d["change_box"]
        fig = go.Figure(go.Box(x=HOURS, q1=b[.25], median=b[.5], q3=b[.75], lowerfence=b[.05], upperfence=b[.95],
                               marker_color=c["ink2"], line=dict(width=1.5), fillcolor=rgba(c["smooth"], 0.25),
                               name="change", hoverinfo="skip"))
        fig.add_hline(y=0, line=dict(color=c["axis"], width=1))
        fig.update_layout(showlegend=False, yaxis_title="€/MWh vs previous hour")
        fig.update_xaxes(tickvals=HOURS[::3])
        chart(fig, c, 420, "Change into each hour (box 25–75%, whiskers 5–95%)")
    with right:
        e = d["extremes"]["by_hour"]
        fig = go.Figure()
        for name, colour in (("Spikes (> 400)", UP), ("Floor (< −400)", DOWN)):
            fig.add_bar(x=HOURS, y=e[name], name=name, marker=dict(color=colour, cornerradius=4))
        fig.update_layout(barmode="group", yaxis_title="number of 15-min prices")
        fig.update_xaxes(tickvals=HOURS[::3])
        chart(fig, c, 420, "When extreme prices happen")
    ex = d["extremes"]
    top = ", ".join(f"{m} ({n})" for m, n in ex["spike_months"].head(2).items())
    med = d["change_box"][.5]
    takeaway(f"the price climbs hardest into {med.idxmax():02d}h (median +{med.max():.0f} €/MWh) and drops hardest into "
             f"{med.idxmin():02d}h ({med.min():.0f}). Spikes above 400 are rare and "
             f"cluster in the evening, mostly in {top}; the −500 floor only hit on {' and '.join(ex['floor_days'])}, "
             "at midday, in several zones at once. Rare but predictable in time, so the model uses an "
             "outlier-robust (MAE) loss.")


def ch_sawtooth(d: dict, c: dict, zc: dict) -> None:
    chapter(6, "Inside the hour: the sawtooth",
            "Last quarter minus first quarter of every hour, averaged over the year. Inside each hour the price "
            "slides in the direction the day is moving, then snaps back when the next hour starts.")
    left, right = st.columns([1.6, 1], gap="large")
    t = d["tilt"]
    with left:
        fig = go.Figure(go.Bar(x=HOURS, y=t["mean"], marker=dict(color=[UP if v > 0 else DOWN for v in t["mean"]],
                                                                 cornerradius=4),
                               text=[f"{v:.0f}%" for v in t.consistency], textposition="outside",
                               textfont=dict(color=c["ink2"], size=FONT["small"] - 4), cliponaxis=False,
                               customdata=t.consistency,
                               hovertemplate="%{x}: Q4 − Q1 = %{y:+.1f} €/MWh<br>same direction on %{customdata:.0f}% "
                                             "of days<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=c["axis"], width=1))
        fig.update_layout(showlegend=False, yaxis_title="Q4 − Q1 (€/MWh)")
        fig.update_xaxes(tickvals=HOURS[::3])
        chart(fig, c, 440, "Orange: climbs inside the hour · blue: drops · label: days in that direction")
    with right:
        q = d["quarter_err"]
        fig = go.Figure()
        for z in q.index:
            fig.add_bar(x=q.columns, y=q.loc[z], name=zone_name(z), marker=dict(color=zc[z], cornerradius=4))
        fig.update_layout(barmode="group", yaxis_title="flat-guess error (€/MWh)")
        chart(fig, c, 440, "Error of a flat guess, by quarter")
    edge = float((q[["Q1", "Q4"]].mean(axis=1) / q[["Q2", "Q3"]].mean(axis=1)).mean())
    takeaway(f"the first and last quarter carry about {edge:.1f}× the error of the middle two. The biggest climb is at "
             f"{t['mean'].idxmax():02d}h (+{t['mean'].max():.0f} €/MWh) and the biggest drop at {t['mean'].idxmin():02d}h "
             f"({t['mean'].min():.0f}). {int((t.consistency >= 80).sum())} of 24 hours move the same way on at least 80% "
             f"of days; the turning points ({', '.join(f'{h:02d}h' for h in t.consistency.nsmallest(3).sort_index().index)}) "
             "are close to a coin flip.")


def ch_slope(d: dict, c: dict, zc: dict) -> None:
    chapter(7, "The hourly curve already tells the shape",
            "On the test day we only get 24 hourly prices. Does their slope predict the tilt inside the hour?")
    left, right = st.columns(2, gap="large")
    with left:
        s = d["sample"]
        lim_x, lim_y = np.nanpercentile(np.abs(s.trend), 99), np.nanpercentile(np.abs(s.tilt), 99)
        fig = go.Figure(go.Histogram2d(x=s.trend, y=s.tilt, nbinsx=70, nbinsy=70,
                                       xbins=dict(start=-lim_x, end=lim_x), ybins=dict(start=-lim_y, end=lim_y),
                                       colorscale=[[0, c["surface"]], [0.02, "#cde2fb"], [1, "#104281"]],
                                       showscale=False, hovertemplate="%{z} hours<extra></extra>"))
        b = float(np.sum(s.trend * s.tilt) / np.sum(s.trend ** 2))
        fig.add_scatter(x=[-lim_x, lim_x], y=[-b * lim_x, b * lim_x], mode="lines", line=dict(color=UP, width=2.5),
                        name=f"best line: tilt ≈ {b:.2f} × slope", hoverinfo="skip")
        fig.update_layout(xaxis_title="hourly slope around the hour (€/MWh per hour)", yaxis_title="Q4 − Q1 (€/MWh)")
        chart(fig, c, 440, "Each cell: how many hours look like this")
    with right:
        fig = go.Figure()
        for z in d["zones"]:
            fig.add_scatter(x=d["ks"], y=d["slope_rule"][z], name=zone_name(z), line=dict(color=zc[z], width=2.5),
                            hovertemplate="k = %{x}: %{y:.2f} €/MWh")
        fig.update_layout(hovermode="x unified", xaxis_title="k: how strongly the 4 quarters are tilted (0 = flat)",
                          yaxis_title="error (€/MWh)")
        chart(fig, c, 440, "Error of the slope rule")
    flat = np.mean([v[0] for v in d["slope_rule"].values()])
    best = np.mean([min(v) for v in d["slope_rule"].values()])
    corr = d["corr"]
    takeaway(f"the slope of the hourly curve correlates {min(corr.values()):.2f}–{max(corr.values()):.2f} with the "
             f"tilt inside the hour. One parameter, no training, cuts the flat error from {flat:.2f} to {best:.2f} "
             f"€/MWh ({(1 - best / flat) * 100:.0f}%). The model starts from a smooth curve through the hourly "
             "prices and learns what this simple rule misses (shown here on the full year, for exploration only).")


def ch_conclusion() -> None:
    st.subheader("What the model takes from the data")
    st.markdown(
        "1. **Predict only the shape inside the hour.** The hourly level is given; four zero-sum corrections are "
        "learned and re-centred so every hour averages exactly to its price.\n"
        "2. **Start from the slope.** A smooth curve through the 24 hourly prices already captures most of the shape.\n"
        "3. **Add what the slope misses:** curvature where the price turns, the turning points of the day, the day's "
        "swing, hour of day, season and zone.\n"
        "4. **Be robust.** Fat tails and synchronized extremes call for an MAE (L1) loss.\n"
        "5. **Check honestly.** Every result is scored on later dates than the data it learned from.\n\n"
        "→ See it at work on the **Model Results** page.")


def render() -> None:
    c, zc = colours(), zone_colours()
    inject_style(c)
    st.markdown(
        "<style>"
        f".chapter-num {{ font-size: 0.85rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;"
        f" color: {c['ink2']}; margin-top: 1.2rem; }}"
        f".takeaway {{ border-left: 4px solid {c['model']}; padding: 0.7rem 1rem; margin: 0.4rem 0 1.4rem;"
        f" background: {rgba(c['model'], 0.06)}; border-radius: 0 10px 10px 0; font-size: 1.02rem; color: {c['ink']}; }}"
        f".chart-title {{ font-weight: 600; font-size: 1rem; color: {c['ink']}; margin-bottom: -0.3rem; }}"
        ".zone-card .kpi { border-top-left-radius: 0; border-top-right-radius: 0; }"
        "</style>", unsafe_allow_html=True)
    try:
        path = str(train_csv_path())
    except FileNotFoundError as exc:
        st.error(f"Official training CSV not found: {exc}. Start the app with `-- --train PATH`.")
        st.stop()
    d = load_story(path)
    st.markdown("<div class='hero-title'>The story behind the prices</div>"
                "<div class='hero-sub'>What one year of 15-minute prices taught us before building the model.</div>"
                f"<div class='hero-meta'>{len(d['zones'])} zones · {d['first']:%d %b %Y} – {d['last']:%d %b %Y} · "
                f"{d['days']} days · official training data only, descriptive</div>", unsafe_allow_html=True)
    for i, ch in enumerate((ch_zones, ch_rhythm, ch_seasons, ch_trends, ch_spikes, ch_sawtooth, ch_slope)):
        if i:
            st.divider()
        ch(d, c, zc)
    st.divider()
    ch_conclusion()
