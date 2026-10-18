"""Risk Alerts page: which hours to trust less, flagged before the day's 15-minute prices are known."""
import pandas as pd
import streamlit as st

from dashboard import (ALL, RISK_NOTE, RISK_REASONS, STRETCH, colours, hour_label, inject_style,
                       kpi_tile, load_results, zone_name)


def controls(zones: list[str], dates: list[str]) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    first, last = pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date()
    st.session_state.setdefault("risk_zone", ALL)
    st.session_state.setdefault("risk_range", (first, last))
    with st.container(border=True):
        zone_col, range_col = st.columns([1.5, 1], gap="medium", vertical_alignment="bottom")
        zone = zone_col.segmented_control("Zone", [ALL, *zones], key="risk_zone", format_func=zone_name) or ALL
        picked = range_col.date_input("Dates", min_value=first, max_value=last, key="risk_range", format="DD/MM/YYYY")
        picked = picked if isinstance(picked, (tuple, list)) else (picked,)
        start, end = (picked[0], picked[-1]) if picked else (first, last)
    return zone, pd.Timestamp(start), pd.Timestamp(end)


def hour_errors(rows: pd.DataFrame) -> pd.DataFrame:
    """Model and flat MAE per zone, date and hour (mean over the four quarters)."""
    return (rows.assign(model_error=(rows.model - rows.actual).abs(), flat_error=(rows.hourly - rows.actual).abs())
            .groupby(["market_id", "delivery_date", "hour_index"], as_index=False)[["model_error", "flat_error"]].mean())


def render() -> None:
    import nav   # imported late: nav imports this module
    c = colours()
    inject_style(c)
    quarters, hours_all = load_results()
    st.markdown("<div class='hero-title'>Risk alerts</div>"
                "<div class='hero-sub'>Which hours to trust less, flagged before the 15-minute prices are known.</div>",
                unsafe_allow_html=True)
    st.markdown(
        "Every hour of every day and zone gets a label (low, medium or high) using only what is known in advance: "
        "the day's 24 hourly prices and the previous days. Three signals are combined:\n"
        "1. **Hourly ramp:** how fast the hourly price changes around this hour.\n"
        "2. **Day volatility:** how much the 24 hourly prices swing during the day.\n"
        "3. **Recent difficulty:** how far the actual 15-minute prices moved away from the smooth curve at this hour "
        "over the previous 7 and 28 days.")
    st.caption(RISK_NOTE)

    zone, start, end = controls(sorted(quarters.market_id.unique()), sorted(quarters.delivery_date.unique()))
    between = lambda frame: frame[frame.delivery_date.between(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}")
                                  & ((frame.market_id == zone) if zone != ALL else True)]
    rows, hours = between(quarters), between(hours_all)
    hours = hours.merge(hour_errors(rows), on=["market_id", "delivery_date", "hour_index"])
    risky = hours[hours.risk_label == "high"].sort_values(["delivery_date", "hour_index", "market_id"])
    others = hours[hours.risk_label != "high"]

    # Does the flag mean something? Compare the model's error on flagged hours with the other hours.
    tiles = st.columns(3)
    tiles[0].markdown(kpi_tile(c, "High-risk hours", f"{len(risky)}", f" of {len(hours)}",
                               f"{len(risky) / max(len(hours), 1) * 100:.0f}% of the zone-hours in this selection"),
                      unsafe_allow_html=True)
    if len(risky):
        tiles[1].markdown(kpi_tile(c, "Model error · high-risk hours", f"{risky.model_error.mean():.2f}", " €/MWh",
                                   f"flat baseline: {risky.flat_error.mean():.2f} €/MWh"), unsafe_allow_html=True)
    tiles[2].markdown(kpi_tile(c, "Model error · other hours", f"{others.model_error.mean():.2f}", " €/MWh",
                               f"flat baseline: {others.flat_error.mean():.2f} €/MWh"), unsafe_allow_html=True)

    period = f"{start:%d %b}" + (f" – {end:%d %b %Y}" if end > start else f" {start:%Y}")
    st.subheader(f"High-risk hours · {zone_name(zone)} · {period}")
    if risky.empty:
        st.markdown("🟢 No high-risk hours in this selection.")
        return
    st.caption("Select a row to see how that prediction was built on the Understanding the Model page.")
    event = st.dataframe(pd.DataFrame({
        "Day": pd.to_datetime(risky.delivery_date).dt.strftime("%a %d %b"), "Zone": risky.market_id.map(zone_name),
        "Hour": risky.hour_index.map(hour_label), "Main reason": risky.dominant_signal.map(RISK_REASONS),
        "Model error": risky.model_error, "Flat baseline error": risky.flat_error,
    }), hide_index=True, **STRETCH, on_select="rerun", selection_mode="single-row", key="risk_table",
        column_config={"Model error": st.column_config.NumberColumn(format="%.2f €/MWh"),
                       "Flat baseline error": st.column_config.NumberColumn(format="%.2f €/MWh"),
                       "Main reason": st.column_config.TextColumn(width="large")})
    selected = event.selection.rows if event else []
    if selected:
        hour = risky.iloc[selected[0]]
        st.session_state.explain_zone, st.session_state.explain_date = hour.market_id, hour.delivery_date
        st.session_state.explain_hour = int(hour.hour_index)
        del st.session_state["risk_table"]   # so coming back to this page does not jump again
        st.switch_page(nav.UNDERSTANDING)
