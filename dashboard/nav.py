"""The app's pages, defined once so pages can link to each other (st.switch_page / st.page_link)."""
import streamlit as st

import dashboard
import eda_page
import explain_page
import risk_page

DATA_STORY = st.Page(eda_page.render, title="Data Story", icon=":material/auto_stories:", url_path="data-story")
MODEL_RESULTS = st.Page(dashboard.main, title="Model Results", icon=":material/query_stats:", url_path="model-results",
                        default=True)
UNDERSTANDING = st.Page(explain_page.render, title="Understanding the Model", icon=":material/psychology:", url_path="understanding-the-model")
RISK_ALERTS = st.Page(risk_page.render, title="Risk Alerts", icon=":material/warning:", url_path="risk-alerts")
PAGES = [MODEL_RESULTS, UNDERSTANDING, RISK_ALERTS, DATA_STORY]

# Widget values that should survive moving between pages (Streamlit forgets widgets that are not drawn).
SHARED_KEYS = ("zone", "period", "day", "week", "custom_range", "show",          # Model Results
               "explain_zone", "explain_date", "explain_hour",                    # Understanding the Model
               "risk_zone", "risk_range",                                         # Risk Alerts
               "eda_zone", "eda_trend")                                           # Data Story


def keep_shared_state() -> None:
    for key in SHARED_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
