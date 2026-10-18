"""Multipage dashboard: what we found, how well it works, why it works.

Run from the repository root:
    python -m streamlit run dashboard/app.py --server.address localhost
or from dashboard/ (the repository root is added to sys.path below):
    python -m streamlit run app.py --server.address localhost
"""
import sys
from pathlib import Path

# The `aet` package lives in the repository root, one level above this folder.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st  # noqa: E402

import nav  # noqa: E402

st.set_page_config(page_title="Quarter-hour prices", layout="wide")
nav.keep_shared_state()
st.navigation(nav.PAGES, position="top").run()
