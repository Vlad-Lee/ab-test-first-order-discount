import streamlit as st
from src.sidebar import render_sidebar

# set_page_config must be the first Streamlit call in the script
st.set_page_config(
    page_title="A/B Test Simulator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

st.title("First Order Discount: A/B Test Simulator")

st.markdown(
    "A food delivery platform is considering offering a first-order discount to improve "
    "30-day retention. This simulator models the full experimental workflow, from design "
    "to statistical analysis to revenue impact. Use the sidebar to configure parameters "
    "and navigate the pages using the left panel."
)

st.divider()

st.markdown("""
| Page | What it covers |
|---|---|
| **Experiment Design** | How many users you need and how long the experiment will take |
| **Simulation Results** | Statistical test, retention chart, novelty decay, revenue impact |
| **Segment Analysis** | Heterogeneous treatment effects and targeted vs. full rollout recommendation |
""")