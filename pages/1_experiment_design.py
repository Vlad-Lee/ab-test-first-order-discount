import streamlit as st
from src.sidebar import render_sidebar
from src.stats import calculate_sample_size
from src.viz import plot_power_curve, plot_duration_tradeoff

render_sidebar()

st.title("Experiment Design")

required_n = calculate_sample_size(
    st.session_state["base_rate"],
    st.session_state["mde"],
    st.session_state["alpha"],
    st.session_state["power"],
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Required n (per group)", f"{required_n:,}")
col2.metric("Total users needed",     f"{required_n * 2:,}")
col3.metric("Days at 1k/day",         f"{required_n / 500:.0f}")
col4.metric("Days at 5k/day",         f"{required_n / 2500:.0f}")

st.divider()

left, right = st.columns(2)
with left:
    st.plotly_chart(
        plot_power_curve(st.session_state["base_rate"], st.session_state["alpha"], st.session_state["power"]),
        use_container_width=True,
    )
with right:
    st.plotly_chart(
        plot_duration_tradeoff(required_n),
        use_container_width=True,
    )