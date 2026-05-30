import streamlit as st
from src.sidebar import render_sidebar
from src.stats import calculate_sample_size
from src.viz import plot_power_curve, plot_duration_tradeoff

render_sidebar()

st.title("Experiment Design")

with st.expander("About this page"):
    st.markdown("""
    **What this page answers:** How many users do you need, and how long will the experiment take?

    Before running any experiment you need to commit to a sample size. The two key inputs are the
    **base retention rate** (what retention looks like today without the discount) and the
    **minimum detectable effect (MDE)**, the smallest lift you actually care about detecting.

    **Power curve:** Sample size grows exponentially as MDE shrinks. Designing to detect a 1pp lift
    requires roughly 16× more users than detecting a 4pp lift. This tradeoff is the central tension
    in experiment design, smaller effects require longer experiments.

    **Duration chart:** Translates sample size into calendar time given your platform's daily new-user
    volume. The 14 and 30-day reference lines mark typical experiment windows. If your required duration
    exceeds 30 days, you either need to relax the MDE or accept lower power.
    """)

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