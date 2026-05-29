import streamlit as st

DEFAULTS = {
    "base_rate": 0.27,
    "mde": 0.04,
    "alpha": 0.05,
    "power": 0.80,
    "n_users": 10_000,
    "treatment_effect": 0.04,
    "aov": 35.0,
    "discount_amount": 10.0,
}

def render_sidebar():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
        else:
            st.session_state[k] = type(v)(st.session_state[k])

    with st.sidebar:
        st.header("Experiment Parameters")
        st.slider("Base Retention Rate", 0.10, 0.50, step=0.01, key="base_rate")
        st.slider("MDE (pp)", 0.01, 0.10, step=0.01, key="mde")
        st.selectbox("Alpha", [0.01, 0.05, 0.10], index=[0.01, 0.05, 0.10].index(st.session_state["alpha"]), key="alpha")
        st.slider("Power", 0.70, 0.95, step=0.05, key="power")
        st.subheader("Simulation Parameters")
        st.slider("Users", 1_000, 50_000, step=1_000, key="n_users")
        st.slider("Treatment Effect (pp)", 0.01, 0.10, step=0.01, key="treatment_effect")
        st.subheader("Revenue Parameters")
        st.slider("AOV ($)", 10.0, 100.0, step=1.0, key="aov")
        st.slider("Discount ($)", 1.0, 20.0, step=0.5, key="discount_amount")