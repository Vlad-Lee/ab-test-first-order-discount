import streamlit as st
import numpy as np
from src.sidebar import render_sidebar
from src.simulate import generate_base_data, apply_novelty_effect
from src.stats import run_proportions_test, apply_bh_correction, calculate_revenue_impact
from src.viz import plot_retention_comparison, plot_novelty_effect, plot_revenue_waterfall

render_sidebar()

st.title("Simulation Results")

# ── Generate data ─────────────────────────────────────────────────────────────

@st.cache_data
def cached_generate(n_users, base_rate, treatment_effect, seed=42):
    return generate_base_data(n_users, base_rate, treatment_effect, seed=seed)

df = cached_generate(
    st.session_state["n_users"],
    st.session_state["base_rate"],
    st.session_state["treatment_effect"],
)

df_control   = df[df["group"] == "control"]
df_treatment = df[df["group"] == "treatment"]

# ── Proportions test ──────────────────────────────────────────────────────────

results = run_proportions_test(
    conversions_control   = int(df_control["retained"].sum()),
    n_control             = len(df_control),
    conversions_treatment = int(df_treatment["retained"].sum()),
    n_treatment           = len(df_treatment),
    alpha                 = st.session_state["alpha"],
)

# ── Metric summary row ────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Control Rate",    f"{results['rate_control']:.1%}")
col2.metric("Treatment Rate",  f"{results['rate_treatment']:.1%}")
col3.metric("Absolute Lift",   f"{results['lift_absolute']:.3f}")
col4.metric("p-value",         f"{results['p_value']:.4f}")

st.divider()

# ── Retention chart + revenue waterfall ───────────────────────────────────────

revenue_results = calculate_revenue_impact(
    rate_control             = results["rate_control"],
    rate_treatment           = results["rate_treatment"],
    aov                      = st.session_state["aov"],
    discount_amount          = st.session_state["discount_amount"],
    n_users                  = st.session_state["n_users"],
    orders_per_retained_user = 3.0,
)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        plot_retention_comparison(results=results, alpha=st.session_state["alpha"]),
        use_container_width=True,
    )
with right:
    st.plotly_chart(
        plot_revenue_waterfall(revenue_results=revenue_results),
        use_container_width=True,
    )

# ── Ship / no-ship verdict ────────────────────────────────────────────────────

if results["significant"] and revenue_results["roi_positive"]:
    st.success("✓ Ship — statistically significant AND revenue-positive.")
elif results["significant"] and not revenue_results["roi_positive"]:
    st.warning("⚠ No ship — significant lift but discount costs more than it earns.")
else:
    st.error("✗ No ship — result is not statistically significant.")

st.divider()

# ── Novelty effect ────────────────────────────────────────────────────────────

st.subheader("Novelty Effect")
df_novelty = apply_novelty_effect(df, base_rate=st.session_state["base_rate"])
st.plotly_chart(plot_novelty_effect(df_novelty), use_container_width=True)

st.divider()

# ── Multi-metric BH correction ────────────────────────────────────────────────
# Run the same proportions test on two additional simulated metrics (reorder rate
# and high-value order flag) drawn from the same df to demonstrate FDR control.

rng = np.random.default_rng(seed=42)

reorder_c = rng.binomial(len(df_control),   st.session_state["base_rate"] * 0.6)
reorder_t = rng.binomial(len(df_treatment), st.session_state["base_rate"] * 0.62)
p_reorder = run_proportions_test(reorder_c, len(df_control), reorder_t, len(df_treatment))["p_value"]

aov_c = rng.binomial(len(df_control),   0.40)
aov_t = rng.binomial(len(df_treatment), 0.41)
p_aov = run_proportions_test(aov_c, len(df_control), aov_t, len(df_treatment))["p_value"]

bh_df = apply_bh_correction(
    p_values     = [results["p_value"], p_reorder, p_aov],
    metric_names = ["retention", "reorder_rate", "avg_order_value"],
    alpha        = st.session_state["alpha"],
)

with st.expander("Multiple metric correction (Benjamini-Hochberg)"):
    st.dataframe(bh_df, use_container_width=True)
