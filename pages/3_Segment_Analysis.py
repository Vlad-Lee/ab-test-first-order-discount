import streamlit as st
from src.sidebar import render_sidebar
from src.simulate import generate_base_data, apply_segment_heterogeneity
from src.viz import plot_segment_effects

render_sidebar()

st.title("Segment Analysis")

with st.expander("About this page"):
    st.markdown("""
    **What this page shows:** How the treatment effect varies across user segments, and whether
    a targeted rollout is more efficient than offering the discount to everyone.

    **Heterogeneous treatment effects:** The average treatment effect (ATE) reported on the
    Simulation Results page is a single number that masks variation across users. Price-sensitive
    users respond strongly to the discount. Frequent orderers who would have reordered anyway
    show near-zero lift. Applying the discount to low-response segments wastes margin without
    generating meaningful retention gains.

    **Segments:** Users are binned by price sensitivity (low / med / high) and order frequency
    (low / med / high), producing 9 segment combinations. The lift table shows the treatment minus
    control retention rate for each segment, sorted from highest to lowest lift.

    **Rollout recommendation:** If the top segment's lift is more than 2× the ATE, a targeted
    rollout is recommended. Offer the discount only to high-sensitivity / low-frequency users.
    Otherwise the effects are uniform enough that a full rollout is efficient.
    """)

# ── Generate data ─────────────────────────────────────────────────────────────

@st.cache_data
def cached_generate(n_users, base_rate, treatment_effect, seed=42):
    return generate_base_data(n_users, base_rate, treatment_effect, seed=seed)

df = cached_generate(
    st.session_state["n_users"],
    st.session_state["base_rate"],
    st.session_state["treatment_effect"],
)

# ── Segment heterogeneity ─────────────────────────────────────────────────────

df_seg = apply_segment_heterogeneity(
    df               = df,
    base_rate        = st.session_state["base_rate"],
    treatment_effect = st.session_state["treatment_effect"],
)

st.plotly_chart(plot_segment_effects(df_seg), use_container_width=True)

st.divider()

# ── Lift table ────────────────────────────────────────────────────────────────

sensitivity_labels = {0.5: "Low", 1.0: "Med", 1.8: "High"}
frequency_labels   = {1.5: "Low", 1.0: "Med", 0.4: "High"}

df_seg["segment"] = (
    df_seg["sensitivity_mult"].map(sensitivity_labels) + " Sens / "
    + df_seg["frequency_mult"].map(frequency_labels)   + " Freq"
)

lift_table = (
    df_seg.groupby(["segment", "group"])["retained"]
    .mean()
    .unstack()
    .assign(lift=lambda x: x["treatment"] - x["control"])
    .round(3)
    .sort_values("lift", ascending=False)
)

st.dataframe(
    lift_table.style.background_gradient(subset=["lift"], cmap="RdYlGn"),
    use_container_width=True,
)

st.divider()

# ── Rollout recommendation ────────────────────────────────────────────────────

ate      = lift_table["lift"].mean()
top_lift = lift_table["lift"].max()
top_seg  = lift_table["lift"].idxmax()

if top_lift > 2 * ate:
    st.info(
        f"**Targeted rollout recommended.** Segment '{top_seg}' shows "
        f"{top_lift:.1%} lift vs. ATE of {ate:.1%}. "
        f"Offer discount only to high-sensitivity / low-frequency users."
    )
else:
    st.success("**Full rollout is efficient.** Treatment effects are roughly uniform across segments.")