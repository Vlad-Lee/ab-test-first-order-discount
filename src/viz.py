import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.stats import calculate_sample_size
from src.simulate import generate_base_data, apply_novelty_effect, apply_segment_heterogeneity

# ── Color palette ─────────────────────────────────────────────────────────────
# Used consistently across all charts so the viewer doesn't have to re-learn
# what colors mean on each page.
COLOR_CONTROL   = "steelblue"
COLOR_TREATMENT = "mediumseagreen"
COLOR_NEGATIVE  = "crimson"


# ── Page 1 ────────────────────────────────────────────────────────────────────

def plot_power_curve(
    base_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    mde_range: tuple = (0.01, 0.10),
) -> go.Figure:
    """Line chart showing required sample size per group across a range of MDEs.

    Makes the sample size / sensitivity tradeoff tangible — dragging the MDE
    slider down reveals the exponential cost of chasing smaller effects.
    """

    if not 0 < base_rate < 1:
        raise ValueError("base_rate must be between 0 and 1.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if not 0 < power < 1:
        raise ValueError("power must be between 0 and 1.")
    if not (0 < mde_range[0] < mde_range[1] < 1):
        raise ValueError("mde_range must be an increasing tuple within (0, 1).")

    mde_values   = np.linspace(mde_range[0], mde_range[1], 50)
    sample_sizes = [calculate_sample_size(base_rate, mde, alpha, power) for mde in mde_values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mde_values,
        y=sample_sizes,
        mode="lines",
        name="Required n per group",
        line=dict(color=COLOR_CONTROL),
    ))

    fig.update_layout(
        title="Required Sample Size vs. MDE",
        xaxis_title="MDE",
        yaxis_title="Sample Size per Group",
        xaxis=dict(tickformat=".2f"),
    )

    return fig


def plot_duration_tradeoff(
    required_n: int,
    daily_users_range: tuple = (500, 5000),
) -> go.Figure:
    """Line chart showing experiment duration in days as a function of daily traffic.

    Connects sample size to calendar time — makes it visible when an experiment
    is infeasible given the platform's daily new-user volume.
    Reference lines at 14 and 30 days mark typical experiment windows.
    """

    if required_n <= 0:
        raise ValueError("required_n must be greater than zero.")
    if not (0 < daily_users_range[0] < daily_users_range[1]):
        raise ValueError("daily_users_range must be an increasing tuple of positive values.")

    daily_users = np.linspace(daily_users_range[0], daily_users_range[1], 50)
    # Each group receives half the daily traffic under a 50/50 split
    durations   = [required_n / (d * 0.5) for d in daily_users]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_users,
        y=durations,
        mode="lines",
        name="Experiment Duration",
        line=dict(color=COLOR_CONTROL),
    ))

    fig.add_hline(y=14, line_dash="dash", line_color="grey", annotation_text="14 Days")
    fig.add_hline(y=30, line_dash="dash", line_color="grey", annotation_text="30 Days")

    fig.update_layout(
        title="Experiment Duration vs. Daily Traffic",
        xaxis_title="Daily New Users",
        yaxis_title="Days to Complete",
    )

    return fig


# ── Page 2 ────────────────────────────────────────────────────────────────────

def plot_retention_comparison(
    results: dict,
    alpha: float = 0.05,
) -> go.Figure:
    """Bar chart comparing control vs treatment retention rates.

    Treatment bar includes a 95% CI error bar. Annotation shows absolute lift
    and p-value. Bar color is green if significant, grey if not.
    results: output of run_proportions_test().
    """

    if not results:
        raise ValueError("Results dictionary is empty. Run run_proportions_test first.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    ci_half         = (results["ci_high"] - results["ci_low"]) / 2
    treatment_color = COLOR_TREATMENT if results["significant"] else "grey"

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Control"],
        y=[results["rate_control"]],
        name="Control",
        marker_color=COLOR_CONTROL,
    ))

    fig.add_trace(go.Bar(
        x=["Treatment"],
        y=[results["rate_treatment"]],
        name="Treatment",
        marker_color=treatment_color,
        error_y=dict(
            type="data",
            array=[ci_half],
            arrayminus=[ci_half],
            visible=True,
        ),
    ))

    # Lift and p-value — positioned above the top of the error bar
    fig.add_annotation(
        x="Treatment",
        y=results["rate_treatment"] + ci_half + 0.015,
        text=f"Lift: {results['lift_absolute']:.3f} | p={results['p_value']:.4f}",
        showarrow=False,
        font=dict(size=11, color=treatment_color),
    )

    # Significance label above the lift annotation
    fig.add_annotation(
        x="Treatment",
        y=results["rate_treatment"] + ci_half + 0.035,
        text="✓ Significant" if results["significant"] else "✗ Not significant",
        showarrow=False,
        font=dict(size=11, color=treatment_color),
    )

    fig.update_layout(
        title="Retention Rate by Group",
        xaxis_title="Group",
        yaxis_title="Retention Rate",
        yaxis=dict(tickformat=".2f"),
    )

    return fig


def plot_novelty_effect(
    df: pd.DataFrame,
) -> go.Figure:
    """Line chart showing mean retention rate by week for control and treatment.

    Makes novelty decay visible — treatment starts inflated in week 1 then
    converges toward control as the discount loses salience.
    df: output of apply_novelty_effect().
    """

    if df.empty:
        raise ValueError("DataFrame is empty. Run apply_novelty_effect first.")

    grouped = (
        df.groupby(["group", "week"])["retained"]
        .mean()
        .to_frame("avg_retention")
        .reset_index()
    )

    colors = {"control": COLOR_CONTROL, "treatment": COLOR_TREATMENT}

    fig = go.Figure()

    for group in ["control", "treatment"]:
        subset = grouped[grouped["group"] == group]
        fig.add_trace(go.Scatter(
            x=subset["week"],
            y=subset["avg_retention"],
            name=group.capitalize(),
            mode="lines+markers",
            line=dict(color=colors[group]),
        ))

    fig.update_layout(
        title="Average Weekly Retention Rate by Group",
        xaxis_title="Week",
        yaxis_title="Retention Rate",
        xaxis=dict(dtick=1),
        yaxis=dict(tickformat=".2f"),
    )

    return fig


def plot_revenue_waterfall(
    revenue_results: dict,
) -> go.Figure:
    """Waterfall chart decomposing net revenue impact into its components.

    Incremental retention revenue (green) minus discount cost (red) equals
    net revenue impact. A red net bar means no-ship regardless of statistical
    significance.
    revenue_results: output of calculate_revenue_impact().
    """

    if not revenue_results:
        raise ValueError("Dictionary is empty. Run calculate_revenue_impact first.")

    net_color = "mediumseagreen" if revenue_results["roi_positive"] else COLOR_NEGATIVE

    fig = go.Figure(go.Waterfall(
        name="Revenue Impact",
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["Incremental Retention Revenue", "Discount Cost", "Net Revenue"],
        y=[
            revenue_results["incremental_retention_revenue"],
            -revenue_results["discount_cost"],
            0,  # total computed automatically from preceding relative bars
        ],
        connector=dict(line=dict(color="grey", dash="dot")),
        increasing=dict(marker_color="mediumseagreen"),
        decreasing=dict(marker_color=COLOR_NEGATIVE),
        totals=dict(marker_color=net_color),
    ))

    fig.update_layout(
        title="Revenue Impact Breakdown",
        yaxis_title="Amount ($)",
        yaxis=dict(tickformat="$.0f"),
    )

    return fig


# ── Page 3 ────────────────────────────────────────────────────────────────────

def plot_segment_effects(
    df: pd.DataFrame,
) -> go.Figure:
    """Grouped bar chart showing control vs treatment retention by segment.

    Makes heterogeneous treatment effects visible — high price-sensitivity users
    respond strongly to the discount while frequent orderers show near-zero lift.
    Drives the targeted vs. full rollout recommendation on Page 3.
    df: output of apply_segment_heterogeneity().
    """

    if df.empty:
        raise ValueError("DataFrame is empty. Run apply_segment_heterogeneity first.")

    sensitivity_labels = {0.5: "Low Sensitivity", 1.0: "Med Sensitivity", 1.8: "High Sensitivity"}
    frequency_labels   = {1.5: "Low Frequency",   1.0: "Med Frequency",   0.4: "High Frequency"}

    df = df.copy()
    df["segment"] = (
        df["sensitivity_mult"].map(sensitivity_labels)
        + " / "
        + df["frequency_mult"].map(frequency_labels)
    )

    grouped   = df.groupby(["segment", "group"])["retained"].mean().reset_index()
    control   = grouped[grouped["group"] == "control"]
    treatment = grouped[grouped["group"] == "treatment"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=control["segment"],
        y=control["retained"],
        name="Control",
        marker_color=COLOR_CONTROL,
    ))

    fig.add_trace(go.Bar(
        x=treatment["segment"],
        y=treatment["retained"],
        name="Treatment",
        marker_color=COLOR_TREATMENT,
    ))

    fig.update_layout(
        title="Retention Rate by Segment",
        xaxis_title="Segment",
        yaxis_title="Average Retention Rate",
        yaxis=dict(tickformat=".2f"),
        barmode="group",
        showlegend=True,
    )

    return fig