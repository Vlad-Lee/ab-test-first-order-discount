import pandas as pd
import numpy as np


# ── Base Data ─────────────────────────────────────────────────────────────────

def generate_base_data(
    n_users: int = 10_000,
    base_rate: float = 0.27,
    treatment_effect: float = 0.04,
    split: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic user cohort with 50/50 control/treatment split.

    Each user has a retained outcome drawn from a Bernoulli distribution.
    Treatment users are drawn at base_rate + treatment_effect.
    Segment columns (price_sensitivity, order_frequency) are uniform draws
    used by apply_segment_heterogeneity.
    """

    if n_users <= 0:
        raise ValueError("n_users must be greater than zero.")
    if not 0 < base_rate < 1:
        raise ValueError("base_rate must be between 0 and 1.")
    if not 0 < treatment_effect < 1:
        raise ValueError("treatment_effect must be between 0 and 1.")
    if not 0 < split < 1:
        raise ValueError("split must be between 0 and 1.")

    n_treat = int(n_users * split)
    n_control = n_users - n_treat

    rng = np.random.default_rng(seed)

    user_ids = rng.integers(1, 1_000_000, size=n_users)

    # Bernoulli draw for each group - treatment gets the lifted rate
    ret_c = rng.binomial(1, base_rate, size=n_control)
    ret_t = rng.binomial(1, base_rate + treatment_effect, size=n_treat)

    # Align group labels and retention outcomes in the same order
    groups   = ["control"] * n_control + ["treatment"] * n_treat
    retained = np.concatenate([ret_c, ret_t])

    # Segment covariates - binned into low/medium/high by apply_segment_heterogeneity
    price_sensitivity = rng.uniform(0, 1, size=n_users)
    order_frequency   = rng.uniform(0, 1, size=n_users)

    return pd.DataFrame({
        "user_id":           user_ids,
        "group":             groups,
        "retained":          retained,
        "week":              1,
        "price_sensitivity": price_sensitivity,
        "order_frequency":   order_frequency,
    })


# ── Novelty Effect ────────────────────────────────────────────────────────────

def apply_novelty_effect(
    df: pd.DataFrame,
    base_rate: float = 0.27,
    treatment_effect: float = 0.04,
    n_weeks: int = 4,
    decay_rate: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulates novelty effect - treatment retention starts inflated then decays
    toward the true effect over n_weeks. Expands DataFrame to n_users × n_weeks rows."""

    if df.empty:
        raise ValueError("DataFrame is empty. Run generate_base_data first.")
    if not 0 < base_rate < 1:
        raise ValueError("base_rate must be between 0 and 1.")
    if not 0 < treatment_effect < 1:
        raise ValueError("treatment_effect must be between 0 and 1.")
    if not 0 < decay_rate < 1:
        raise ValueError("decay_rate must be between 0 and 1.")
    if n_weeks <= 0:
        raise ValueError("n_weeks must be greater than zero.")

    rng = np.random.default_rng(seed)
    weekly_frames = []

    treatment_mask = df["group"] == "treatment"
    control_mask   = df["group"] == "control"

    for week in range(1, n_weeks + 1):
        week_df = df.copy()
        week_df["week"] = week

        # Decayed treatment probability at this week
        p_t = base_rate + treatment_effect * np.exp(-decay_rate * (week - 1))

        # Redraw retained for each group at this week's probability
        week_df.loc[treatment_mask, "retained"] = rng.binomial(1, p_t, size=treatment_mask.sum())
        week_df.loc[control_mask,   "retained"] = rng.binomial(1, base_rate, size=control_mask.sum())

        weekly_frames.append(week_df)

    return pd.concat(weekly_frames, ignore_index=True)


# ── Contamination ─────────────────────────────────────────────────────────────

def apply_contamination(
    df: pd.DataFrame,
    base_rate: float = 0.27,
    treatment_effect: float = 0.04,
    contamination_rate: float = 0.05,
    partial_effect: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulates leakage - a fraction of control users are exposed to the discount
    and their retention is partially lifted."""

    if df.empty:
        raise ValueError("DataFrame is empty. Run generate_base_data first.")
    if not 0 < base_rate < 1:
        raise ValueError("base_rate must be between 0 and 1.")
    if not 0 < treatment_effect < 1:
        raise ValueError("treatment_effect must be between 0 and 1.")
    if not 0 < contamination_rate < 1:
        raise ValueError("contamination_rate must be between 0 and 1.")
    if not 0 < partial_effect < 1:
        raise ValueError("partial_effect must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    df = df.copy()
    df["contaminated"] = False

    control_idx = df[df["group"] == "control"].index
    n_contaminated = int(len(control_idx) * contamination_rate)

    # Randomly select control users to contaminate
    subset = rng.choice(control_idx, size=n_contaminated, replace=False)
    p_contaminated = base_rate + treatment_effect * partial_effect

    df.loc[subset, "retained"] = rng.binomial(1, p_contaminated, size=n_contaminated)
    df.loc[subset, "contaminated"] = True

    return df


# ── Segment Heterogeneity ─────────────────────────────────────────────────────

def apply_segment_heterogeneity(
    df: pd.DataFrame,
    base_rate: float = 0.27,
    treatment_effect: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    """Varies treatment effect by segment - price-sensitive users respond more to the
    discount, frequent orderers less so. Redraws retained for treatment users only."""

    if df.empty:
        raise ValueError("DataFrame is empty. Run generate_base_data first.")
    if not 0 < base_rate < 1:
        raise ValueError("base_rate must be between 0 and 1.")
    if not 0 < treatment_effect < 1:
        raise ValueError("treatment_effect must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    df = df.copy()

    # Bin price sensitivity and order frequency into low/medium/high multipliers
    df["sensitivity_mult"] = pd.cut(
        df["price_sensitivity"],
        bins=[-np.inf, 0.33, 0.67, np.inf],
        labels=[0.5, 1.0, 1.8],
    ).astype(float)

    df["frequency_mult"] = pd.cut(
        df["order_frequency"],
        bins=[-np.inf, 0.33, 0.67, np.inf],
        labels=[1.5, 1.0, 0.4],
    ).astype(float)

    treatment_mask = df["group"] == "treatment"

    # Compute segment-specific retention probability, clip to valid range
    p_segment = (
        base_rate
        + treatment_effect
        * df.loc[treatment_mask, "sensitivity_mult"]
        * df.loc[treatment_mask, "frequency_mult"]
    ).clip(0, 1)

    df.loc[treatment_mask, "retained"] = rng.binomial(1, p_segment, size=treatment_mask.sum())

    return df