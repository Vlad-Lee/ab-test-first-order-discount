import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import chisquare
from statsmodels.stats.multitest import multipletests


###################################################################################################
def calculate_sample_size(
    base_rate: float = 0.35,
    mde: float = 0.04,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Returns users needed per group to detect the MDE at the given alpha and power."""

    if not 0 < mde < 1:
        raise ValueError("MDE must be between 0 and 1.")
    if base_rate + mde >= 1.0:
        raise ValueError("base_rate + mde must be below 1.")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")
    if not 0 < power < 1:
        raise ValueError("Power must be between 0 and 1.")

    p1 = base_rate
    p2 = base_rate + mde
    p_bar = (p1 + p2) / 2

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    # Two-proportion form: separate variance terms for p1 and p2 under H1
    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p1 - p2) ** 2

    return math.ceil(numerator / denominator)


###################################################################################################
def run_proportions_test(
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> dict:
    """Two-sample z-test of proportions. Returns test statistics and CI on the lift."""

    if n_control == 0 or n_treatment == 0:
        raise ValueError("n_control and n_treatment must be greater than zero.")

    p_c = conversions_control / n_control
    p_t = conversions_treatment / n_treatment

    # Pooled proportion under H0
    pooled_prop = (conversions_control + conversions_treatment) / (n_control + n_treatment)
    pooled_se = np.sqrt(pooled_prop * (1 - pooled_prop) * (1 / n_control + 1 / n_treatment))

    z_stat = (p_t - p_c) / pooled_se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    # Unpooled SE for confidence interval
    se_diff = np.sqrt(
        (p_c * (1 - p_c)) / n_control +
        (p_t * (1 - p_t)) / n_treatment
    )
    z_crit = norm.ppf(1 - alpha / 2)
    ci_low = (p_t - p_c) - z_crit * se_diff
    ci_high = (p_t - p_c) + z_crit * se_diff

    return {
        "rate_control":   p_c,
        "rate_treatment": p_t,
        "lift_absolute":  p_t - p_c,
        "lift_relative":  (p_t - p_c) / p_c,
        "z_stat":         z_stat,
        "p_value":        p_value,
        "ci_low":         ci_low,
        "ci_high":        ci_high,
        "significant":    p_value < alpha,
    }


###################################################################################################
def apply_bh_correction(
    p_values: list[float],
    metric_names: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Applies Benjamini-Hochberg FDR correction across multiple metrics."""

    if len(p_values) != len(metric_names):
        raise ValueError("p_values and metric_names must be the same length.")

    sorted_pairs = sorted(zip(p_values, metric_names))

    m = len(p_values)
    p_sorted, names_sorted = zip(*sorted_pairs)

    reject, p_adjusted, _, _ = multipletests(p_sorted, alpha=alpha, method='fdr_bh')
    bh_thresholds = [(i + 1) / m * alpha for i in range(m)]

    return pd.DataFrame({
        "metric":       names_sorted,
        "p_raw":        p_sorted,
        "p_adjusted":   p_adjusted,
        "bh_threshold": bh_thresholds,
        "significant":  reject,
    })


###################################################################################################
def check_srm(
    n_control: int,
    n_treatment: int,
    expected_split: float = 0.5,
) -> dict:
    """Detects sample ratio mismatch using a chi-squared goodness-of-fit test."""

    if not 0 < expected_split < 1:
        raise ValueError("expected_split must be between 0 and 1.")
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("n_control and n_treatment must be greater than zero.")

    n_total = n_control + n_treatment
    observed = [n_control, n_treatment]
    expected = [n_total * (1 - expected_split), n_total * expected_split]

    chi2_stat, p_value = chisquare(observed, f_exp=expected)

    return {
        "n_control":      n_control,
        "n_treatment":    n_treatment,
        "observed_split": n_treatment / n_total,
        "expected_split": expected_split,
        "chi2_stat":      chi2_stat,
        "p_value":        p_value,
        "srm_detected":   p_value < 0.01,
    }


###################################################################################################
def run_aa_test(
    base_rate: float,
    n_per_group: int,
    n_simulations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Simulates AA tests to verify the false positive rate is calibrated to alpha."""

    if not 0 < base_rate < 1:
        raise ValueError("base_rate must be between 0 and 1.")
    if n_per_group <= 0:
        raise ValueError("n_per_group must be greater than zero.")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be greater than zero.")

    rng = np.random.default_rng(seed)
    p_values = []

    for _ in range(n_simulations):
        x_c = rng.binomial(n_per_group, base_rate)
        x_t = rng.binomial(n_per_group, base_rate)
        result = run_proportions_test(x_c, n_per_group, x_t, n_per_group, alpha=alpha)
        p_values.append(result["p_value"])

    false_positive_rate = np.mean([p < alpha for p in p_values])

    return {
        "n_simulations":       n_simulations,
        "false_positive_rate": false_positive_rate,
        "expected_fpr":        alpha,
        "fpr_in_bounds":       abs(false_positive_rate - alpha) < 0.02,
        "p_values":            p_values,
    }


###################################################################################################
def calculate_revenue_impact(
    rate_control: float,
    rate_treatment: float,
    aov: float,
    discount_amount: float,
    n_users: int,
    orders_per_retained_user: float = 3.0,
) -> dict:
    """Translates the statistical result into net revenue. Connects the experiment to the ship decision."""

    if not 0 < rate_control < 1:
        raise ValueError("rate_control must be between 0 and 1.")
    if not 0 < rate_treatment < 1:
        raise ValueError("rate_treatment must be between 0 and 1.")
    if n_users <= 0:
        raise ValueError("n_users must be greater than zero.")
    if discount_amount <= 0:
        raise ValueError("discount_amount must be greater than zero.")
    if orders_per_retained_user <= 0:
        raise ValueError("orders_per_retained_user must be greater than zero.")

    # Incremental retained users
    ret_users = (rate_treatment - rate_control) * n_users
    # Revenue from follow-on orders (no discount on subsequent orders)
    rev_incr = ret_users * aov * (orders_per_retained_user - 1)
    # Discount cost applied to all retained treatment users
    cost_disc = discount_amount * n_users * rate_treatment
    # Net revenue impact
    net_rev = rev_incr - cost_disc
    # Break-even lift: the retention lift needed for net revenue to equal zero
    breakeven_lift = discount_amount / ((orders_per_retained_user - 1) * aov)

    return {
        "incremental_retained_users":    ret_users,
        "incremental_retention_revenue": rev_incr,
        "discount_cost":                 cost_disc,
        "net_revenue_impact":            net_rev,
        "revenue_per_user_control":      rate_control * aov * orders_per_retained_user,
        "revenue_per_user_treatment":    rate_treatment * aov * orders_per_retained_user - discount_amount,
        "breakeven_lift":                breakeven_lift,
        "roi_positive":                  net_rev > 0,
    }