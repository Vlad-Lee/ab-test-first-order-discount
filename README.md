# A/B Test: First Order Discount and 30-Day Retention

## Overview
Simulated end-to-end A/B test evaluating whether a first-order discount 
increases 30-day retention on a food delivery platform. Covers experiment 
design, power analysis, statistical testing, and business impact analysis.

## Business Question
Does offering a $10 discount on a user's first order increase the likelihood 
of placing a second order within 30 days, and does the retention lift justify 
the revenue cost of the discount?

## Experiment Design
- **Randomization unit:** User
- **Treatment:** $10 discount on first order
- **Control:** No discount
- **Primary metric:** 30-day retention rate
- **Secondary metrics:** Average order value, 60-day LTV
- **Guardrail metric:** Revenue per user
- **Significance level:** 5%
- **Statistical power:** 80%

## Methodology
- Power analysis to determine required sample size
- Simulated user-level data with realistic base rates and treatment effects
- AA test validation prior to analysis
- Two-sample test of binomial proportions for primary metric
- Holm-Bonferroni correction across multiple metrics
- Segment analysis for heterogeneous treatment effects
- Novelty effect and sample ratio mismatch checks

## Results
*To be updated upon completion*

## Repository Structure
