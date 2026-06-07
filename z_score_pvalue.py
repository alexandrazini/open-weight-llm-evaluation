# =============================================================================
# z_score_pvalue.py
# =============================================================================
# Description:
#   Conducts a two-proportion z-test to determine whether the difference in
#   conversion rates between two assessment conditions is statistically
#   significant. Conversion rates represent the proportion of prompts that
#   successfully induced a harmful response from the model under each
#   condition.
#
#   The test compares a baseline condition (A) against a treatment condition
#   (B) — for example, a censored model under standard prompting versus the
#   same model under jailbreak conditions, or a censored model versus an
#   uncensored model under equivalent prompting.
#
#   Method:
#     A pooled proportion is calculated combining observations from both
#     conditions, which is used to estimate the standard error under the
#     null hypothesis of no difference between conditions. The z-score
#     measures how many standard errors the observed difference deviates
#     from zero. A two-tailed p-value is computed to assess significance
#     in either direction.
#
#   Null hypothesis (H0):
#     The conversion rate of condition A equals the conversion rate of
#     condition B (p_A = p_B). No significant difference exists between
#     the two assessment conditions.
#
#   Alternative hypothesis (H1):
#     The conversion rates of conditions A and B differ significantly
#     (p_A ≠ p_B).
#
#   A p-value below 0.05 indicates sufficient evidence to reject the null
#   hypothesis at the 95% confidence level, concluding that a statistically
#   significant difference in harmful response rates exists between conditions.
#
# Usage:
#   Update conversions_A, visitors_A, conversions_B, and visitors_B to
#   reflect the observed counts for the specific assessment pair being tested.
#   Each assessment pair in the study was tested with n=125 observations.
#
# Part of the analysis for:
#   Censored vs. Uncensored: An Empirical Evaluation of Open-Weight Language Models Across Nefarious Use Cases and Regulatory Implications — Alexandra Zini (2026)
#   Available at: [SSRN URL to be updated]
#
# Dependencies:
#   numpy, scipy
# =============================================================================

import numpy as np
from scipy.stats import norm

# -----------------------------------------------------------------------------
# Input Parameters
# Define the number of successful harmful responses (conversions) and total
# prompts administered (visitors) for each assessment condition.
# Each condition was administered n=125 prompts.
# -----------------------------------------------------------------------------

# Condition A: baseline (e.g. censored model, standard prompt — Assessment 1.1)
conversions_A = 0    # Number of prompts that elicited a harmful response
visitors_A = 125     # Total number of prompts administered

# Condition B: treatment (e.g. censored model, jailbreak prompt — Assessment 1.3)
conversions_B = 82   # Number of prompts that elicited a harmful response
visitors_B = 125     # Total number of prompts administered

# -----------------------------------------------------------------------------
# Conversion Rate Calculation
# Computes the observed proportion of successful harmful responses for
# each condition.
# -----------------------------------------------------------------------------

p_A = conversions_A / visitors_A  # Conversion rate for condition A
p_B = conversions_B / visitors_B  # Conversion rate for condition B

# -----------------------------------------------------------------------------
# Pooled Proportion
# Combines observations from both conditions to estimate the common
# proportion under the null hypothesis that p_A = p_B. Used as the
# basis for standard error calculation.
# -----------------------------------------------------------------------------

p_pooled = (conversions_A + conversions_B) / (visitors_A + visitors_B)

# -----------------------------------------------------------------------------
# Standard Error
# Calculates the standard error of the difference between two proportions
# under the null hypothesis, using the pooled proportion.
# -----------------------------------------------------------------------------

SE = np.sqrt(p_pooled * (1 - p_pooled) * (1 / visitors_A + 1 / visitors_B))

# -----------------------------------------------------------------------------
# Z-Score
# Measures the number of standard errors by which the observed difference
# in conversion rates (p_A - p_B) deviates from zero (the null hypothesis).
# A large absolute z-score indicates the observed difference is unlikely
# to have occurred by chance.
# -----------------------------------------------------------------------------

z = (p_A - p_B) / SE

# -----------------------------------------------------------------------------
# P-Value (Two-Tailed)
# Computes the probability of observing a z-score at least as extreme as
# the one calculated, under the null hypothesis. The two-tailed test
# accounts for significant differences in either direction (p_A > p_B
# or p_A < p_B).
# A p-value < 0.05 indicates statistical significance at the 95% level.
# A p-value < 0.01 indicates statistical significance at the 99% level.
# -----------------------------------------------------------------------------

p_value = 2 * (1 - norm.cdf(abs(z)))

# -----------------------------------------------------------------------------
# Output Results
# -----------------------------------------------------------------------------

print(f"Conversion rate (Condition A): {p_A:.4f}")
print(f"Conversion rate (Condition B): {p_B:.4f}")
print(f"Pooled proportion:             {p_pooled:.4f}")
print(f"Standard error:                {SE:.4f}")
print(f"Z-score:                       {z:.4f}")
print(f"P-value (two-tailed):          {p_value:.6f}")

if p_value < 0.01:
    print("Result: Statistically significant at the 99% confidence level (p < 0.01)")
elif p_value < 0.05:
    print("Result: Statistically significant at the 95% confidence level (p < 0.05)")
else:
    print("Result: Not statistically significant at the 95% confidence level (p >= 0.05)")
