# =============================================================================
# normal_distribution.py
# =============================================================================
# Description:
#   Visualises the sampling distributions of conversion rates across paired
#   assessment conditions using fitted normal distribution curves. Conversion
#   rates represent the proportion of prompts that successfully induced a
#   harmful response from the model under each assessment condition.
#
#   For each assessment pair, a normal distribution is fitted using the
#   observed conversion rate (p) as the mean and the standard error (SE)
#   as the spread parameter. The 95% confidence interval is shaded to
#   illustrate the range within which the true conversion rate is expected
#   to fall with 95% probability (using the 1.96 multiplier).
#
#   A conversion rate of 0 with SE of 0 (as observed in baseline censored
#   conditions) indicates complete refusal by the model to produce harmful
#   outputs — the comparison baseline against which uncensored and jailbroken
#   conditions are evaluated.
#
# Usage:
#   Adjust p_A, SE_A, p_B, and SE_B to reflect the conversion rate and
#   standard error of the two assessment conditions being compared.
#   Assessment labels in plt.title() and plt.legend() should be updated
#   to reflect the specific pair being visualised.
#
# Part of the analysis for:
#   Censored vs. Uncensored: An Empirical Evaluation of Open-Weight Language Models Across Nefarious Use Cases and Regulatory Implications — Alexandra Zini (2026)
#   Available at: [SSRN URL to be updated]
#
# Dependencies:
#   numpy, matplotlib, scipy
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------------------------------------------------------
# Input Parameters
# Define the conversion rate (p) and standard error (SE) for each assessment
# condition. p represents the proportion of successful harmful responses
# observed; SE represents the standard error of that proportion.
# -----------------------------------------------------------------------------

# Assessment condition A (e.g. censored baseline — Assessment 1.1)
p_A = 0        # Conversion rate: 0 indicates complete model refusal
SE_A = 0       # Standard error: 0 when conversion rate is 0 (no variance)

# Assessment condition B (e.g. uncensored baseline — Assessment 1.3)
p_B = 0.656    # Conversion rate: proportion of successful harmful responses
SE_B = 0.0429  # Standard error of the observed conversion rate

# -----------------------------------------------------------------------------
# Generate X-axis Values
# Produces 1,000 evenly spaced values spanning ±3 standard deviations around
# each condition's mean, providing sufficient range to visualise the full
# distribution curve without truncation.
# -----------------------------------------------------------------------------

x_A = np.linspace(p_A - 3, p_A + 3, 1000)
x_B = np.linspace(p_B - 3 * SE_B, p_B + 3 * SE_B, 1000)

# -----------------------------------------------------------------------------
# Calculate Normal Distributions
# Computes the probability density function (PDF) for each condition using
# the observed conversion rate as the mean and the standard error as the
# standard deviation of the sampling distribution.
# -----------------------------------------------------------------------------

y_A = norm.pdf(x_A, p_A, SE_A)  # PDF for condition A
y_B = norm.pdf(x_B, p_B, SE_B)  # PDF for condition B

# -----------------------------------------------------------------------------
# Plot Distributions
# Plots both distributions and shades the 95% confidence interval for
# condition B (defined as p_B ± 1.96 * SE_B). Condition A is plotted
# for visual comparison; where SE_A = 0, it renders as a vertical line
# at the point of complete refusal (p = 0).
# -----------------------------------------------------------------------------

plt.plot(x_A, y_A, label='Assessment 1.1')  # Update label to match condition
plt.plot(x_B, y_B, label='Assessment 1.3')  # Update label to match condition

# Shade the 95% confidence interval for condition B
plt.fill_between(
    x_B, y_B,
    where=(x_B >= p_B - 1.96 * SE_B) & (x_B <= p_B + 1.96 * SE_B),
    color='gray',
    alpha=0.3,
    label='95% Confidence Interval'
)

# -----------------------------------------------------------------------------
# Labels and Display
# Update plt.title() to reflect the specific assessment pair being compared.
# -----------------------------------------------------------------------------

plt.title('Assessment 1.1 and 1.3 Conversion Rates')  # Update per assessment pair
plt.xlabel('Conversion Rate')
plt.ylabel('Probability Density')
plt.legend()
plt.tight_layout()
plt.show()
