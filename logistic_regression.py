# =============================================================================
# logistic_regression.py
# =============================================================================
# Description:
#   Estimates a logistic regression model to assess the relative contribution
#   of three independent variables — Scenario, Prompt_Type, and Model_Type —
#   in predicting the probability of a successful harmful response
#   (Successful_Response = 1) across 1,000 prompt-response observations.
#
#   Model specification:
#     logit(P(Successful_Response = 1)) =
#       β₀ + β_Scenario · Scenario + β_Prompt_Type · Prompt_Type
#          + β_Model_Type · Model_Type + Σ βᵢXᵢ
#
#   The model is estimated via Maximum Likelihood Estimation (MLE) using
#   the statsmodels Logit implementation. Categorical independent variables
#   (Scenario, Prompt_Type, Model_Type) are encoded using target encoding
#   prior to model estimation, which maps each category to a value derived
#   from its empirical relationship with the outcome variable.
#
#   Target encoding note:
#     Target encoding is applied to the full dataset as the model's purpose
#     is inferential (coefficient estimation and significance testing) rather
#     than predictive (out-of-sample generalisation). The encoding approach
#     is consistent with this inferential aim.
#
#   Key outputs:
#     - Regression coefficients (log-odds) for each predictor
#     - Standard errors and z-scores for each coefficient
#     - P-values indicating statistical significance of each predictor
#     - 95% confidence intervals for each coefficient
#     - Pseudo R-squared (McFadden) as a model fit statistic
#     - Log-likelihood and likelihood ratio test p-value
#
#   Input data:
#     regressionlimit.csv — structured dataset containing one row per
#     prompt-response observation with columns:
#       Test_ID           : unique identifier for each observation
#       Scenario          : harm category (targeting/fraud, misinformation,
#                           radicalisation, malware, weapon instruction)
#       Prompt_Type       : prompt condition (baseline, spam, jailbreak)
#       Model_Type        : model condition (censored Llama3, uncensored
#                           Dolphin Llama3)
#       Successful_Response: binary outcome (1 = harmful response elicited,
#                           0 = model refused or response was non-harmful)
#
#   Raw data not included in this repository given the sensitive nature
#   of the harm categories evaluated. Available from the author upon
#   reasonable request for verified research purposes.
#
# Part of the analysis for:
#   Censored vs. Uncensored: An Empirical Evaluation of Open-Weight Language Models Across Nefarious Use Cases and Regulatory Implications — Alexandra Zini (2026)
#   Available at: [SSRN URL to be updated]
#
# Dependencies:
#   pandas, statsmodels, category_encoders
# =============================================================================

import pandas as pd
import statsmodels.api as sm
import category_encoders as ce

# -----------------------------------------------------------------------------
# Load Data
# Reads the structured prompt-response dataset. Each row represents a single
# prompt administered to a model under a specific scenario and prompt type
# condition, with the binary outcome recorded.
# -----------------------------------------------------------------------------

data = pd.read_csv('regressionlimit.csv')

# Confirm column data types before encoding
print("Column data types:")
print(data.dtypes)
print(f"\nTotal observations: {len(data)}")
print(f"Outcome distribution:\n{data['Successful_Response'].value_counts()}\n")

# -----------------------------------------------------------------------------
# Target Encoding
# Converts categorical variables (Scenario, Prompt_Type, Model_Type) into
# continuous numerical values using target encoding. Each category is mapped
# to a smoothed estimate of the mean outcome (Successful_Response) for that
# category, capturing the empirical relationship between each categorical
# value and the probability of a successful harmful response.
#
# Target encoding is appropriate here as the model is inferential:
# coefficients reflect the predictive direction and magnitude of each
# variable relative to the outcome, not out-of-sample generalisation.
# -----------------------------------------------------------------------------

categorical_columns = ['Scenario', 'Prompt_Type', 'Model_Type']

# Initialise and fit target encoder
encoder = ce.TargetEncoder(cols=categorical_columns)
encoder.fit(data, data['Successful_Response'])

# Apply encoding to categorical columns
data_encoded = encoder.transform(data)

# -----------------------------------------------------------------------------
# Prepare Model Variables
# Removes the identifier column (Test_ID) which carries no predictive
# information, then separates independent variables (X) from the
# dependent variable (y).
# -----------------------------------------------------------------------------

# Remove unique identifier — not a predictor
data_encoded = data_encoded.drop(columns=['Test_ID'])

# Independent variables (predictors)
X = data_encoded.drop(columns=['Successful_Response'])

# Dependent variable (binary outcome)
y = data_encoded['Successful_Response']

# Add constant term to allow intercept estimation
X = sm.add_constant(X)

# Confirm final model matrix structure
print("Model matrix preview (first 5 rows):")
print(X.head())
print(f"\nModel matrix data types:\n{X.dtypes}\n")

# -----------------------------------------------------------------------------
# Logistic Regression Estimation
# Fits the logistic regression model via Maximum Likelihood Estimation.
# maxiter=100 increases the iteration limit beyond the default to ensure
# convergence given the model's specification.
#
# Interpretation of coefficients:
#   Coefficients are in log-odds units. A positive coefficient indicates
#   that an increase in the encoded variable value is associated with a
#   higher probability of a successful harmful response. The magnitude
#   reflects the strength of that association conditional on the other
#   predictors.
# -----------------------------------------------------------------------------

model = sm.Logit(y, X).fit(maxiter=100)

# -----------------------------------------------------------------------------
# Output Full Regression Summary
# Reports coefficients, standard errors, z-scores, p-values, confidence
# intervals, and model fit statistics (Pseudo R-squared, log-likelihood,
# likelihood ratio test).
# -----------------------------------------------------------------------------

print(model.summary())
