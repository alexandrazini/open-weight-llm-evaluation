# open-weight-llm-evaluation

Python analysis scripts for: Censored vs. Uncensored: An Empirical Evaluation of Open-Weight Language Models Across Nefarious Use Cases and Regulatory Implications

## Scripts

- `normal_distribution.py` — fits normal distribution curves to 
  conversion rates across assessment conditions
- `z_score_pvalue.py` — two-proportion z-tests comparing conversion 
  rates between paired assessment conditions  
- `logistic_regression.py` — logistic regression model estimating 
  predictors of successful harm induction across 1,000 observations

## Requirements

See `requirements.txt`

## Data

Raw response data is not included in this repository given the 
sensitive nature of the harm categories evaluated. The complete 
dataset is available from the author upon reasonable request 
for verified research purposes.
