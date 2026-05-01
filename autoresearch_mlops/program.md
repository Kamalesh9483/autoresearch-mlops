# AUTORESEARCH OPERATING SPECIFICATION

This document defines the autonomous research loop.

------------------------------------------------------------
## 1. OBJECTIVE
------------------------------------------------------------
Find model + hyperparameters that minimize RMSE on regression dataset.

Primary metric:
- RMSE (lower is better)

Target:
- RMSE ≤ 0.10

------------------------------------------------------------
## 2. DATA ASSUMPTIONS
------------------------------------------------------------
- Tabular regression dataset
- No missing values (or handled in preprocessing)
- Train/test split is stable

------------------------------------------------------------
## 3. MODEL SEARCH SPACE
------------------------------------------------------------

Models:
1. Neural Network
   - hidden size: [32, 64, 128]
   - activation: ReLU
   - optimizer: Adam

2. XGBoost
   - learning_rate: [1e-4 → 1e-1]
   - n_estimators: [50 → 500]
   - max_depth: [3 → 10]

------------------------------------------------------------
## 4. SEARCH STRATEGY
------------------------------------------------------------

Phase 1: Exploration
- Random + Bayesian search (Optuna)

Phase 2: Exploitation
- Focus on top 20% configurations

Phase 3: Refinement
- LLM proposes hyperparameter shifts

------------------------------------------------------------
## 5. MEMORY POLICY
------------------------------------------------------------

Store:
- best 50 experiments
- worst 20 failures
- top-performing configurations

Avoid:
- repeating failed configs
- re-testing same LR ranges repeatedly

------------------------------------------------------------
## 6. DECISION RULES
------------------------------------------------------------

Accept improvement if:
- RMSE improves by > 1%

Reject config if:
- unstable training
- divergence
- repeated failure pattern

------------------------------------------------------------
## 7. STOPPING CONDITIONS
------------------------------------------------------------

Stop experiment loop when ANY:

1. RMSE ≤ 0.10 (success)
2. No improvement in 10 trials
3. Max 50 trials reached
4. Search space exhausted

------------------------------------------------------------
## 8. LLM ROLE
------------------------------------------------------------

LLM is allowed to:
- suggest hyperparameter shifts
- detect patterns in failures
- bias search direction

LLM is NOT allowed to:
- override evaluation metrics
- modify dataset