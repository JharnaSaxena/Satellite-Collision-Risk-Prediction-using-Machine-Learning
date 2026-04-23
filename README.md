# Satellite Collision Risk Prediction using Machine Learning

## Project Overview
This project predicts satellite collision risk using Conjunction Data Messages (CDMs) from the European Space Agency's Spacecraft Collision Avoidance Challenge dataset. The model classifies events as **high-risk** (log10 Pc ≥ -6) or **low-risk**, enabling earlier and more accurate collision avoidance decisions.

## Problem Statement
- Tracking systems generate hundreds of conjunction warnings weekly, over 98% are false alarms
- Satellites need at least 48+ hours lead time for avoidance maneuvers
- By the time risk estimates become accurate, it is often too late to act
- One collision can trigger Kessler Syndrome – a cascading chain reaction

## Dataset
| Attribute | Value |
|-----------|-------|
| Source | ESA Spacecraft Collision Avoidance Challenge (Kaggle) |
| Training samples | 162,634 rows × 103 columns |
| Test samples | 24,484 rows × 103 columns |
| Target variable | `risk` (log10 of collision probability) |
| High-risk threshold | risk ≥ -6 (ESA operational standard) |
| Class imbalance | 1:15 (only 6.26% high-risk events) |

## Models Trained (5 Classifiers)
| Model | Hyperparameters |
|-------|-----------------|
| Random Forest | 100 trees, max depth 10 |
| XGBoost | 100 rounds, max depth 6, learning rate 0.05 |
| CatBoost | 100 iterations, depth 6 |
| Gradient Boosting | 100 estimators, max depth 5, learning rate 0.1 |
| SVM | RBF kernel, C=1.0, probability=True |

## Preprocessing Steps
1. Missing value imputation (median for numeric, mode for categorical)
2. Removed 25 highly correlated features (|r| > 0.95) – 101 → 76 numeric features
3. Log1p transformation for extreme features (`max_risk_scaling`, covariance determinants)
4. Stratified train/validation/test split: 70/15/15
5. One-hot encoding for categorical feature (`c_object_type`)
6. Standard scaling for SVM

## Results (Gradient Boosting – Best Model)

### Test Set Performance
| Metric | Value |
|--------|-------|
| Accuracy | 99.85% |
| Recall | 98.89% (1511/1528 high-risk detected) |
| Precision | 98.76% |
| F1 Score | 0.9882 |
| ROC AUC | 1.0000 |
| PR AUC | 0.9994 |
| False Negatives | 17 (missed dangerous events) |
| False Positives | 19 (unnecessary alarms) |

### Validation Set Comparison
| Model | Accuracy | Recall | ROC AUC |
|-------|----------|--------|---------|
| Gradient Boosting | 99.89% | **98.95%** | 1.0000 |
| XGBoost | 99.81% | 98.62% | 0.9999 |
| CatBoost | 99.81% | 98.56% | 0.9999 |
| Random Forest | 99.04% | 85.85% | 0.9992 |
| SVM | 98.84% | 86.84% | 0.9974 |

### Why Gradient Boosting Won
Gradient Boosting builds trees sequentially, each new tree correcting errors of previous trees. This sequential error-correction naturally focuses on rare, hard-to-predict high-risk events.

### Top 5 Features
1. `max_risk_scaling` – Risk scaling factor
2. `max_risk_estimate` – Maximum risk in event history
3. `miss_distance` – Separation at closest approach (m)
4. `mahalanobis_distance` – Statistical distance with uncertainty
5. `time_to_tca` – Hours until closest approach

## Visualizations Generated
For each of the 5 models:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance (for tree-based models)

Additional:
- Model Comparison Bar Charts (6 metrics)

Acknowledgments
We sincerely thank our mentor, Dr. Rajlakshmi Nayak, for her invaluable guidance, constructive feedback, and continuous support throughout this project. Her insights on proper validation methodology, feature selection, and evaluation metrics helped us avoid data leakage and significantly improve our model's performance.

We also thank the European Space Agency (ESA) for providing the dataset and Kaggle for hosting the competition.

Authors
Jharna Saxena (Primary Author)

Nidhish Tripathi (Contributor)

Prateek Saxena (Contributor)

Mentor
Dr. Rajlakshmi Nayak
Department of Computer Science
JK Lakshmipat University, Jaipur

Course
CS1138 – Machine Learning Project | Year: March 2026

License
All Rights Reserved – For academic submission only.
