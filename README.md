# Satellite Collision Risk Prediction using Machine Learning

---

**Department of Computer Science**  
**JK Lakshmipat University, Jaipur**  
**Course:** CS1138 – Machine Learning Project  
**Year:** March 2026

---

**Authors:**  
- Jharna Saxena (Primary)  
- Nidhish Tripathi (Contributor)  
- Prateek Saxena (Contributor)

**Supervisor:** Dr. Rajlakshmi Nayak

---

## Abstract

There are many active satellites and debris objects on Earth's orbit. Tracking systems generate hundreds of conjunction warnings every week, yet over 98% of these are false alarms. The core problem is that physics-based collision calculation is theoretical and sometimes uncertain, but satellites need at least 48+ hours of lead time to safely execute avoidance maneuvers. By the time the risk estimate becomes accurate, it is often too late to act.

This project uses the European Space Agency's Spacecraft Collision Avoidance Challenge dataset to train machine learning models that predict the final collision risk using early Conjunction Data Messages (CDMs), enabling earlier, more accurate, and automated risk evaluations.

---

## 1. Introduction

Space debris has become a critical hazard for satellites and their operations. After more than six decades of space activities, over 28,000 trackable objects larger than 10cm orbit Earth along with an estimated 900,000 fragments between 1cm and 10cm [18]. The first accidental in-orbit collision (Iridium-33 / Kosmos-2251) in 2009 generated more than 2,300 trackable fragments, and events like the 2007 FengYun-1C anti-satellite test increased the debris population by 25%. Without intervention, the "Kessler syndrome" – a self-sustaining cascade of collisions – could render some orbital regions unusable.

This project uses the European Space Agency's Spacecraft Collision Avoidance Challenge dataset [1] to train machine learning models that predict the final collision risk using early CDM data.

---

## 2. Dataset Description

| Attribute | Value |
|-----------|-------|
| Source | ESA Spacecraft Collision Avoidance Challenge (Kaggle) |
| Training samples | 162,634 rows × 103 columns |
| Test samples | 24,484 rows × 103 columns |
| Target variable | `risk` (log10 of collision probability) |
| High-risk threshold | risk ≥ -6 (ESA operational standard) |
| Class imbalance | 1:15 (only 6.26% high-risk events) |

---

## 3. Methodology

### 3.1 Data Preprocessing

The preprocessing follows literature best practices:

1. **Missing value imputation:** Numerical features filled with median, categorical with mode.
2. **Correlation reduction:** Computed correlation matrix of all numerical features and removed features from any pair with correlation coefficient > 0.95 to reduce multicollinearity. This reduced the feature count from 101 to 76.
3. **Train/validation/test split:** 70% training, 15% validation, 15% testing (stratified by high-risk label).

### 3.2 Model Training

We trained and compared five supervised **classification** models:

| Model | Hyperparameters |
|-------|-----------------|
| Random Forest | 100 trees, max depth 10 |
| XGBoost | 100 boosting rounds, max depth 6, learning rate 0.05 |
| CatBoost | 100 iterations, depth 6 |
| Gradient Boosting (sklearn) | 100 estimators, max depth 5, learning rate 0.1 |
| SVM (RBF kernel) | C=1.0, probability=True |

All models were trained to predict the binary target: high-risk (1) if log10 Pc ≥ -6, else low-risk (0).

### 3.3 Evaluation Metrics

Evaluation uses classification metrics:
- Confusion matrix, Recall, Precision, F1 score
- ROC AUC (Area Under the Receiver Operating Characteristic curve)
- PR AUC (Area Under the Precision-Recall curve)

**Recall is prioritized** because missing a true collision (false negative) is far more harmful than a false alarm.

---

## 4. Results

### 4.1 Best Model: Gradient Boosting

Gradient Boosting outperformed all other models, achieving the highest validation recall (98.95%) and perfect ROC AUC (1.0000).

**Test Set Performance:**

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

### 4.2 Validation Set Comparison

| Model | Accuracy | Recall | ROC AUC |
|-------|----------|--------|---------|
| Gradient Boosting | 99.89% | **98.95%** | 1.0000 |
| XGBoost | 99.81% | 98.62% | 0.9999 |
| CatBoost | 99.81% | 98.56% | 0.9999 |
| Random Forest | 99.04% | 85.85% | 0.9992 |
| SVM | 98.84% | 86.84% | 0.9974 |

### 4.3 Why Gradient Boosting Won

Gradient Boosting builds trees sequentially, each new tree focusing on correcting the errors of all previous trees. This sequential error-correction naturally concentrates on rare, hard-to-predict high-risk events.

### 4.4 Top 5 Features

1. `max_risk_scaling` – Risk scaling factor
2. `max_risk_estimate` – Maximum risk in event history
3. `miss_distance` – Separation at closest approach (m)
4. `mahalanobis_distance` – Statistical distance with uncertainty
5. `time_to_tca` – Hours until closest approach

---

## 5. Visualizations Generated

For each of the 5 models:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance (for tree-based models)

Additional:
- Model Comparison Bar Charts (6 metrics)

---

## 6. Limitations & Future Work
No temporal modelling – used only latest CDM (future: LSTM/Transformers)

Class imbalance (1:15) – handled only by stratified sampling (future: SMOTE, cost-sensitive learning)

No ground-truth maneuver labels – future: synthetic data with COLGen framework

## 7. Conclusion
In summary, gradient boosting with proper validation and classification-focused evaluation delivers state-of-the-art collision risk detection, suitable for real-world early warning systems. The model achieves 98.89% recall with only 17 false negatives, demonstrating its effectiveness for this safety-critical application.

## 8. Acknowledgments
We sincerely thank our mentor, Dr. Rajlakshmi Nayak, for her invaluable guidance, constructive feedback, and continuous support throughout this project. Her insights on proper validation methodology, feature selection, and evaluation metrics helped us avoid data leakage and significantly improve our model's performance.

We also thank the European Space Agency (ESA) for providing the dataset and Kaggle for hosting the competition.
