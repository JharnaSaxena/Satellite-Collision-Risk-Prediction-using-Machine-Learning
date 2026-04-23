import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             recall_score, precision_score, f1_score)
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
# ==========================================
# 0. LOAD DATA (UPDATE PATH HERE)
# ==========================================
dataset_path = r"C:\Users\Lab\Desktop\ML_JS_PS_NT"
print("=" * 80)
print("SATELLITE COLLISION RISK PREDICTION - RESEARCH PROJECT")
print("=" * 80)
train_data=pd.read_csv(os.path.join(dataset_path, "train_data.csv"))
test_data=pd.read_csv(os.path.join(dataset_path, "test_data.csv"))
print(f"\nTraining data: {train_data.shape[0]:,} rows × {train_data.shape[1]} columns")
print(f"Test data: {test_data.shape[0]:,} rows × {test_data.shape[1]} columns")

# 1. DATASET OVERVIEW

print("\n" + "=" * 60)
print("1. DATA OVERVIEW")
print("=" * 60)
print("\nFirst 3 rows:")
print(train_data.head(3))
print("\nLast 3 rows:")
print(train_data.tail(3))
print("\nColumn names (first 20):")
for i, col in enumerate(train_data.columns[:], 1):
    print(f"{i:3d}. {col}")
print("\nData types:")
print(train_data.dtypes.value_counts())
print("\nBasic statistics (first 5 numeric columns):")
print(train_data.describe().iloc[:, :5])

# 2. TARGET VARIABLE ANALYSIS

print("\n" + "=" * 60)
print("2. TARGET VARIABLE ANALYSIS")
print("=" * 60)
target_col="risk" #chose this because of literature review articles
X=train_data.drop(columns=[target_col])
y=train_data[target_col].copy()
X_test=test_data.drop(columns=[target_col]) if target_col in test_data.columns else test_data.copy()
print(f"\nTarget column: '{target_col}' (log₁₀ of collision probability)")
print("\nTarget column statistics:")
print(y.describe())
high_risk_thresh=-6
high_risk_count=(y >= high_risk_thresh).sum()
low_risk_count=(y < high_risk_thresh).sum()
print(f"\nHigh-risk events (risk ≥ {high_risk_thresh}): {high_risk_count} ({100*high_risk_count/len(y):.2f}%)")
print(f"Low-risk events: {low_risk_count} ({100*low_risk_count/len(y):.2f}%)")
print(f"Imbalance ratio: 1:{low_risk_count/high_risk_count:.1f}")
# GRAPH OF RISK DISTRIBUTION WITH THRESHOLD OF -6
plt.figure(figsize=(8,5))
plt.hist(y, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=-6, color='red', linestyle='--', linewidth=2, label='Threshold = -6')
plt.xlabel('Risk (log10 scale)', fontweight='bold')
plt.ylabel('Number of Events', fontweight='bold')
plt.title('Distribution of Collision Risk', fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
# GRAPH OF SEVERITY CATEGORIES ACCORIDNG TO DIFFERENT CLASSES PRESENT
risk_ranges = {
    "Extremely Safe\n(< -10)":(y<-10).sum(),
    "Very Safe\n(-10 to -8)":((y>=-10)&(y < -8)).sum(),
    "Safe\n(-8 to -6)":((y >=-8)&(y<-6)).sum(),
    "Monitor\n(-6 to -4)":((y>=-6)&(y<-4)).sum(),
    "HIGH RISK\n(≥ -4)":(y>=-4).sum()
}
plt.figure(figsize=(12, 6))
colors=['darkgreen','green','yellow','orange','red']
plt.bar(risk_ranges.keys(), risk_ranges.values(),color=colors, edgecolor='black', alpha=0.8)
plt.ylabel('Number of Events', fontweight='bold')
plt.title('Risk Distribution by Severity Category',fontweight='bold')
plt.xticks(rotation=0)
plt.grid(axis='y',alpha=0.3)
for i, (cat, cnt) in enumerate(risk_ranges.items()):
    plt.text(i,cnt+ 500,f'{cnt:,}',ha='center',fontweight='bold')
plt.tight_layout()
plt.show()

# 3. HANDLING MISSING VALUES

print("\n" + "=" * 60)
print("3. HANDLING MISSING VALUES")
print("=" * 60)

missing_X=X.isnull().sum()
missing_X=missing_X[missing_X > 0].sort_values(ascending=False)
if len(missing_X)>0:
    print(f"\nColumns with missing values in training features: {len(missing_X)}")
    print("\nTop 20 features with most missing values:")
    print(missing_X.head(20))
else:
    print("No missing values in training features.")
missing_X_test=X_test.isnull().sum()
missing_X_test=missing_X_test[missing_X_test>0].sort_values(ascending=False)
if len(missing_X_test)>0:
    print(f"\nColumns with missing values in test features: {len(missing_X_test)}")
    print(missing_X_test.head(20))
numeric_cols=X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols=X.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"\nNumeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
# Impute missing values
if numeric_cols:
    num_imputer=SimpleImputer(strategy='median')
    X[numeric_cols]=num_imputer.fit_transform(X[numeric_cols])
    X_test[numeric_cols]=num_imputer.transform(X_test[numeric_cols])
    print("Numeric columns imputed with median.")
if categorical_cols:
    cat_imputer=SimpleImputer(strategy='most_frequent')
    X[categorical_cols]=cat_imputer.fit_transform(X[categorical_cols])
    X_test[categorical_cols]=cat_imputer.transform(X_test[categorical_cols])
    print("Categorical columns imputed with mode.")
print(f"\nMissing values after imputation: Train = {X.isnull().sum().sum()}, Test = {X_test.isnull().sum().sum()}")
# 4. CORRELATION ANALYSIS & FEATURE REDUCTION
print("\n"+"="*60)
print("4. CORRELATION ANALYSIS & FEATURE REDUCTION")
print("="*60)
print("\nCalculating correlation matrix...")
corr_matrix=X[numeric_cols].corr().abs()
plt.figure(figsize=(22,18))
sns.heatmap(corr_matrix,annot=False,cmap='coolwarm',center=0,
            cbar_kws={'label':'Absolute Correlation'},
            linewidths=0.1,linecolor='gray')
plt.title('Correlation Matrix - BEFORE Feature Removal\n(All 101 Numeric Features)',
          fontweight='bold',fontsize=18,pad=20)
plt.xlabel('Features',fontweight='bold')
plt.ylabel('Features',fontweight='bold')
plt.tight_layout()
plt.show()
print(f"Dimensions: {corr_matrix.shape[0]} x {corr_matrix.shape[1]} features")
# Identify highly correlated pairs (|r| > 0.95)
upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
to_drop=[col for col in upper.columns if any(upper[col]>0.95)]
print(f"\nFound {len(to_drop)} features with correlation >0.95.")
print("\nFeatures to be removed:")
print("="*80)
for i,col in enumerate(to_drop,1):
    correlated_with=upper[col][upper[col]>0.95]
    if len(correlated_with)>0:
        partner=correlated_with.index[0]
        corr_val=correlated_with.iloc[0]
        print(f"{i:2d}. {col:<40} (r={corr_val:.4f} with {partner})")
# Remove highly correlated features
if to_drop:
    print("\n"+"="*60)
    print("REMOVING HIGHLY CORRELATED FEATURES")
    print("="*60)
    X=X.drop(columns=to_drop)
    X_test=X_test.drop(columns=to_drop,errors='ignore')
    print(f"Removed: {len(to_drop)} features")
    print(f"Remaining numeric features: {X.select_dtypes(include=[np.number]).shape[1]}")
    print(f"Total features: {X.shape[1]} (including categorical)")
    # Heatmap after removal
    corr_matrix_after=X.select_dtypes(include=[np.number]).corr().abs()
    plt.figure(figsize=(20,16))
    sns.heatmap(corr_matrix_after,annot=False,cmap='coolwarm',center=0,
                cbar_kws={'label':'Absolute Correlation'},
                linewidths=0.1,linecolor='gray')
    plt.title('Correlation Matrix - AFTER Feature Removal\n(Remaining Numeric Features)',
              fontweight='bold',fontsize=18,pad=20)
    plt.xlabel('Features',fontweight='bold')
    plt.ylabel('Features',fontweight='bold')
    plt.tight_layout()
    plt.show()
    print(f"Dimensions: {corr_matrix_after.shape[0]} x {corr_matrix_after.shape[1]} features")
    # Summary statistics after removal
    upper_after=corr_matrix_after.where(np.triu(np.ones(corr_matrix_after.shape),k=1).astype(bool))
    correlations_after=upper_after.values.flatten()
    correlations_after=correlations_after[~np.isnan(correlations_after)]
    print("\n"+"="*60)
    print("CORRELATION SUMMARY (After Removal)")
    print("="*60)
    print(f"Maximum correlation: {correlations_after.max():.4f}")
    print(f"Mean correlation:    {correlations_after.mean():.4f}")
    print(f"Median correlation:  {np.median(correlations_after):.4f}")
    print(f"Features with |r|>0.90: {(correlations_after>0.90).sum()}")
    print(f"Features with |r|>0.80: {(correlations_after>0.80).sum()}")
else:
    print("No features to drop.")
print("\n"+"="*60)
print("CORRELATION ANALYSIS COMPLETE")
print(f"\nFinal dataset:")
print(f"Features before: 103 (101 numeric + categorical + target)")
print(f"Features removed: {len(to_drop)}")
print(f"Features after: {X.shape[1]}")

# 5. FEATURE DICTIONARY

print("\n" + "=" * 80)
print("5. COMPREHENSIVE FEATURE DICTIONARY")
print("=" * 80)
print("""
This section explains every column in the dataset, organized by functional groups.

Note: t_ = Target satellite (ESA satellite being protected)
      c_ = Chaser object (approaching debris/satellite)

================================================================================
IDENTIFIERS & METADATA
================================================================================
  event_id                       - Unique identifier for each conjunction event
  mission_id                     - ESA satellite mission identifier
  time_to_tca                    - Hours until Time of Closest Approach
  risk                           - TARGET: Log10 of collision probability
  max_risk_estimate              - Maximum collision risk in event history
  max_risk_scaling               - Risk scaling factor for normalization

================================================================================
GEOMETRIC FEATURES
================================================================================
  miss_distance                  - Separation distance at closest approach (meters)
  relative_speed                 - Relative velocity between objects (m/s)
  mahalanobis_distance           - Statistical distance considering position uncertainties
  relative_position_r            - Relative position in radial direction (meters)
  relative_position_t            - Relative position in tangential direction (meters)
  relative_position_n            - Relative position in normal direction (meters)
  relative_velocity_r            - Relative velocity in radial direction (m/s)
  relative_velocity_t            - Relative velocity in tangential direction (m/s)
  relative_velocity_n            - Relative velocity in normal direction (m/s)
  geocentric_latitude            - Latitude at closest approach (degrees)
  azimuth                        - Horizontal angle from observer (degrees)
  elevation                      - Vertical angle from horizon (degrees)

================================================================================
TARGET SATELLITE - TRACKING QUALITY
================================================================================
  t_time_lastob_start            - Start time of last observation (days from epoch)
  t_time_lastob_end              - End time of last observation (days from epoch)
  t_recommended_od_span          - Recommended orbit determination span (days)
  t_actual_od_span               - Actual orbit determination span used (days)
  t_obs_available                - Total number of observations available
  t_obs_used                     - Number of observations used in orbit determination
  t_residuals_accepted           - Percentage of residuals accepted (%)
  t_weighted_rms                 - Root mean square of weighted residuals

================================================================================
TARGET SATELLITE - PHYSICAL PROPERTIES
================================================================================
  t_rcs_estimate                 - Radar Cross Section estimate (m2) - object size indicator
  t_cd_area_over_mass            - Drag coefficient times area divided by mass
  t_cr_area_over_mass            - Solar radiation coefficient times area divided by mass
  t_sedr                         - Solar Exclusion Diameter Ratio
  t_span                         - Satellite size span (meters)
  t_h_apo                        - Apogee height - highest orbit point (km)
  t_h_per                        - Perigee height - lowest orbit point (km)

================================================================================
TARGET SATELLITE - ORBITAL ELEMENTS
================================================================================
  t_j2k_sma                      - Semi-major axis - orbit size (km)
  t_j2k_ecc                      - Eccentricity - orbit shape (0=circular)
  t_j2k_inc                      - Inclination - orbit tilt relative to equator (degrees)

================================================================================
TARGET SATELLITE - POSITION UNCERTAINTY (SIGMA)
================================================================================
  t_sigma_r                      - Position uncertainty in radial direction (meters)
  t_sigma_t                      - Position uncertainty in tangential direction (meters)
  t_sigma_n                      - Position uncertainty in normal direction (meters)
  t_sigma_rdot                   - Velocity uncertainty in radial direction (m/s)
  t_sigma_tdot                   - Velocity uncertainty in tangential direction (m/s)
  t_sigma_ndot                   - Velocity uncertainty in normal direction (m/s)

================================================================================
TARGET SATELLITE - COVARIANCE MATRIX ELEMENTS
================================================================================
  t_ct_r                         - Covariance: tangential position times radial position
  t_cn_r                         - Covariance: normal position times radial position
  t_cn_t                         - Covariance: normal position times tangential position
  t_crdot_r                      - Covariance: radial velocity times radial position
  t_crdot_t                      - Covariance: radial velocity times tangential position
  t_crdot_n                      - Covariance: radial velocity times normal position
  t_ctdot_r                      - Covariance: tangential velocity times radial position
  t_ctdot_t                      - Covariance: tangential velocity times tangential position
  t_ctdot_n                      - Covariance: tangential velocity times normal position
  t_ctdot_rdot                   - Covariance: tangential velocity times radial velocity
  t_cndot_r                      - Covariance: normal velocity times radial position
  t_cndot_t                      - Covariance: normal velocity times tangential position
  t_cndot_n                      - Covariance: normal velocity times normal position
  t_cndot_rdot                   - Covariance: normal velocity times radial velocity
  t_cndot_tdot                   - Covariance: normal velocity times tangential velocity

================================================================================
CHASER OBJECT - TYPE
================================================================================
  c_object_type                  - Object classification (PAYLOAD/DEBRIS/ROCKET BODY)

================================================================================
CHASER OBJECT - TRACKING QUALITY
================================================================================
  c_time_lastob_start            - Start time of last observation (days from epoch)
  c_time_lastob_end              - End time of last observation (days from epoch)
  c_recommended_od_span          - Recommended orbit determination span (days)
  c_actual_od_span               - Actual orbit determination span used (days)
  c_obs_available                - Total number of observations available
  c_obs_used                     - Number of observations used in orbit determination
  c_residuals_accepted           - Percentage of residuals accepted (%)
  c_weighted_rms                 - Root mean square of weighted residuals

================================================================================
CHASER OBJECT - PHYSICAL PROPERTIES
================================================================================
  c_rcs_estimate                 - Radar Cross Section estimate (m2) - object size indicator
  c_cd_area_over_mass            - Drag coefficient times area divided by mass
  c_cr_area_over_mass            - Solar radiation coefficient times area divided by mass
  c_sedr                         - Solar Exclusion Diameter Ratio
  c_span                         - Object size span (meters)
  c_h_apo                        - Apogee height - highest orbit point (km)
  c_h_per                        - Perigee height - lowest orbit point (km)

================================================================================
CHASER OBJECT - ORBITAL ELEMENTS
================================================================================
  c_j2k_sma                      - Semi-major axis - orbit size (km)
  c_j2k_ecc                      - Eccentricity - orbit shape (0=circular)
  c_j2k_inc                      - Inclination - orbit tilt relative to equator (degrees)

================================================================================
CHASER OBJECT - POSITION UNCERTAINTY (SIGMA)
================================================================================
  c_sigma_r                      - Position uncertainty in radial direction (meters)
  c_sigma_t                      - Position uncertainty in tangential direction (meters)
  c_sigma_n                      - Position uncertainty in normal direction (meters)
  c_sigma_rdot                   - Velocity uncertainty in radial direction (m/s)
  c_sigma_tdot                   - Velocity uncertainty in tangential direction (m/s)
  c_sigma_ndot                   - Velocity uncertainty in normal direction (m/s)

================================================================================
CHASER OBJECT - COVARIANCE MATRIX ELEMENTS
================================================================================
  c_ct_r                         - Covariance: tangential position times radial position
  c_cn_r                         - Covariance: normal position times radial position
  c_cn_t                         - Covariance: normal position times tangential position
  c_crdot_r                      - Covariance: radial velocity times radial position
  c_crdot_t                      - Covariance: radial velocity times tangential position
  c_crdot_n                      - Covariance: radial velocity times normal position
  c_ctdot_r                      - Covariance: tangential velocity times radial position
  c_ctdot_t                      - Covariance: tangential velocity times tangential position
  c_ctdot_n                      - Covariance: tangential velocity times normal position
  c_ctdot_rdot                   - Covariance: tangential velocity times radial velocity
  c_cndot_r                      - Covariance: normal velocity times radial position
  c_cndot_t                      - Covariance: normal velocity times tangential position
  c_cndot_n                      - Covariance: normal velocity times normal position
  c_cndot_rdot                   - Covariance: normal velocity times radial velocity
  c_cndot_tdot                   - Covariance: normal velocity times tangential velocity

================================================================================
UNCERTAINTY METRICS
================================================================================
  t_position_covariance_det      - Determinant of target position covariance (uncertainty volume)
  c_position_covariance_det      - Determinant of chaser position covariance (uncertainty volume)

================================================================================
SPACE WEATHER CONDITIONS
================================================================================
  F10                            - Solar radio flux at 10.7 cm wavelength
  F3M                            - 3-month average of F10.7 solar flux
  SSN                            - Sunspot Number (solar activity indicator)
  AP                             - Geomagnetic activity index (planetary A-index)

================================================================================
LEGEND:
  t_ prefix = Target satellite (ESA satellite being protected)
  c_ prefix = Chaser object (approaching debris/satellite)
  Target variable is 'risk' (log10 of collision probability)
""")
# ==========================================
# 6. DATA SPLITTING & ADDITIONAL PREPROCESSING
# ==========================================
print("\n" + "=" * 80)
print("6. DATA SPLITTING & ADDITIONAL PREPROCESSING")
print("=" * 80)

# Convert target to binary classification (high-risk = risk >= -6)
y_binary = (y >= -6).astype(int)
print(f"Binary target: High-risk (1) = {y_binary.sum():,} ({y_binary.mean()*100:.2f}%), Low-risk (0) = {(1-y_binary).sum():,}")

# Stratified split based on binary label
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_binary, test_size=0.30, random_state=42, stratify=y_binary
)
# y_temp is already binary, but we need stratify again
X_val, X_test_split, y_val, y_test_split = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Data split:")
print(f"  Training:   {X_train.shape[0]:>7,} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Validation: {X_val.shape[0]:>7,} ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"  Test:       {X_test_split.shape[0]:>7,} ({X_test_split.shape[0]/len(X)*100:.1f}%)")

# Log1p transformation for extreme features (still useful for classification)
extreme_features = ['max_risk_scaling', 't_position_covariance_det', 'c_position_covariance_det']
for feat in extreme_features:
    if feat in X_train.columns:
        shift = 0
        if X_train[feat].min() < 0:
            shift = -X_train[feat].min() + 1
        X_train[feat] = np.log1p(X_train[feat] + shift)
        X_val[feat] = np.log1p(X_val[feat] + shift)
        X_test_split[feat] = np.log1p(X_test_split[feat] + shift)
        X_test[feat] = np.log1p(X_test[feat] + shift)
        print(f"Transformed {feat} (shift={shift:.2f})")

# ==========================================
# HANDLE CATEGORICAL FEATURE
# ==========================================
categorical_feature = 'c_object_type'
categorical_present = categorical_feature in X_train.columns

if categorical_present:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_train = ohe.fit_transform(X_train[[categorical_feature]])
    cat_val = ohe.transform(X_val[[categorical_feature]])
    cat_test = ohe.transform(X_test_split[[categorical_feature]])
    cat_final = ohe.transform(X_test[[categorical_feature]])

    X_train_enc = X_train.drop(columns=[categorical_feature]).values
    X_val_enc = X_val.drop(columns=[categorical_feature]).values
    X_test_enc = X_test_split.drop(columns=[categorical_feature]).values
    X_test_final_enc = X_test.drop(columns=[categorical_feature]).values

    X_train_enc = np.hstack([X_train_enc, cat_train])
    X_val_enc = np.hstack([X_val_enc, cat_val])
    X_test_enc = np.hstack([X_test_enc, cat_test])
    X_test_final_enc = np.hstack([X_test_final_enc, cat_final])

    # For XGBoost and CatBoost native categorical
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    X_test_xgb = X_test_split.copy()
    X_test_final_xgb = X_test.copy()
    X_train_xgb[categorical_feature] = X_train_xgb[categorical_feature].astype('category')
    X_val_xgb[categorical_feature] = X_val_xgb[categorical_feature].astype('category')
    X_test_xgb[categorical_feature] = X_test_xgb[categorical_feature].astype('category')
    X_test_final_xgb[categorical_feature] = X_test_final_xgb[categorical_feature].astype('category')

    X_train_cb = X_train.copy()
    X_val_cb = X_val.copy()
    X_test_cb = X_test_split.copy()
    X_test_final_cb = X_test.copy()
    cat_indices = [X_train_cb.columns.get_loc(categorical_feature)]

    print("Categorical feature processed for all models")
else:
    X_train_enc = X_train.values
    X_val_enc = X_val.values
    X_test_enc = X_test_split.values
    X_test_final_enc = X_test.values
    X_train_xgb = X_train
    X_val_xgb = X_val
    X_test_xgb = X_test_split
    X_test_final_xgb = X_test
    X_train_cb = X_train
    X_val_cb = X_val
    X_test_cb = X_test_split
    X_test_final_cb = X_test
    cat_indices = None

# Scale features for SVM
scaler_svm = StandardScaler()
X_train_svm = scaler_svm.fit_transform(X_train_enc)
X_val_svm = scaler_svm.transform(X_val_enc)
X_test_svm = scaler_svm.transform(X_test_enc)
X_test_final_svm = scaler_svm.transform(X_test_final_enc)

# ==========================================
# 7. MODEL TRAINING & VALIDATION (CLASSIFIERS)
# ==========================================
print("\n" + "=" * 80)
print("7. MODEL TRAINING & VALIDATION (CLASSIFIERS)")
print("=" * 80)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC

models = {}
val_results = {}
training_times = {}

all_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, enable_categorical=True, verbosity=0, use_label_encoder=False, eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(iterations=100, depth=6, learning_rate=0.05, random_seed=42, verbose=False, cat_features=cat_indices if cat_indices else None),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42)  # probability=True for ROC/PR
}

for name, model in all_models.items():
    print(f"\nTraining {name}...")
    start = time.time()
    
    if name == 'SVM':
        model.fit(X_train_svm, y_train)
        y_pred_val = model.predict(X_val_svm)
        y_proba_val = model.predict_proba(X_val_svm)[:, 1]  # probabilities for ROC/PR
    elif name in ['XGBoost', 'CatBoost']:
        if name == 'XGBoost':
            model.fit(X_train_xgb, y_train)
            y_pred_val = model.predict(X_val_xgb)
            y_proba_val = model.predict_proba(X_val_xgb)[:, 1]
        else:
            model.fit(X_train_cb, y_train)
            y_pred_val = model.predict(X_val_cb)
            y_proba_val = model.predict_proba(X_val_cb)[:, 1]
    else:
        model.fit(X_train_enc, y_train)
        y_pred_val = model.predict(X_val_enc)
        y_proba_val = model.predict_proba(X_val_enc)[:, 1]
    
    training_times[name] = time.time() - start
    models[name] = model
    val_results[name] = (y_pred_val, y_proba_val)
    
    # Classification metrics on validation
    acc = accuracy_score(y_val, y_pred_val)
    recall = recall_score(y_val, y_pred_val)
    print(f"  Validation Accuracy: {acc:.4f}, Recall: {recall:.4f}")

# ==========================================
# 8. MODEL SELECTION (Using VALIDATION Recall)
# ==========================================
print("\n" + "=" * 80)
print("8. MODEL SELECTION (Using VALIDATION Recall)")
print("=" * 80)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score

eval_table = []
for name, (y_pred_val, y_proba_val) in val_results.items():
    acc = accuracy_score(y_val, y_pred_val)
    recall = recall_score(y_val, y_pred_val)
    precision = precision_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val)
    roc_auc = roc_auc_score(y_val, y_proba_val)
    pr_auc = average_precision_score(y_val, y_proba_val)
    
    eval_table.append({
        'Model': name,
        'Accuracy': acc,
        'Recall': recall,
        'Precision': precision,
        'F1': f1,
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc,
        'Time (s)': training_times[name]
    })

val_comparison_df = pd.DataFrame(eval_table).round(4)
print("\nVALIDATION SET COMPARISON TABLE")
print("=" * 80)
print(val_comparison_df.to_string(index=False))

best_model_name = val_comparison_df.sort_values('Recall', ascending=False).iloc[0]['Model']
best_val_recall = val_comparison_df[val_comparison_df['Model'] == best_model_name]['Recall'].values[0]
print(f"\n🏆 Best model selected: {best_model_name} (Validation Recall = {best_val_recall:.4f})")

# ==========================================
# 9. FINAL EVALUATION ON TEST SET (ALL MODELS)
# ==========================================
print("\n" + "=" * 80)
print("9. FINAL EVALUATION ON TEST SET (ALL MODELS FOR COMPARISON)")
print("=" * 80)

test_results = {}
for name, model in models.items():
    if name == 'SVM':
        y_pred_test = model.predict(X_test_svm)
        y_proba_test = model.predict_proba(X_test_svm)[:, 1]
    elif name == 'XGBoost':
        y_pred_test = model.predict(X_test_xgb)
        y_proba_test = model.predict_proba(X_test_xgb)[:, 1]
    elif name == 'CatBoost':
        y_pred_test = model.predict(X_test_cb)
        y_proba_test = model.predict_proba(X_test_cb)[:, 1]
    else:
        y_pred_test = model.predict(X_test_enc)
        y_proba_test = model.predict_proba(X_test_enc)[:, 1]
    test_results[name] = (y_pred_test, y_proba_test)
    
    acc = accuracy_score(y_test_split, y_pred_test)
    recall = recall_score(y_test_split, y_pred_test)
    print(f"{name:20} - Test Accuracy: {acc:.4f}, Recall: {recall:.4f}")

# ==========================================
# 10. FINAL EVALUATION OF BEST MODEL ON TEST SET
# ==========================================
print("\n" + "=" * 80)
print("10. FINAL EVALUATION OF BEST MODEL ON TEST SET")
print("=" * 80)

best_model = models[best_model_name]
if best_model_name == 'SVM':
    y_pred_test_best = best_model.predict(X_test_svm)
    y_proba_test_best = best_model.predict_proba(X_test_svm)[:, 1]
elif best_model_name == 'XGBoost':
    y_pred_test_best = best_model.predict(X_test_xgb)
    y_proba_test_best = best_model.predict_proba(X_test_xgb)[:, 1]
elif best_model_name == 'CatBoost':
    y_pred_test_best = best_model.predict(X_test_cb)
    y_proba_test_best = best_model.predict_proba(X_test_cb)[:, 1]
else:
    y_pred_test_best = best_model.predict(X_test_enc)
    y_proba_test_best = best_model.predict_proba(X_test_enc)[:, 1]

# Classification metrics
acc_best = accuracy_score(y_test_split, y_pred_test_best)
recall_best = recall_score(y_test_split, y_pred_test_best)
precision_best = precision_score(y_test_split, y_pred_test_best)
f1_best = f1_score(y_test_split, y_pred_test_best)
roc_auc_best = roc_auc_score(y_test_split, y_proba_test_best)
pr_auc_best = average_precision_score(y_test_split, y_proba_test_best)

tn, fp, fn, tp = confusion_matrix(y_test_split, y_pred_test_best).ravel()

print(f"\n{best_model_name} - FINAL TEST SET RESULTS (CLASSIFICATION):")
print("=" * 60)
print(f"  Accuracy:   {acc_best:.4f}")
print(f"  Recall:     {recall_best:.4f} ({tp}/{tp+fn} high-risk events detected)")
print(f"  Precision:  {precision_best:.4f}")
print(f"  F1 Score:   {f1_best:.4f}")
print(f"  ROC AUC:    {roc_auc_best:.4f}")
print(f"  PR AUC:     {pr_auc_best:.4f}")
print(f"  False Negatives: {fn} (missed dangerous events)")
print(f"  False Positives: {fp} (unnecessary alarms)")

# ==========================================
# 11. VISUALIZATIONS FOR ALL 5 MODELS (CLASSIFICATION)
# ==========================================
print("\n" + "=" * 80)
print("11. VISUALIZATIONS FOR ALL 5 MODELS")
print("=" * 80)

for name, (y_pred, y_proba) in test_results.items():
    print(f"\n--- Generating plots for {name} ---")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_split, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low-Risk', 'High-Risk'],
                yticklabels=['Low-Risk', 'High-Risk'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontweight='bold')
    plt.ylabel('Actual', fontweight='bold')
    plt.title(f'Confusion Matrix - {name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png', dpi=300)
    plt.close()
    print(f"  Saved confusion_matrix_{name.replace(' ', '_')}.png")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_split, y_proba)
    roc_auc_val = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'ROC (AUC = {roc_auc_val:.3f})')
    plt.plot([0,1], [0,1], 'navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(f'ROC Curve - {name}', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'roc_curve_{name.replace(" ", "_")}.png', dpi=300)
    plt.close()
    print(f"  Saved roc_curve_{name.replace(' ', '_')}.png")
    
    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test_split, y_proba)
    pr_auc_val = average_precision_score(y_test_split, y_proba)
    plt.figure(figsize=(8,6))
    plt.plot(rec, prec, 'green', lw=2, label=f'PR (AP = {pr_auc_val:.3f})')
    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title(f'Precision-Recall Curve - {name}', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'precision_recall_{name.replace(" ", "_")}.png', dpi=300)
    plt.close()
    print(f"  Saved precision_recall_{name.replace(' ', '_')}.png")
    
    # Feature Importance (only for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        if categorical_present:
            orig_feats = list(X_train.drop(columns=[categorical_feature]).columns)
            cat_feats = [f'c_object_type_{cat}' for cat in ohe.categories_[0]]
            feat_names = orig_feats + cat_feats
        else:
            feat_names = X_train.columns
        imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
        imp_df = imp_df.sort_values('Importance', ascending=False).head(15)
        plt.figure(figsize=(10,6))
        plt.barh(imp_df['Feature'], imp_df['Importance'], color='steelblue', edgecolor='black')
        plt.xlabel('Importance', fontweight='bold')
        plt.title(f'Top 15 Feature Importances - {name}', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'feature_importance_{name.replace(" ", "_")}.png', dpi=300)
        plt.close()
        print(f"  Saved feature_importance_{name.replace(' ', '_')}.png")

# ==========================================
# 12. ADDITIONAL VISUALIZATIONS FOR BEST MODEL
# ==========================================
print("\n" + "=" * 80)
print("12. ADDITIONAL VISUALIZATIONS FOR BEST MODEL")
print("=" * 80)

# Best model's confusion matrix (extra copy)
cm_best = confusion_matrix(y_test_split, y_pred_test_best)
plt.figure(figsize=(8,6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low-Risk', 'High-Risk'],
            yticklabels=['Low-Risk', 'High-Risk'],
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted', fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.title(f'Confusion Matrix - {best_model_name} (Best Model)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'confusion_matrix_{best_model_name}_best.png', dpi=300)
plt.close()

# Best model's ROC curve
fpr_best, tpr_best, _ = roc_curve(y_test_split, y_proba_test_best)
plt.figure(figsize=(8,6))
plt.plot(fpr_best, tpr_best, 'darkorange', lw=2, label=f'ROC (AUC = {roc_auc_best:.3f})')
plt.plot([0,1], [0,1], 'navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title(f'ROC Curve - {best_model_name} (Best Model)', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'roc_curve_{best_model_name}_best.png', dpi=300)
plt.close()

# Best model's PR curve
prec_best, rec_best, _ = precision_recall_curve(y_test_split, y_proba_test_best)
plt.figure(figsize=(8,6))
plt.plot(rec_best, prec_best, 'green', lw=2, label=f'PR (AP = {pr_auc_best:.3f})')
plt.xlabel('Recall', fontweight='bold')
plt.ylabel('Precision', fontweight='bold')
plt.title(f'Precision-Recall Curve - {best_model_name} (Best Model)', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'precision_recall_{best_model_name}_best.png', dpi=300)
plt.close()

print(f"✓ Saved additional best model plots with '_best' suffix")

# ==========================================
# 13. MODEL COMPARISON BAR CHARTS (CLASSIFICATION)
# ==========================================
print("\n" + "=" * 80)
print("13. MODEL COMPARISON BAR CHARTS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
metrics_plot = ['Accuracy', 'Recall', 'Precision', 'F1', 'ROC AUC', 'PR AUC']
for i, metric in enumerate(metrics_plot):
    ax = axes[i//3, i%3]
    bars = ax.bar(val_comparison_df['Model'], val_comparison_df[metric], color='steelblue', edgecolor='black')
    ax.set_title(f'{metric} Comparison (Validation Set)', fontweight='bold', fontsize=12)
    ax.set_ylabel(metric, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('model_comparison_charts.png', dpi=300)
plt.close()
print("Saved model_comparison_charts.png")

# ==========================================
# 14. KAGGLE SUBMISSION (Probability of High-Risk)
# ==========================================
print("\n" + "=" * 60)
print("14. KAGGLE SUBMISSION (Probability of High-Risk)")
print("=" * 60)

if best_model_name == 'SVM':
    final_proba = best_model.predict_proba(X_test_final_svm)[:, 1]
elif best_model_name == 'XGBoost':
    final_proba = best_model.predict_proba(X_test_final_xgb)[:, 1]
elif best_model_name == 'CatBoost':
    final_proba = best_model.predict_proba(X_test_final_cb)[:, 1]
else:
    final_proba = best_model.predict_proba(X_test_final_enc)[:, 1]

submission = pd.DataFrame({'risk': final_proba})  # probability of high-risk
if 'event_id' in test_data.columns:
    submission.insert(0, 'event_id', test_data['event_id'])
else:
    submission.insert(0, 'id', range(len(test_data)))
submission.to_csv('submission.csv', index=False)
print(f"Saved submission.csv using {best_model_name} (Probability of high-risk)")
print(f"  Model was selected based on VALIDATION recall")

# ==========================================
# 15. FINAL SUMMARY
# ==========================================
print("\n" + "=" * 80)
print("ALL TASKS COMPLETE")
print("=" * 80)
print(f"""
FINAL SUMMARY (CLASSIFICATION ONLY):
  • Total models trained: 5
  • Best model (by validation recall): {best_model_name}
  • Validation recall of best model: {best_val_recall:.4f}
  • Test recall of best model: {recall_best:.4f} ({tp}/{tp+fn} high-risk events detected)
  • Test ROC AUC: {roc_auc_best:.4f}
  • Test PR AUC: {pr_auc_best:.4f}
  • Submission file: submission.csv (probability of high-risk)

VISUALIZATIONS GENERATED:
  For each of the 5 models:
    ✓ Confusion Matrix
    ✓ ROC Curve
    ✓ Precision-Recall Curve
    ✓ Feature Importance (for tree-based models)
  
  Additional for best model (with '_best' suffix):
    ✓ Confusion Matrix
    ✓ ROC Curve
    ✓ Precision-Recall Curve
  
  Additional comparison:
    ✓ Model Comparison Bar Charts (6 classification metrics)

  All PNG files saved in current directory.
""") is this fully okay now proper classification
