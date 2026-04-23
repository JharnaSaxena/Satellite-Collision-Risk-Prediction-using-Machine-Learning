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
