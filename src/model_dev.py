# model_training.py
"""
Train classification and regression models with caching to skip already trained models.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, classification_report
from sklearn.impute import SimpleImputer
import joblib
url = "https://www.dropbox.com/scl/fi/x45dmh7gr7zdoxhmv0grj/updated_engineered_features_with_provinces.csv?rlkey=e4fc80irn8a1yacjw80ilvirn&st=wokg0z5e&dl=1"
# Paths and directories
DATA_PATH = url
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Load engineered data
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])

# Prepare feature matrix
feature_cols = [c for c in df.columns if c not in ['Date', 'District', 'climate_zone', 'flood_event', 'vulnerability_class', 'Impact']]
X = df[feature_cols]
# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)
# Impute any remaining NaNs
X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)

# Utility to skip already trained models
def should_train(fname):
    return not os.path.exists(os.path.join(MODELS_DIR, fname))

# Classification models
## 1) Climate Zone Random Forest
zone_model = 'rf_climate_zone.pkl'
if should_train(zone_model):
    print('Training RandomForest for climate_zone...')
    y = df['climate_zone']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    print('RF Accuracy:', accuracy_score(y_te, preds))
    print(classification_report(y_te, preds))
    joblib.dump(model, os.path.join(MODELS_DIR, zone_model))
else:
    print(f'{zone_model} exists, skipping.')

## 2) Flood Event HistGradientBoostingClassifier
flood_model = 'histgb_flood_event.pkl'
if 'flood_event' in df.columns and should_train(flood_model):
    print('Training HistGradientBoosting for flood_event...')
    y = df['flood_event']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    print('Flood Event F1:', f1_score(y_te, preds))
    joblib.dump(model, os.path.join(MODELS_DIR, flood_model))
else:
    print(f'{flood_model} exists or no flood_event, skipping.')

## 3) Vulnerability GradientBoostingClassifier
vuln_model = 'gb_vulnerability.pkl'
if 'vulnerability_class' in df.columns and should_train(vuln_model):
    print('Training GradientBoosting for vulnerability_class...')
    y = df['vulnerability_class']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = GradientBoostingClassifier(n_estimators=100, random_state=42, n_iter_no_change=10)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    print('Vulnerability F1 (weighted):', f1_score(y_te, preds, average='weighted'))
    joblib.dump(model, os.path.join(MODELS_DIR, vuln_model))
else:
    print(f'{vuln_model} exists or no vulnerability_class, skipping.')

# Regression models for Impact
def train_regression(model, name):
    fname = f'{name}.pkl'
    if should_train(fname):
        print(f'Training {name}...')
        y = df['Impact']
        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)).mean()
        print(f'{name} CV RMSE: {cv_rmse:.3f}')
        # Final train/test
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        print(f'{name} Test RMSE: {np.sqrt(mean_squared_error(y_te, preds)):.3f}, MAE: {mean_absolute_error(y_te, preds):.3f}')
        joblib.dump(model, os.path.join(MODELS_DIR, fname))
    else:
        print(f'{fname} exists, skipping.')

# Instantiate and train regressors
regressors = [
    (LinearRegression(), 'linreg_impact'),
    (Ridge(alpha=1.0), 'ridge_impact'),
    (Lasso(alpha=0.1), 'lasso_impact'),
    (GradientBoostingRegressor(n_estimators=100, random_state=42), 'gbreg_impact')
]
for model, name in regressors:
    if 'Impact' in df.columns:
        train_regression(model, name)
    else:
        print('No Impact column; skipping regression.')

print('All done. Models saved in', MODELS_DIR)
