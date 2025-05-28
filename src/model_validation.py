# model_evaluation.py
"""
Evaluate and validate classification and regression models for climate impact prediction.
Includes:
- Appropriate evaluation metrics for different model types
- Cross-validation strategies for robust evaluation
- Sensitivity analysis for key parameters
- Comparison against baseline approaches
- Evaluation across geographical regions
- Assessment of prediction accuracy for different time horizons
- Documentation of uncertainty in model predictions
- Validation against recent climate events
"""
import os
import pandas as pd
import numpy as np
from datetime import timedelta
import joblib
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold,
                                     cross_val_score, TimeSeriesSplit)
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.dummy import DummyClassifier, DummyRegressor
url = "https://www.dropbox.com/scl/fi/x45dmh7gr7zdoxhmv0grj/updated_engineered_features_with_provinces.csv?rlkey=e4fc80irn8a1yacjw80ilvirn&st=wokg0z5e&dl=1"
def load_data(path=url):
    df = pd.read_csv(path, parse_dates=['Date'])
    return df

class ModelEvaluator:
    def __init__(self, df, models_dir='models'):
        self.df = df
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.feature_cols = [c for c in df.columns if c not in
                             ['Date', 'District', 'climate_zone',
                              'flood_event', 'vulnerability_class', 'Impact']]
        X = df[self.feature_cols]
        X = pd.get_dummies(X, drop_first=True)
        X = pd.DataFrame(
            X.fillna(X.mean()).values,
            columns=X.columns
        )
        self.X = X

    def evaluate_classification(self, target, model_name, cv_folds=5):
        """
        Evaluate classification models with multiple metrics and cross-validation.
        """
        y = self.df[target]
        # Split
        X_tr, X_te, y_tr, y_te = train_test_split(
            self.X, y, test_size=0.2, stratify=y, random_state=42)
        # Load model
        model = joblib.load(os.path.join(self.models_dir, model_name))
        # Predictions
        y_pred = model.predict(X_te)
        # Baseline Dummy
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_tr, y_tr)
        y_dummy = dummy.predict(X_te)
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_te, y_pred),
            'f1_macro': f1_score(y_te, y_pred, average='macro'),
            'precision_macro': precision_score(y_te, y_pred, average='macro'),
            'recall_macro': recall_score(y_te, y_pred, average='macro'),
            'baseline_accuracy': accuracy_score(y_te, y_dummy)
        }
        # Cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, self.X, y, cv=skf, scoring='f1_macro', n_jobs=-1)
        metrics['cv_f1_macro_mean'] = cv_scores.mean()
        metrics['cv_f1_macro_std'] = cv_scores.std()
        return metrics

    def evaluate_regression(self, model_name, cv_folds=5):
        """
        Evaluate regression models with multiple metrics and cross-validation.
        """
        y = self.df['Impact']
        # Split
        X_tr, X_te, y_tr, y_te = train_test_split(
            self.X, y, test_size=0.2, random_state=42)
        # Load model
        model = joblib.load(os.path.join(self.models_dir, model_name))
        # Predictions
        y_pred = model.predict(X_te)
        # Baseline Dummy
        dummy = DummyRegressor(strategy='mean')
        dummy.fit(X_tr, y_tr)
        y_dummy = dummy.predict(X_te)
        # Metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_te, y_pred)),
            'mae': mean_absolute_error(y_te, y_pred),
            'r2': r2_score(y_te, y_pred),
            'baseline_rmse': np.sqrt(mean_squared_error(y_te, y_dummy))
        }
        # Cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_mse = -cross_val_score(
            model, self.X, y, cv=kf,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        metrics['cv_rmse_mean'] = np.sqrt(cv_mse).mean()
        metrics['cv_rmse_std'] = np.sqrt(cv_mse).std()
        return metrics

    def sensitivity_analysis(self, model_name, param_name, param_values, target_type='regression'):
        """
        Conduct sensitivity analysis by varying a key parameter and recording performance.
        """
        results = {}
        for val in param_values:
            # Reload fresh model instance with updated parameter
            model = joblib.load(os.path.join(self.models_dir, model_name))
            setattr(model, param_name, val)
            # re-evaluate
            if target_type == 'regression':
                metrics = self.evaluate_regression(model_name)
            else:
                metrics = self.evaluate_classification(target_type, model_name)
            results[val] = metrics
        return results

    def evaluate_by_region(self, model_name, target_type='regression'):
        """
        Evaluate model performance across different geographical regions (District).
        """
        metrics_by_region = {}
        for region, group in self.df.groupby('District'):
            X_region = self.X.loc[group.index]
            if target_type == 'regression':
                y_region = group['Impact']
                model = joblib.load(os.path.join(self.models_dir, model_name))
                preds = model.predict(X_region)
                metrics_by_region[region] = {
                    'rmse': np.sqrt(mean_squared_error(y_region, preds)),
                    'mae': mean_absolute_error(y_region, preds)
                }
            else:
                y_region = group[target_type]
                model = joblib.load(os.path.join(self.models_dir, model_name))
                preds = model.predict(X_region)
                metrics_by_region[region] = {
                    'accuracy': accuracy_score(y_region, preds),
                    'f1': f1_score(y_region, preds, average='macro')
                }
        return metrics_by_region

    def evaluate_time_horizons(self, model_name, horizons_days=[7, 30, 90] , target_type='regression'):
        """
        Assess prediction accuracy for different time horizons.
        """
        metrics_by_horizon = {}
        for h in horizons_days:
            cutoff = self.df['Date'].max() - timedelta(days=h)
            idx = self.df['Date'] >= cutoff
            if target_type == 'regression':
                X_h = self.X[idx]
                y_h = self.df.loc[idx, 'Impact']
                model = joblib.load(os.path.join(self.models_dir, model_name))
                preds = model.predict(X_h)
                metrics_by_horizon[h] = {
                    'rmse': np.sqrt(mean_squared_error(y_h, preds)),
                    'mae': mean_absolute_error(y_h, preds)
                }
            else:
                X_h = self.X[idx]
                y_h = self.df.loc[idx, target_type]
                model = joblib.load(os.path.join(self.models_dir, model_name))
                preds = model.predict(X_h)
                metrics_by_horizon[h] = {
                    'accuracy': accuracy_score(y_h, preds),
                    'f1': f1_score(y_h, preds, average='macro')
                }
        return metrics_by_horizon

    def bootstrap_uncertainty(self, model_name, n_bootstraps=1000, target_type='regression'):
        """
        Document uncertainty via bootstrap resampling.
        """
        coefs = []
        y = self.df['Impact'] if target_type=='regression' else self.df[target_type]
        for _ in range(n_bootstraps):
            sample_idx = np.random.choice(self.df.index, size=len(self.df), replace=True)
            X_samp = self.X.loc[sample_idx]
            y_samp = y.loc[sample_idx]
            model = joblib.load(os.path.join(self.models_dir, model_name))
            model.fit(X_samp, y_samp)
            if hasattr(model, 'coef_'):
                coefs.append(model.coef_)
        coefs = np.array(coefs)
        return {
            'coef_mean': np.mean(coefs, axis=0),
            'coef_std': np.std(coefs, axis=0)
        }

    def validate_recent_events(self, model_name, recent_start_date, target_type='regression'):
        """
        Validate model predictions against recent climate events starting from a given date.
        """
        idx = self.df['Date'] >= pd.to_datetime(recent_start_date)
        X_recent = self.X[idx]
        if target_type=='regression':
            y_recent = self.df.loc[idx, 'Impact']
            model = joblib.load(os.path.join(self.models_dir, model_name))
            preds = model.predict(X_recent)
            return {
                'rmse': np.sqrt(mean_squared_error(y_recent, preds)),
                'mae': mean_absolute_error(y_recent, preds)
            }
        else:
            y_recent = self.df.loc[idx, target_type]
            model = joblib.load(os.path.join(self.models_dir, model_name))
            preds = model.predict(X_recent)
            return {
                'accuracy': accuracy_score(y_recent, preds),
                'f1': f1_score(y_recent, preds, average='macro')
            }

if __name__ == '__main__':
    df = load_data()
    evaluator = ModelEvaluator(df)

    # Example usage:
    # Classification evaluation
    cls_metrics = evaluator.evaluate_classification(
        target='climate_zone', model_name='rf_climate_zone.pkl')
    print("Climate Zone Classification Metrics:", cls_metrics)

    # Regression evaluation
    reg_metrics = evaluator.evaluate_regression(model_name='gbreg_impact.pkl')
    print("Impact Regression Metrics:", reg_metrics)

    # Sensitivity analysis for number of estimators in GBRegressor
    sens = evaluator.sensitivity_analysis(
        model_name='gbreg_impact.pkl', param_name='n_estimators',
        param_values=[50, 100, 200], target_type='regression')
    print("Sensitivity Analysis:", sens)

    # Region-wise evaluation
    region_metrics = evaluator.evaluate_by_region(
        model_name='gbreg_impact.pkl', target_type='regression')
    print("Region-wise Regression Metrics:", region_metrics)

    # Time horizon evaluation
    horizon_metrics = evaluator.evaluate_time_horizons(
        model_name='gbreg_impact.pkl', horizons_days=[7,30,90], target_type='regression')
    print("Time Horizon Metrics:", horizon_metrics)

    # Bootstrap uncertainty
    uncert = evaluator.bootstrap_uncertainty(
        model_name='linreg_impact.pkl', n_bootstraps=500)
    print("Coefficient Uncertainty:", uncert)

    # Validation against recent events since Jan 1, 2025
    recent_metrics = evaluator.validate_recent_events(
        model_name='gbreg_impact.pkl', recent_start_date='2025-01-01')
    print("Recent Events Validation Metrics:", recent_metrics)
