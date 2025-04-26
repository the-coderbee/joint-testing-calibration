"""Model training script with improved pipeline, feature selection, halving grid search, and outlier handling."""
import glob
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import train_test_split, HalvingGridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def train_model(data_glob_pattern: str, output_path: str, logger=None):
    logger = logger or logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # 1. Load and unify datasets
    files = glob.glob(data_glob_pattern)
    if not files:
        raise FileNotFoundError(f"No CSV files found for pattern {data_glob_pattern}")

    dfs = []
    rename_map = {
        'fail': 'fail_flag',
        'Voltage (V)': 'voltage', 'Current (A)': 'current',
        'Temperature (°C)': 'temperature', 'Humidity (%)': 'humidity',
        'Vibration (m/s²)': 'vibration', 'Power (W)': 'power',
        'aq': 'AQ', 'uss': 'USS', 'cs': 'CS', 'voc': 'VOC', 'rp': 'RP', 'ip': 'IP',
        'temp_mode': 'tempMode'
    }
    for f in files:
        df = pd.read_csv(f)
        df = df.rename(columns=rename_map)
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['hour'] = df['Timestamp'].dt.hour
            df['dayofweek'] = df['Timestamp'].dt.dayofweek
            df = df.drop(columns=['Timestamp'])
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded data shape: {data.shape}")
    if data.shape[0] == 0:
        raise RuntimeError("No rows loaded")
    logger.info(f"Loaded data shape: {data.shape}")

    # 2. Handle target
    target = 'fail_flag' if 'fail_flag' in data.columns else 'true_strength'
    data = data.dropna(subset=[target])

    # 3. Feature engineering
    skew_feats = [col for col in data.select_dtypes(include=['float64','int64']).columns if data[col].skew() > 1]
    for col in skew_feats:
        data[col] = np.log1p(data[col])

    if 'voltage' in data and 'current' in data:
        data['calculated_power'] = data['voltage'] * data['current']
    if 'temperature' in data and 'humidity' in data:
        data['heat_index'] = data['temperature'] + 0.05 * data['humidity']

    # Outlier removal (IQR method)
    def remove_outliers():
        for col in data.select_dtypes(include=['float64','int64']).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    
    # remove_outliers()

    # 4. Split X, y
    X = data.drop(columns=[target])
    y = data[target]

    # 5. Identify feature types
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category', 'bool']).columns.tolist()

    # 6. Preprocessing pipeline
    numeric_transformer = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ], remainder='drop')

    # 7. Feature selector integrated
    selector = SelectFromModel(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42),
        threshold='median'
    )

    # 8. Define pipeline for Gradient Boosting
    gbr_pipe = Pipeline([
        ('preproc', preprocessor),
        ('selector', selector),
        ('gbr', GradientBoostingRegressor(random_state=42))
    ])

    # 9. Hyperparameter tuning with HalvingGridSearchCV
    param_grid = {
        'selector__estimator__n_estimators': [50, 100],
        'gbr__n_estimators': [100, 200],
        'gbr__learning_rate': [0.01, 0.1],
        'gbr__max_depth': [3, 5]
    }
    search = HalvingGridSearchCV(
        gbr_pipe, param_grid, cv=5,
        scoring='neg_root_mean_squared_error', factor=3,
        n_jobs=-1
    )

    # 10. Split data
    stratify = y if set(np.unique(y)) <= {0, 1} else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # 11. Fit search
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    logger.info(f"Best parameters: {search.best_params_}")

    # 12. Evaluate with cross_val_score and on test set
    cv_scores = -cross_val_score(best_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    logger.info(f"5-fold CV RMSE: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Test RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # 13. Save best model pipeline
    joblib.dump(best_model, output_path)
    logger.info(f"Saved best model to {output_path}")

    return best_model


if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_pattern = os.path.join(project_root, 'data', 'raw_data', '*.csv')
    output_file = os.path.join(project_root, 'calibration_pipeline.joblib')
    train_model(data_pattern, output_file)
