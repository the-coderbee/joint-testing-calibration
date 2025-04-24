"""Model training script."""
import glob
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel

# Import our utility functions
from src.utils.visualization import save_result_image, create_heatmap


def train_model():
    """Train the model and save it to disk."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    # 1. Load and inspect all CSVs
    logger.info("Loading data files")
    
    # Use absolute path based on the current file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../..'))
    data_path = os.path.join(project_root, 'data', 'raw_data', '*.csv')
    
    logger.info(f"Looking for data files in: {data_path}")
    files = glob.glob(data_path)
    
    # If no files found in project data directory, try src data directory
    if not files:
        src_data_path = os.path.join(project_root, 'src', 'data', 'raw_data', '*.csv')
        logger.info(f"No files found. Trying alternative path: {src_data_path}")
        files = glob.glob(src_data_path)
    
    if not files:
        logger.error("No CSV files found in data directories!")
        raise FileNotFoundError("No CSV files found in data/raw_data or src/data/raw_data directories")
    
    logger.info(f"Found {len(files)} CSV files: {files}")
    
    df_list = []
    for f in files:
        df = pd.read_csv(f)
        logger.info(f"{f} → columns: {list(df.columns)}")  # inspect schema

        # 2. Rename to canonical names
        rename_map = {
            'fail': 'fail_flag',           # file1's label
            'Voltage (V)': 'voltage',      # file2 numeric features
            'Current (A)': 'current',
            'Temperature (°C)': 'temperature',
            'Humidity (%)': 'humidity',
            'Vibration (m/s²)': 'vibration',
            'aq':        'AQ',
            'uss':       'USS',
            'cs':        'CS',
            'voc':       'VOC',
            'rp':        'RP',
            'ip':        'IP',
            'temp_mode': 'tempMode'
        }
        df = df.rename(columns=rename_map)
        if 'Timestamp' in df:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['hour']      = df['Timestamp'].dt.hour
            df['dayofweek'] = df['Timestamp'].dt.dayofweek
        df_list.append(df)

    # 3. Concatenate into one DataFrame
    data = pd.concat(df_list, axis=0, ignore_index=True)
    logger.info(f"Combined dataset shape: {data.shape}")

    # 4. Exploratory Data Analysis
    target = 'fail_flag' if 'fail_flag' in data.columns else 'true_strength'
    logger.info(f"Target variable: {target}")

    # Check how many NaN values are in the target
    nan_count = data[target].isna().sum()
    logger.info(f"NaN values in target variable '{target}': {nan_count}")

    if nan_count > 0:
        # Option 1: Drop rows with NaN targets (usually the best approach)
        logger.info(f"Dropping {nan_count} rows with NaN target values")
        data = data.dropna(subset=[target])
        

    # Verify no more NaNs in target
    remaining_nans = data[target].isna().sum()
    logger.info(f"Remaining NaNs in target after handling: {remaining_nans}")
    logger.info("Performing exploratory data analysis")
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    correlation = numeric_data.corr()

    # Optional: Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    # Save to uploads/results/images/heatmaps
    corr_fig = plt.gcf()
    uploads_dir = os.path.join(project_root, 'uploads', 'results', 'images', 'heatmaps')
    os.makedirs(uploads_dir, exist_ok=True)
    corr_path = os.path.join(uploads_dir, 'correlation_heatmap.png')
    corr_fig.savefig(corr_path, bbox_inches='tight', dpi=300)
    logger.info(f"Correlation heatmap saved to {corr_path}")
    plt.close()

    # 5. Check for outliers and handle them
    def detect_outliers(df, cols):
        outlier_info = {}
        for col in cols:
            if df[col].dtype in ['int64', 'float64']:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
                outlier_info[col] = outliers
        return outlier_info

    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    outlier_dict = detect_outliers(data, num_cols)
    logger.info(f"Outliers per column: {outlier_dict}")

    # 6. Drop irrelevant columns
    to_drop = ['Equipment Relationship', 'External Factors', 'Last Maintenance Date']
    data = data.drop(columns=to_drop, errors='ignore')
    logger.info(f"Dropped irrelevant columns: {to_drop}")

    # 7. Handle missing values - use KNN for better imputation
    missing_values = data.isnull().sum()
    logger.info(f"Missing values per column: {missing_values[missing_values > 0]}")

    # 8. Extract datetime features if present
    if 'Timestamp' in data.columns:
        logger.info("Extracting datetime features")
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data['hour'] = data['Timestamp'].dt.hour
        data['dayofweek'] = data['Timestamp'].dt.dayofweek
        data['month'] = data['Timestamp'].dt.month  # Additional feature
        data['day'] = data['Timestamp'].dt.day      # Additional feature
        data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)  # Weekend feature

    # 9. Prepare feature matrix X and target y
    target = 'fail_flag' if 'fail_flag' in data.columns else 'true_strength'
    logger.info(f"Target variable: {target}")

    # 10. Create feature interactions that might be relevant
    logger.info("Creating feature interactions")
    if 'Power' in data.columns and 'voltage' in data.columns and 'current' in data.columns:
        # Calculate power if not present or verify its correctness
        data['calculated_power'] = data['voltage'] * data['current']
        
    if 'temperature' in data.columns and 'humidity' in data.columns:
        # Create heat index feature (simplified version)
        data['heat_index'] = data['temperature'] + 0.05 * data['humidity']

    # 11. Identify numeric vs. categorical features
    X = data.drop(columns=[target])
    y = data[target]

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    logger.info(f"Categorical features: {cat_cols}")
    logger.info(f"Numerical features: {num_cols}")

    # 12. More robust preprocessing
    numeric_transformer = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),  # KNN imputer often works better than median
        ('scaler', RobustScaler())  # More robust to outliers than StandardScaler
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ], remainder='drop')

    # 13. Split data for training/testing with stratification if binary target
    if np.array_equal(np.unique(y), [0, 1]):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info("Using stratified split for binary target")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info("Using regular split for non-binary target")

    # 14. Feature selection to improve model
    feature_selector = RandomForestRegressor(n_estimators=100, random_state=42)
    # We need to preprocess the data first before feature selection
    X_train_processed = preprocessor.fit_transform(X_train)
    feature_selector.fit(X_train_processed, y_train)

    # Create the selector with the trained model
    selector = SelectFromModel(feature_selector, threshold='median')
    selector.fit(X_train_processed, y_train)

    # Get selected feature names/indices for reference
    selected_mask = selector.get_support()
    feature_names = []
    try:
        # Try to get the feature names after preprocessing
        feature_names = preprocessor.get_feature_names_out()
        selected_features = [feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]]
        logger.info(f"Selected features: {selected_features}")
    except:
        logger.info(f"Number of selected features: {sum(selected_mask)}")

    # Now update the pipelines to include the selector
    # For GBR pipeline
    gbr_pipeline = Pipeline([
        ('preproc', preprocessor),
        ('selector', selector),  # Add the selector here
        ('gbr', GradientBoostingRegressor(random_state=42))
    ])

    # For RF pipeline
    rfr_pipeline = Pipeline([
        ('preproc', preprocessor),
        ('selector', selector),  # Add the selector here
        ('rfr', RandomForestRegressor(random_state=42))
    ])

    # For ElasticNet pipeline
    elastic_pipeline = Pipeline([
        ('preproc', preprocessor),
        ('selector', selector),  # Add the selector here
        ('elastic', ElasticNet(random_state=42))
    ])

    # 15. Grid search for hyperparameter tuning
    logger.info("Training models with hyperparameter optimization")

    # GBR pipeline with grid search
    gbr_pipeline = Pipeline([
        ('preproc', preprocessor),
        ('gbr', GradientBoostingRegressor(random_state=42))
    ])

    gbr_params = {
        'gbr__n_estimators': [50, 100, 200],
        'gbr__learning_rate': [0.01, 0.1, 0.2],
        'gbr__max_depth': [3, 5, 7],
        'gbr__min_samples_split': [2, 5, 10]
    }

    gbr_grid = GridSearchCV(
        gbr_pipeline, gbr_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    gbr_grid.fit(X_train, y_train)
    logger.info(f"Best GBR params: {gbr_grid.best_params_}")

    # RF pipeline with grid search
    rfr_pipeline = Pipeline([
        ('preproc', preprocessor),
        ('rfr', RandomForestRegressor(random_state=42))
    ])

    rfr_params = {
        'rfr__n_estimators': [100, 200],
        'rfr__max_depth': [None, 10, 20],
        'rfr__min_samples_split': [2, 5],
        'rfr__min_samples_leaf': [1, 2, 4]
    }

    rfr_grid = GridSearchCV(
        rfr_pipeline, rfr_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    rfr_grid.fit(X_train, y_train)
    logger.info(f"Best RF params: {rfr_grid.best_params_}")

    # 16. Model evaluation
    # Get best model from gridsearch
    best_gbr = gbr_grid.best_estimator_
    best_rfr = rfr_grid.best_estimator_

    # Make predictions
    y_pred_gbr = best_gbr.predict(X_test)
    y_pred_rfr = best_rfr.predict(X_test)

    # Evaluate models
    logger.info("\nModel Evaluation:")
    for name, y_pred in [('GBR', y_pred_gbr), ('RF', y_pred_rfr)]:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # 17. Plot feature importance for best model
    def plot_feature_importance(model, title):
        try:
            if hasattr(model, 'feature_importances_'):
                # Direct feature importance
                importances = model.feature_importances_
                feature_names = ['Feature_' + str(i) for i in range(len(importances))]
            else:
                # If composite model with named steps (pipeline)
                for step_name, step in model.named_steps.items():
                    if hasattr(step, 'feature_importances_'):
                        importances = step.feature_importances_
                        try:
                            # Try to get transformed feature names
                            feature_names = model.named_steps['preproc'].get_feature_names_out()
                        except:
                            feature_names = ['Feature_' + str(i) for i in range(len(importances))]
                        break
                    
            # Sort feature importance
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(12, 8))
            plt.title(f"{title} Feature Importance")
            plt.bar(range(len(indices[:20])), 
                    [importances[i] for i in indices[:20]], 
                    align='center')
            plt.xticks(range(len(indices[:20])), 
                    [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in indices[:20]], 
                    rotation=90)
            plt.tight_layout()
            
            # Save to uploads/results/images/feature_importance
            fig = plt.gcf()
            uploads_dir = os.path.join(project_root, 'uploads', 'results', 'images', 'feature_importance')
            os.makedirs(uploads_dir, exist_ok=True)
            fig_path = os.path.join(uploads_dir, f'feature_importance_{title}.png')
            fig.savefig(fig_path, bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance plot for {title} saved to {fig_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")

    # Plot feature importance for both models
    try:
        plot_feature_importance(best_gbr.named_steps['gbr'], 'GBR')
        plot_feature_importance(best_rfr.named_steps['rfr'], 'RF')
    except Exception as e:
        logger.error(f"Error in feature importance extraction: {e}")

    # 18. Save the best model
    logger.info("Saving best model")
    model_path = os.path.join(project_root, 'calibration_pipeline.joblib')
    joblib.dump(best_gbr, model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Model training completed successfully")
    return best_gbr


if __name__ == "__main__":
    train_model() 