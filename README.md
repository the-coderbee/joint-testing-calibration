# Calibration Prediction Tool

An end-to-end Flask web application that performs sensor calibration and failure-risk prediction using a machine-learning pipeline. It ingests multisource sensor data (environmental, electrical, mechanical, maintenance logs), applies preprocessing and feature engineering, then uses a Gradient Boosting Regressor to generate calibrated strength values (or continuous failure-risk scores).

---

## 📌 Features

- **Manual data entry** via a dynamic WTForms form with validation and auto‑calculated features (power, heat index, time features).
- **Batch CSV upload** for large-scale predictions.
- **Robust ML pipeline**: KNN imputation, outlier removal (IQR), robust scaling, one‑hot encoding, feature selection, and halving grid search for hyperparameter tuning.
- **Model details**: Gradient Boosting Regressor (best), ElasticNet baseline; 5‑fold CV RMSE \~0.28, R² \~0.94.
- **Interactive UI**: Calibration curves, prediction history, quick‑stats dashboard.
- **Downloadable artifacts**: prediction results CSV, template CSV, feature‑importance and heatmap images.

---

## 📂 Repository Structure

```text
├── data/               # Data directory for training and testing
├── src/
│   ├── api/            # API endpoints
│   ├── ml/             # Machine learning models and pipelines
│   ├── static/         # Static assets (CSS, JS, images)
│   ├── templates/      # HTML templates for the application
│   ├── utils/          # Utility functions
│   ├── app.py          # Flask application
│   └── __init__.py     # Package initialization
├── configs/
│   ├── logging_config.py  # Logging configuration
│   └── logs/           # Log files directory
├── uploads/            # Runtime upload & result storage
├── .env                # Environment variables
├── .gitignore          # Git ignore file
├── calibration.db      # Database for storing predictions
├── calibration_pipeline.joblib  # Trained model
├── environment.yml     # conda environment spec
├── generate_model_viz.py  # Model visualization generation
├── LICENSE             # Project license
├── README.md           # This file
├── requirements.txt    # pip dependencies
├── run.py              # Application entry point
├── sample_calibration_curve.png  # Example visualization
└── train_model.py      # CLI entry point for training
```

---

## 🚀 Installation

### Using pip

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Using conda

1. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate calib-env
   ```
2. (Optional) If you add new packages:
   ```bash
   conda env export > environment.yml
   ```

---

## 🎯 Model Training

Train and save the calibration pipeline:

```bash
python train_model.py
```

- **Outputs**: `calibration_pipeline.joblib` in project root.
- **Logs**: CV RMSE, test RMSE, R², feature-importance plots under `uploads/results/images/`.

---

## 🏃‍♂️ Running the Flask App

1. Ensure `calibration_pipeline.joblib` exists (trained model).
2. Start the server:
   ```bash
   python run.py
   ```
3. Open your browser at `http://localhost:5000`.

### Endpoints

- **/**: Manual data entry & CSV upload form (batch predictions).
- **/predict**: POST single-entry prediction; returns calibrated value or failure score and calibration curve.
- **/upload_csv**: POST batch CSV; returns results table with predictions.
- **/download_results**: GET download last batch CSV of predictions.
- **/download_template**: GET CSV template with correct headers for batch upload.
- **/history**: View a paginated list of all past predictions, timestamps, and inputs.
- **/about_model**: Detailed model information page with performance metrics, feature importance, and diagnostic plots.

---

## 🛠️ CSV Template & Sample Data

- Download the blank template: [Download Template CSV](/download_template)
- Sample input for demonstration: `demo_input.csv` (5 realistic rows). Upload via "Batch Upload".

---

## 📊 About the Model ("Model Info" Page)

- **Model**: GradientBoostingRegressor v1.0
- **Training samples**: \~5,000
- **Features**: 30 raw + 5 engineered
- **5‑fold CV RMSE**: 0.28
- **Test R²**: 0.94
- **Top features**: voltage, temperature, vibration, footfall, AQ

Visualizations:

- Feature importance bar chart
- Residuals histogram
- Calibration curve plots

---

## 📝 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
