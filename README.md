# Calibration Curve Testing

A modular Python application for evaluating model calibration with a web interface.

## Project Structure

The project has been restructured to be more modular:

```
calibration_curve_testing/
├── src/                      # Main source code directory
│   ├── api/                  # Flask web API
│   │   ├── forms.py          # Form definitions
│   │   ├── routes.py         # API routes
│   │   └── __init__.py
│   ├── ml/                   # Machine learning code
│   │   ├── model.py          # Model loading and prediction
│   │   ├── preprocessing.py  # Data preprocessing
│   │   ├── train.py          # Model training
│   │   └── __init__.py
│   ├── utils/                # Utility functions
│   │   ├── db.py             # Database utilities
│   │   ├── visualization.py  # Visualization utilities
│   │   └── __init__.py
│   ├── data/                 # Data directory
│   │   ├── raw_data/         # Raw data files
│   │   └── __init__.py
│   ├── static/               # Static assets
│   │   ├── css/
│   │   └── js/
│   ├── templates/            # HTML templates
│   ├── app.py                # Flask application factory
│   └── __init__.py
├── data/                     # Original data directory
├── uploads/                  # Uploads directory for file submissions
├── run.py                    # Script to run the web application
├── train_model.py            # Script to train the model
├── calibration_pipeline.joblib # Trained model
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
```
git clone <repository_url>
cd calibration_curve_testing
```

2. Install the dependencies:
```
pip install -r requirements.txt
```

## Usage

### Training the Model

To train a new model:

```
python train_model.py
```

This will create a new `calibration_pipeline.joblib` file containing the trained model.

### Running the Web Application

To run the web application:

```
python run.py
```

Then open your browser and navigate to http://localhost:5000.

## Features

- Web interface for predicting on individual inputs
- Batch prediction on CSV files
- Visualization of calibration curves
- Historical prediction tracking
- RESTful API for predictions 