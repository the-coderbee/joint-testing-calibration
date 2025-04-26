"""API routes for the Flask application."""
import os
from flask import (render_template, request, flash, send_file)
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import uuid

from configs.logging_config import setup_logging, get_inference_logger, ModelPerformanceMonitor

from src.api.forms import ManualDataForm, FileUploadForm
from src.ml.preprocessing import preprocess_form_data
from src.ml.model import make_prediction
from src.utils.db import save_prediction, get_past_predictions, init_db
from src.utils.visualization import plot_calibration_curve, save_result_image


# Initialize the main logger
logger = setup_logging()


def register_routes(app):
    """Register Flask routes to the app."""
    # Initialize the database
    logger.info("Initializing database")
    init_db()
    
    # Configure uploads folder
    logger.info(f"Configuring upload folder: {app.config['UPLOAD_FOLDER']}")
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    @app.route('/', methods=['GET', 'POST'])
    def index():
        logger.info("Index route accessed")
        manual_form = ManualDataForm()
        upload_form = FileUploadForm()
        
        if request.method == 'POST':
            # Use logger instead of print
            logger.debug("POST request received")
            logger.debug(f"Form data: {request.form}")
            
            # Check for form-type explicitly
            form_type = request.form.get('form-type')
            logger.info(f"Form type detected: {form_type}")
            
            # Handle manual form submission
            if form_type == 'manual':
                logger.info("Processing manual form submission")
                logger.debug(f"CSRF Token: {manual_form.csrf_token.data}")
                
                # Check form validation
                if manual_form.validate():
                    logger.info("Form validated successfully")
                else:
                    logger.warning(f"Form validation failed: {manual_form.errors}")
                
                try:
                    # Create an inference logger with sensor/equipment context
                    sensor_id = manual_form.sensor_id.data
                    equipment_id = manual_form.equipment_id.data
                    model_name = f"sensor_{sensor_id}_equipment_{equipment_id}"
                    inference_logger = get_inference_logger(model_name)
                    
                    # Process form data with performance monitoring
                    with ModelPerformanceMonitor(inference_logger, "Form Data Processing"):
                        inference_logger.info("Processing form data")
                        input_df = preprocess_form_data(manual_form)
                    
                    # Make prediction with performance monitoring
                    with ModelPerformanceMonitor(inference_logger, "Model Prediction"):
                        inference_logger.info(f"Making prediction for sensor {sensor_id} on equipment {equipment_id}")
                        prediction = make_prediction(input_df)
                    
                    if prediction is None:
                        logger.error("Null prediction returned from model")
                        flash('Error making prediction. Check the input data.', 'danger')
                        return render_template('index.html', manual_form=manual_form, upload_form=upload_form, now=datetime.now())
                    
                    logger.info(f"Prediction successfully made: {prediction}")
                    
                    # Save prediction to database
                    logger.debug(f"Saving prediction to database: {prediction}")
                    save_prediction(input_df.to_dict(), prediction, None, sensor_id, equipment_id)
                    
                    # Get past predictions for calibration curve
                    logger.debug("Retrieving past predictions for calibration curve")
                    past_predictions = get_past_predictions(100)
                    predictions = [row['prediction'] for row in past_predictions]
                    actual_values = [row['actual_value'] for row in past_predictions]
                    
                    # Generate calibration curve and save it
                    logger.info("Generating calibration curve")
                    plot_url = plot_calibration_curve(predictions, actual_values)
                    
                    # Also create and save a custom calibration curve for reference
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    logger.debug(f"Creating custom calibration curve with timestamp {timestamp}")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    sorted_preds = np.sort(predictions)
                    quantiles = np.linspace(0, 1, len(sorted_preds))
                    
                    ax.plot(sorted_preds, quantiles, 'b-', linewidth=2, label='Calibration Curve')
                    ax.set_xlabel('Predicted Value')
                    ax.set_ylabel('Quantile')
                    ax.set_title(f'Calibration Curve - {timestamp}')
                    ax.grid(True, alpha=0.3)
                    
                    # Highlight the current prediction
                    idx = np.searchsorted(sorted_preds, prediction)
                    if idx < len(quantiles):
                        ax.scatter([prediction], [quantiles[idx]], color='red', s=100, 
                                  label=f'Current Prediction: {prediction:.2f}', zorder=5)
                    
                    ax.legend()
                    plt.tight_layout()
                    
                    # Save the figure
                    logger.debug("Saving calibration curve image")
                    img_path = save_result_image(fig, f'calibration_{timestamp}.png', 'calibration_curves')
                    
                    # Format input data for the template
                    formatted_input_data = {}
                    for key, value in input_df.to_dict().items():
                        if key != 'true_strength':  # Skip target variable
                            formatted_input_data[key] = value
                    
                    logger.info("Rendering result template")
                    # Use the original result.html template
                    return render_template('result.html', 
                                          prediction=prediction, 
                                          input_data=formatted_input_data,
                                          plot_url=plot_url,
                                          saved_image=img_path,
                                          now=datetime.now())
                except Exception as e:
                    logger.error(f"Error processing form: {str(e)}", exc_info=True)
                    flash(f"Error processing form: {str(e)}", "danger")
            
            # File upload form handling should also be updated with logging, but kept out for brevity
            elif form_type == 'upload':
                logger.info("File upload form submitted")
                
                if upload_form.validate_on_submit():
                    logger.debug("File upload form validated")
                    
                    try:
                        # Create necessary directories
                        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'results', 'images', 'batch_predictions'), exist_ok=True)
                        
                        # Get the uploaded file
                        csv_file = upload_form.csv_file.data
                        logger.info(f"Received file: {csv_file.filename}")
                        
                        # Create a unique filename
                        filename = f"{uuid.uuid4().hex}_{csv_file.filename}"
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        
                        # Save the uploaded file
                        csv_file.save(file_path)
                        logger.info(f"Saved file to: {file_path}")
                        
                        # Read the CSV file
                        df = pd.read_csv(file_path)
                        logger.info(f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns")
                        logger.debug(f"CSV columns: {list(df.columns)}")
                        
                        # Process each row in the CSV
                        results = []
                        predictions = []
                        
                        # Create a model inference logger
                        model_name = f"batch_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        inference_logger = get_inference_logger(model_name)
                        
                        # Make predictions for each row
                        with ModelPerformanceMonitor(inference_logger, "Batch Prediction"):
                            for index, row in df.iterrows():
                                # Convert row to DataFrame for prediction
                                row_df = pd.DataFrame([row])
                                
                                # Check if row_df has the expected columns
                                inference_logger.info(f"Processing row {index+1}/{len(df)}")
                                
                                try:
                                    # Make prediction
                                    prediction = make_prediction(row_df)
                                    
                                    if prediction is not None:
                                        # Store prediction with row data
                                        row_dict = row.to_dict()
                                        results.append({
                                            'row': index + 1,
                                            'data': row_dict,
                                            'prediction': prediction
                                        })
                                        predictions.append(prediction)
                                        
                                        # Save to database
                                        sensor_id = row.get('sensor_id', row.get('Sensor_ID', 'batch'))
                                        equipment_id = row.get('equipment_id', row.get('Equipment_ID', 'batch'))
                                        save_prediction(row_dict, prediction, None, sensor_id, equipment_id)
                                        
                                    else:
                                        inference_logger.warning(f"Null prediction for row {index+1}")
                                        results.append({
                                            'row': index + 1,
                                            'data': row.to_dict(),
                                            'prediction': None,
                                            'error': 'Null prediction returned'
                                        })
                                except Exception as e:
                                    inference_logger.error(f"Error processing row {index+1}: {str(e)}", exc_info=True)
                                    results.append({
                                        'row': index + 1,
                                        'data': row.to_dict(),
                                        'prediction': None,
                                        'error': str(e)
                                    })
                        
                        # Generate summary statistics
                        summary = {}
                        if predictions:
                            summary = {
                                'count': len(predictions),
                                'mean': np.mean(predictions),
                                'std': np.std(predictions),
                                'min': np.min(predictions),
                                'max': np.max(predictions)
                            }
                            
                        # Generate calibration curve
                        past_predictions = get_past_predictions(100)
                        pred_values = [row['prediction'] for row in past_predictions]
                        actual_values = [row['actual_value'] for row in past_predictions]
                        
                        # Create calibration curve
                        plot_url = plot_calibration_curve(pred_values, actual_values)
                        
                        # Create a histogram of predictions
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(predictions, bins=20, alpha=0.7, color='blue')
                        ax.set_xlabel('Predicted Value')
                        ax.set_ylabel('Count')
                        ax.set_title(f'Distribution of Batch Predictions - {timestamp}')
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        # Save the histogram
                        img_filename = f'batch_histogram_{timestamp}.png'
                        img_path = save_result_image(fig, img_filename, 'batch_predictions')
                        
                        logger.info(f"Batch prediction completed with {len(results)} results")
                        return render_template('batch_result.html',
                                              results=results,
                                              summary=summary,
                                              plot_url=plot_url,
                                              saved_image=img_path,
                                              file_path=file_path,
                                              filename=filename,
                                              now=datetime.now())
                    
                    except Exception as e:
                        logger.error(f"Error processing CSV file: {str(e)}", exc_info=True)
                        flash(f"Error processing CSV file: {str(e)}", "danger")
                else:
                    logger.warning(f"File upload form validation failed: {upload_form.errors}")
                    flash(f"File upload validation failed: {upload_form.errors}", "danger")
        
        # Default GET request - render the forms
        logger.debug("Rendering index template with forms")
        return render_template('index.html', manual_form=manual_form, upload_form=upload_form, now=datetime.now())
    
    
    @app.route('/download/<filename>')
    def download_file(filename):
        logger.info(f"File download requested: {filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            flash("File not found", "danger")
            return render_template('error.html', message="File not found"), 404
            
        logger.info(f"Sending file: {file_path}")
        return send_file(file_path,
                        mimetype='text/csv',
                        as_attachment=True)
    
    
    @app.route('/history')
    def history():
        logger.info("History page accessed")
        
        logger.debug("Retrieving past predictions")
        predictions = get_past_predictions(100)
        
        # Extract prediction values for the calibration curve
        prediction_values = [row['prediction'] for row in predictions]
        actual_values = [row['actual_value'] for row in predictions]
        
        # Generate calibration curve
        plot_url = None
        if prediction_values:
            logger.info(f"Generating calibration curve with {len(prediction_values)} data points")
            plot_url = plot_calibration_curve(prediction_values, actual_values)
        else:
            logger.warning("No prediction data available for calibration curve")
        
        logger.debug("Rendering history template")
        return render_template('history.html', 
                              predictions=predictions, 
                              plot_url=plot_url,
                              now=datetime.now())