"""API routes for the Flask application."""
import os
import pandas as pd
from flask import (Flask, render_template, request, jsonify, redirect, 
                  url_for, flash, session, send_file)
from werkzeug.utils import secure_filename
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import traceback

from src.api.forms import ManualDataForm, FileUploadForm
from src.ml.preprocessing import preprocess_form_data
from src.ml.model import make_prediction
from src.utils.db import save_prediction, get_past_predictions, init_db
from src.utils.visualization import plot_calibration_curve, save_result_image


def register_routes(app):
    """Register Flask routes to the app."""
    # Initialize the database
    init_db()
    
    # Configure uploads folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    @app.route('/', methods=['GET', 'POST'])
    def index():
        manual_form = ManualDataForm()
        upload_form = FileUploadForm()
        
        if request.method == 'POST':
            # Debug logging
            print("POST request received")
            print("Form data:", {k: v for k, v in request.form.items()})
            print("Form keys:", list(request.form.keys()))
            
            # Check for form-type explicitly
            form_type = request.form.get('form-type')
            print(f"Form type detected: {form_type}")
            
            # Handle manual form submission - don't check validation initially
            if form_type == 'manual':
                print("Processing manual form")
                print("CSRF Token:", manual_form.csrf_token.data)
                
                # Check form validation
                if manual_form.validate():
                    print("Form validated")
                else:
                    print("Form validation failed:", manual_form.errors)
                
                try:
                    # Process form data regardless of validation
                    input_df = preprocess_form_data(manual_form)
                    prediction = make_prediction(input_df)
                    
                    if prediction is None:
                        flash('Error making prediction. Check the input data.', 'danger')
                        return render_template('index.html', manual_form=manual_form, upload_form=upload_form, now=datetime.now())
                    
                    print(f"Prediction made: {prediction}")
                    
                    # Save prediction to database
                    sensor_id = manual_form.sensor_id.data
                    equipment_id = manual_form.equipment_id.data
                    save_prediction(input_df.to_dict(), prediction, None, sensor_id, equipment_id)
                    
                    # Get past predictions for calibration curve
                    past_predictions = get_past_predictions(100)
                    predictions = [row['prediction'] for row in past_predictions]
                    actual_values = [row['actual_value'] for row in past_predictions]
                    
                    # Generate calibration curve and save it
                    plot_url = plot_calibration_curve(predictions, actual_values)
                    
                    # Also create and save a custom calibration curve for reference
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
                    img_path = save_result_image(fig, f'calibration_{timestamp}.png', 'calibration_curves')
                    
                    # Format input data for the template
                    formatted_input_data = {}
                    for key, value in input_df.to_dict().items():
                        if key != 'true_strength':  # Skip target variable
                            formatted_input_data[key] = value
                    
                    # Use the original result.html template
                    return render_template('result.html', 
                                          prediction=prediction, 
                                          input_data=formatted_input_data,
                                          plot_url=plot_url,
                                          saved_image=img_path,
                                          now=datetime.now())
                except Exception as e:
                    print(f"Error processing form: {e}")
                    import traceback
                    print(traceback.format_exc())
                    flash(f"Error processing form: {str(e)}", "danger")
            
            # Rest of the code for file upload form handling remains the same
        
        # Default GET request - render the forms
        return render_template('index.html', manual_form=manual_form, upload_form=upload_form, now=datetime.now())
    
    
    @app.route('/download/<filename>')
    def download_file(filename):
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename),
                        mimetype='text/csv',
                        as_attachment=True)
    
    
    @app.route('/history')
    def history():
        predictions = get_past_predictions(100)
        
        # Extract prediction values for the calibration curve
        prediction_values = [row['prediction'] for row in predictions]
        actual_values = [row['actual_value'] for row in predictions]
        
        # Generate calibration curve
        plot_url = None
        if prediction_values:
            plot_url = plot_calibration_curve(prediction_values, actual_values)
        
        return render_template('history.html', 
                              predictions=predictions, 
                              plot_url=plot_url,
                              now=datetime.now()) 