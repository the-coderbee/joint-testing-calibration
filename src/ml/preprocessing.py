"""Preprocessing utilities for ML pipeline."""
import pandas as pd
import numpy as np


def preprocess_form_data(form):
    """Process form data into a DataFrame for prediction."""
    print("Starting form preprocessing")
    form.process_timestamp()
    
    # Calculate any derived features
    heat_index = form.calculate_derived_features()
    print(f"Calculated heat index: {heat_index}")
    
    # Convert form data to dictionary, excluding submit field
    data = {}
    print("Form fields:")
    for field in form:
        field_name = field.name
        field_value = field.data
        print(f"  {field_name}: {field_value}")
        
        if field_name != 'submit' and field_name != 'csrf_token' and field_value is not None:
            # Map field names to model expected column names
            model_field_name = field_name
            
            # Map form fields to expected model column names
            field_mapping = {
                'sensor_id': 'Sensor_ID',
                'equipment_id': 'Equipment_ID',
                'temperature': 'temperature',  # Changed to lowercase to match expected column
                'voltage': 'voltage',
                'current': 'current',
                'power': 'Power (W)',
                'humidity': 'humidity',
                'vibration': 'vibration',
                'operational_status': 'Operational Status',
                'fault_status': 'Fault Status',
                'failure_type': 'Failure Type',
                'maintenance_type': 'Maintenance Type',
                'failure_history': 'Failure History',
                'repair_time': 'Repair Time (hrs)',
                'maintenance_costs': 'Maintenance Costs (USD)',
                'ambient_temperature': 'Ambient Temperature (°C)',
                'ambient_humidity': 'Ambient Humidity (%)',
                'x_coord': 'X',
                'y_coord': 'Y',
                'z_coord': 'Z',
                'equipment_criticality': 'Equipment Criticality',
                'fault_detected': 'Fault Detected',
                'predictive_maintenance_trigger': 'Predictive Maintenance Trigger',
                'hour': 'hour',
                'dayofweek': 'dayofweek',
                'month': 'month',
                'day': 'day',
                'is_weekend': 'is_weekend',
                'footfall': 'footfall',
                'temp_mode': 'tempMode',
                'aq': 'AQ',
                'uss': 'USS',
                'cs': 'CS',
                'voc': 'VOC',
                'rp': 'RP',
                'ip': 'IP'
            }
            
            # Use the mapping or the original field name
            if field_name in field_mapping:
                model_field_name = field_mapping[field_name]
                
            data[model_field_name] = [field_value]
    
    # Add heat index if calculated
    if heat_index is not None:
        data['heat_index'] = [heat_index]
    
    # Print collected data before adding defaults
    print("Collected data from form:")
    for key, value in data.items():
        print(f"  {key}: {value}")
    
    # Add default values for any missing required columns
    required_columns = ['month', 'Operational Status', 'Fault Detected', 'Maintenance Costs (USD)', 
                       'Predictive Maintenance Trigger', 'X', 'Fault Status', 'Ambient Humidity (%)', 
                       'hour', 'dayofweek', 'Repair Time (hrs)', 'Maintenance Type', 'day', 
                       'sensor_reading', 'Equipment_ID', 'Y', 'Power (W)', 'Failure History', 
                       'Sensor_ID', 'Ambient Temperature (°C)', 'Failure Type', 
                       'Equipment Criticality', 'Z', 'temperature', 'true_strength']
    
    for col in required_columns:
        if col not in data:
            # Special handling for true_strength, which is likely the target variable
            if col == 'true_strength':
                data[col] = [None]  # Use None for target variable
                print(f"Added required column '{col}' with None value (target variable)")
            else:
                data[col] = [0]  # Use 0 as a default value for features
                print(f"Added required column '{col}' with default value 0")
        

    df = pd.DataFrame(data)
    
    if 'temperature' in df.columns and 'Temperature' not in df.columns:
        df['Temperature'] = df['temperature']
        print("Added Temperature column (copy of temperature)")
    
    print(f"Final dataframe shape: {df.shape}")
    print(f"Final columns: {sorted(df.columns.tolist())}")
    
    return df 