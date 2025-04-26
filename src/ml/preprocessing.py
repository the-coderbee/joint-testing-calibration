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
                # Core sensor data
                'sensor_id': 'Sensor_ID',
                'equipment_id': 'Equipment_ID',
                'temperature': 'temperature',
                'Temperature': 'Temperature',  # Capitalized temperature field
                'voltage': 'voltage',
                'current': 'current',
                'power': 'Power (W)',
                'humidity': 'humidity',
                'vibration': 'vibration',
                'sensor_reading': 'sensor_reading',
                'true_strength': 'true_strength',
                
                # Status fields
                'operational_status': 'Operational Status',
                'fault_status': 'Fault Status',
                'failure_type': 'Failure Type',
                'maintenance_type': 'Maintenance Type',
                'failure_history': 'Failure History',
                'repair_time': 'Repair Time (hrs)',
                'maintenance_costs': 'Maintenance Costs (USD)',
                'ambient_temperature': 'Ambient Temperature (°C)',
                'ambient_humidity': 'Ambient Humidity (%)',
                
                # Coordinates
                'x_coord': 'X',
                'y_coord': 'Y',
                'z_coord': 'Z',
                
                # Equipment info
                'equipment_criticality': 'Equipment Criticality',
                'fault_detected': 'Fault Detected',
                'predictive_maintenance_trigger': 'Predictive Maintenance Trigger',
                
                # Time features
                'hour': 'hour',
                'dayofweek': 'dayofweek',
                'month': 'month',
                'day': 'day',
                'is_weekend': 'is_weekend',
                
                # Environmental sensors
                'footfall': 'footfall',
                'temp_mode': 'tempMode',
                'aq': 'AQ',
                'uss': 'USS',
                'cs': 'CS',
                'voc': 'VOC',
                'rp': 'RP',
                'ip': 'IP',
                
                # Derived features
                'heat_index': 'heat_index'
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
    
    # Complete list of features the model expects (from model.feature_names_in_)
    required_columns = [
        'footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature',
        'sensor_reading', 'true_strength', 'Sensor_ID', 'Timestamp', 'voltage',
        'current', 'temperature', 'Power (W)', 'humidity', 'vibration', 'Equipment_ID',
        'Operational Status', 'Fault Status', 'Failure Type', 'Maintenance Type',
        'Failure History', 'Repair Time (hrs)', 'Maintenance Costs (USD)',
        'Ambient Temperature (°C)', 'Ambient Humidity (%)', 'X', 'Y', 'Z',
        'Equipment Criticality', 'Fault Detected', 'Predictive Maintenance Trigger',
        'hour', 'dayofweek', 'month', 'day', 'is_weekend', 'heat_index'
    ]
    
    for col in required_columns:
        if col not in data:
            # Special handling for true_strength, which is likely the target variable
            if col == 'true_strength':
                data[col] = [None]  # Use None for target variable
                print(f"Added required column '{col}' with None value (target variable)")
            elif col == 'Timestamp' and 'timestamp' in form and form.timestamp.data:
                # Use the timestamp from the form if available
                data[col] = [form.timestamp.data.isoformat()]
                print(f"Added Timestamp column using form timestamp: {data[col][0]}")
            else:
                data[col] = [0]  # Use 0 as a default value for features
                print(f"Added required column '{col}' with default value 0")
        

    df = pd.DataFrame(data)
    
    # Ensure both 'temperature' and 'Temperature' columns exist
    if 'temperature' in df.columns and 'Temperature' not in df.columns:
        df['Temperature'] = df['temperature']
        print("Added Temperature column (copy of temperature)")
    elif 'Temperature' in df.columns and 'temperature' not in df.columns:
        df['temperature'] = df['Temperature']
        print("Added temperature column (copy of Temperature)")
    
    print(f"Final dataframe shape: {df.shape}")
    print(f"Final columns: {sorted(df.columns.tolist())}")
    
    return df 