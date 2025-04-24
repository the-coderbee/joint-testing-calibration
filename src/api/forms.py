from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import (FloatField, StringField, SelectField, SubmitField, 
                     DateTimeField, IntegerField, BooleanField)
from wtforms.validators import Optional, DataRequired, NumberRange

class ManualDataForm(FlaskForm):
    # Core Sensor Data Features
    sensor_id = StringField('Sensor ID', validators=[Optional()])
    timestamp = DateTimeField('Timestamp (YYYY-MM-DD HH:MM:SS)', format='%Y-%m-%d %H:%M:%S', validators=[Optional()])
    
    # Basic sensor measurements
    voltage = FloatField('Voltage (V)', validators=[Optional(), NumberRange(min=0)])
    current = FloatField('Current (A)', validators=[Optional(), NumberRange(min=0)])
    temperature = FloatField('Temperature (°C)', validators=[Optional()])
    power = FloatField('Power (W)', validators=[Optional(), NumberRange(min=0)])
    humidity = FloatField('Humidity (%)', validators=[Optional(), NumberRange(min=0, max=100)])
    vibration = FloatField('Vibration (m/s²)', validators=[Optional(), NumberRange(min=0)])
    equipment_id = StringField('Equipment ID', validators=[Optional()])
    
    # Status and Maintenance Information
    operational_status = SelectField(
        'Operational Status',
        choices=[('', 'Select...'), ('Operational', 'Operational'), ('Under Maintenance', 'Under Maintenance')],
        validators=[Optional()]
    )
    fault_status = SelectField(
        'Fault Status',
        choices=[('', 'Select...'), ('Fault Detected', 'Fault Detected'), ('No Fault', 'No Fault')],
        validators=[Optional()]
    )
    failure_type = StringField('Failure Type', validators=[Optional()])
    maintenance_type = StringField('Maintenance Type', validators=[Optional()])
    failure_history = StringField('Failure History', validators=[Optional()])
    repair_time = FloatField('Repair Time (hrs)', validators=[Optional(), NumberRange(min=0)])
    maintenance_costs = FloatField('Maintenance Costs (USD)', validators=[Optional(), NumberRange(min=0)])
    ambient_temperature = FloatField('Ambient Temperature (°C)', validators=[Optional()])
    ambient_humidity = FloatField('Ambient Humidity (%)', validators=[Optional(), NumberRange(min=0, max=100)])
    
    # Coordinates
    x_coord = FloatField('X Coordinate', validators=[Optional()])
    y_coord = FloatField('Y Coordinate', validators=[Optional()])
    z_coord = FloatField('Z Coordinate', validators=[Optional()])
    
    # Equipment Info
    equipment_criticality = SelectField(
        'Equipment Criticality',
        choices=[('', 'Select...'), ('High', 'High'), ('Medium', 'Medium'), ('Low', 'Low')],
        validators=[Optional()]
    )
    fault_detected = IntegerField('Fault Detected (0/1)', validators=[Optional(), NumberRange(min=0, max=1)])
    predictive_maintenance_trigger = IntegerField('Predictive Maintenance Trigger (0/1)', 
                                                 validators=[Optional(), NumberRange(min=0, max=1)])
    
    # Sensor readings from file 1
    footfall = FloatField('Footfall', validators=[Optional(), NumberRange(min=0)])
    temp_mode = IntegerField('Temperature Mode', validators=[Optional()])
    aq = IntegerField('Air Quality (AQ)', validators=[Optional()])
    uss = IntegerField('USS', validators=[Optional()])
    cs = IntegerField('CS', validators=[Optional()])
    voc = IntegerField('VOC', validators=[Optional()])
    rp = IntegerField('RP', validators=[Optional()])
    ip = IntegerField('IP', validators=[Optional()])
    
    # Additional fields for feature engineering
    # These will be auto-calculated if timestamp is provided
    hour = IntegerField('Hour (0-23)', validators=[Optional(), NumberRange(min=0, max=23)])
    dayofweek = IntegerField('Day of Week (0-6, 0=Monday)', validators=[Optional(), NumberRange(min=0, max=6)])
    month = IntegerField('Month (1-12)', validators=[Optional(), NumberRange(min=1, max=12)])
    day = IntegerField('Day (1-31)', validators=[Optional(), NumberRange(min=1, max=31)])
    is_weekend = BooleanField('Is Weekend', validators=[Optional()])
    
    submit = SubmitField('Get Prediction')

    def process_timestamp(self):
        """Extract datetime features if timestamp is provided but other fields aren't"""
        if self.timestamp.data:
            ts = self.timestamp.data
            if not self.hour.data:
                self.hour.data = ts.hour
            if not self.dayofweek.data:
                self.dayofweek.data = ts.weekday()
            if not self.month.data:
                self.month.data = ts.month
            if not self.day.data:
                self.day.data = ts.day
            if not self.is_weekend.data:
                self.is_weekend.data = ts.weekday() >= 5

    def calculate_derived_features(self):
        """Calculate any derived features like power and heat index"""
        # Calculate power if voltage and current are provided but power isn't
        if not self.power.data and self.voltage.data and self.current.data:
            self.power.data = self.voltage.data * self.current.data
            
        # Calculate heat index if temperature and humidity are provided
        if self.temperature.data and self.humidity.data:
            return self.temperature.data + 0.05 * self.humidity.data
        return None


class FileUploadForm(FlaskForm):
    csv_file = FileField('Upload CSV File', validators=[
        DataRequired(),
        FileAllowed(['csv'], 'CSV files only!')
    ])
    submit = SubmitField('Upload and Predict') 