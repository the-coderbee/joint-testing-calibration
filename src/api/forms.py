from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import (
    FloatField, StringField, SelectField, SubmitField,
    DateTimeField, IntegerField, BooleanField
)
from wtforms.validators import Optional, DataRequired, NumberRange

class ManualDataForm(FlaskForm):
    # Core Sensor Identifiers
    Sensor_ID = StringField('Sensor ID', validators=[Optional()])
    Equipment_ID = StringField('Equipment ID', validators=[Optional()])
    timestamp = DateTimeField('Timestamp (YYYY-MM-DD HH:MM:SS)', format='%Y-%m-%d %H:%M:%S', validators=[Optional()])

    # Primary sensor readings
    sensor_reading = FloatField('Sensor Reading', validators=[Optional()])
    footfall = FloatField('Footfall', validators=[Optional(), NumberRange(min=0)])
    tempMode = IntegerField('Temperature Mode', validators=[Optional()])  # renamed for model compatibility
    AQ = IntegerField('Air Quality (AQ)', validators=[Optional()])
    USS = IntegerField('USS', validators=[Optional()])
    CS = IntegerField('CS', validators=[Optional()])
    VOC = IntegerField('VOC', validators=[Optional()])
    RP = IntegerField('RP', validators=[Optional()])
    IP = IntegerField('IP', validators=[Optional()])

    # Electrical/mechanical measurements
    voltage = FloatField('Voltage (V)', validators=[Optional(), NumberRange(min=0)])
    current = FloatField('Current (A)', validators=[Optional(), NumberRange(min=0)])
    power = FloatField('Power (W)', validators=[Optional(), NumberRange(min=0)])
    temperature = FloatField('Temperature (°C)', validators=[Optional()])
    humidity = FloatField('Humidity (%)', validators=[Optional(), NumberRange(min=0, max=100)])
    vibration = FloatField('Vibration (m/s²)', validators=[Optional(), NumberRange(min=0)])

    # Maintenance and status fields
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

    # Spatial coordinates
    x_coord = FloatField('X Coordinate', validators=[Optional()])
    y_coord = FloatField('Y Coordinate', validators=[Optional()])
    z_coord = FloatField('Z Coordinate', validators=[Optional()])

    # Equipment criticality and predictive triggers
    equipment_criticality = SelectField(
        'Equipment Criticality',
        choices=[('', 'Select...'), ('High', 'High'), ('Medium', 'Medium'), ('Low', 'Low')],
        validators=[Optional()]
    )
    fault_detected = IntegerField('Fault Detected (0/1)', validators=[Optional(), NumberRange(min=0, max=1)])
    predictive_maintenance_trigger = IntegerField(
        'Predictive Maintenance Trigger (0/1)', validators=[Optional(), NumberRange(min=0, max=1)]
    )

    # Hidden auto-calculated features from timestamp
    hour = IntegerField('Hour', validators=[Optional()], render_kw={'style': 'display:none;'})
    dayofweek = IntegerField('Day of Week', validators=[Optional()], render_kw={'style': 'display:none;'})
    month = IntegerField('Month', validators=[Optional()], render_kw={'style': 'display:none;'})
    day = IntegerField('Day', validators=[Optional()], render_kw={'style': 'display:none;'})
    is_weekend = BooleanField('Is Weekend', validators=[Optional()], render_kw={'style': 'display:none;'})

    submit = SubmitField('Get Prediction')

    def process_timestamp(self):
        if self.timestamp.data:
            ts = self.timestamp.data
            self.hour.data = ts.hour
            self.dayofweek.data = ts.weekday()
            self.month.data = ts.month
            self.day.data = ts.day
            self.is_weekend.data = ts.weekday() >= 5

    def calculate_derived_features(self):
        if (self.voltage.data is not None and self.current.data is not None and not self.power.data):
            self.power.data = self.voltage.data * self.current.data
        if self.temperature.data is not None and self.humidity.data is not None:
            self.heat_index = self.temperature.data + 0.05 * self.humidity.data

class FileUploadForm(FlaskForm):
    csv_file = FileField('Upload CSV File', validators=[
        DataRequired(), FileAllowed(['csv'], 'CSV files only!')
    ])
    submit = SubmitField('Upload and Predict')
