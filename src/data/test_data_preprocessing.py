import pytest
import pandas as pd
from data_preprocessor import DataPreprocessor

@pytest.fixture
def sample_data():
    data = {
        'System Location Code': ['A1', 'A2', 'A3', 'A4'],
        'Previous Cycle Plant Type': ['Type1', 'Type2', 'Type1', 'Type3'],
        'Plant Type': ['TypeA', 'TypeB', 'TypeA', 'TypeC'],
        'Plant Stage': ['Stage1', 'Stage2', 'Stage1', 'Stage3'],
        'Temperature Sensor (°C)': [25.0, 26.5, 24.0, 23.5],
        'Humidity Sensor (%)': [55.0, 60.0, None, 58.0],
        'Light Intensity Sensor (lux)': [1500, 1600, 1550, None],
        'CO2 Sensor (ppm)': [400, 420, 410, 430],
        'EC Sensor (dS/m)': [1.2, 1.3, 1.1, 1.4],
        'O2 Sensor (ppm)': [21, 22, 20, 23],
        'Nutrient N Sensor (ppm)': ['10', '12', '11', '13'],
        'Nutrient P Sensor (ppm)': ['5', '6', '5', '7'],
        'Nutrient K Sensor (ppm)': ['8', '9', '8', '10'],
        'pH Sensor': [6.5, 6.8, 6.7, 6.6],
        'Water Level Sensor (mm)': [100, 105, 95, None]
    }
    return pd.DataFrame(data)

@pytest.fixture
def preprocessor():
    return DataPreprocessor()

def test_data_preprocessing_initialization(preprocessor):
    assert isinstance(preprocessor, DataPreprocessor)

def test_data_preprocessing_clean_data(preprocessor, sample_data):
    cleaned_data = preprocessor.clean_data(sample_data)
    assert not cleaned_data.isnull().values.any()

def test_data_preprocessing_feature_engineering(preprocessor, sample_data):
    engineered_data = preprocessor.feature_engineering(sample_data)
    # Assuming feature engineering adds a 'total_nutrients' column
    assert 'total_nutrients' in engineered_data.columns
    assert engineered_data['total_nutrients'].equals(
        engineered_data['Nutrient N Sensor (ppm)'].astype(float) +
        engineered_data['Nutrient P Sensor (ppm)'].astype(float) +
        engineered_data['Nutrient K Sensor (ppm)'].astype(float)
    )

def test_data_preprocessing_split_data(preprocessor, sample_data):
    train_data, test_data = preprocessor.split_data(sample_data)
    assert len(train_data) + len(test_data) == len(sample_data)
    assert len(train_data) > 0
    assert len(test_data) > 0