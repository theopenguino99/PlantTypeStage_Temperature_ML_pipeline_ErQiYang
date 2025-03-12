import pytest
import pandas as pd
from src.data.datapreprocessor import DataPreprocessor

@pytest.fixture
def sample_data():
    data = {
        'student_id': [1, 2, 3, 4],
        'math_score': [88, 92, 80, 89],
        'reading_score': [95, 85, 78, 92],
        'writing_score': [90, 88, 84, 91]
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
    assert 'average_score' in engineered_data.columns
    assert engineered_data['average_score'].equals(
        (sample_data['math_score'] + sample_data['reading_score'] + sample_data['writing_score']) / 3
    )

def test_data_preprocessing_split_data(preprocessor, sample_data):
    train_data, test_data = preprocessor.split_data(sample_data)
    assert len(train_data) + len(test_data) == len(sample_data)
    assert len(train_data) > 0
    assert len(test_data) > 0