# Student Performance Prediction ML Pipeline

This machine learning pipeline is designed to predict student performance (final_score) based on various features related to student demographics, behaviors, and academic background.

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting student performance. The pipeline includes data cleaning, preprocessing, feature engineering, model training, evaluation, and prediction.

### Dataset Features

The dataset contains the following features:
- index: Unique identifier
- number_of_siblings: Number of siblings the student has
- direct_admission: Whether the student was directly admitted
- CCA: Co-curricular activities
- learning_style: Student's learning style
- student_id: Unique student identifier
- gender: Student's gender
- tuition: Whether the student takes tuition classes
- final_test: Score in the final test
- n_male: Number of male students in class
- n_female: Number of female students in class
- age: Student's age
- hours_per_week: Study hours per week
- attendance_rate: Attendance rate
- sleep_time: Time the student goes to sleep
- wake_time: Time the student wakes up
- mode_of_transport: Transportation mode to school
- bag_color: Color of the student's bag

The target variable is `final_score` (not shown in the feature list but mentioned as the label).

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup
1. Clone this repository
```bash
git clone https://github.com/yourusername/student_performance_ml_pipeline.git
cd student_performance_ml_pipeline
```

2. Install the required packages
```bash
pip install -r requirements.txt
```

## Usage

### Running the Pipeline
To run the entire ML pipeline with default configuration:

```bash
./run.sh
```

Or you can run it with a custom configuration:

```bash
./run.sh --config config/custom_config.yaml
```

### Pipeline Components

1. **Data Loading**: Loads the raw data from the data source
2. **Data Cleaning**: Handles missing values, duplicates, and inconsistencies
3. **Data Preprocessing**: Transforms categorical variables, normalizes numerical features
4. **Feature Engineering**: Creates new features, selects relevant features
5. **Model Training**: Trains various ML models with hyperparameter optimization
6. **Model Evaluation**: Evaluates models using appropriate metrics
7. **Model Selection**: Selects the best performing model
8. **Prediction**: Makes predictions on new data

## Customization

### Configuration Files
The pipeline can be customized through YAML configuration files located in the `config/` directory:

- `config.yaml`: General pipeline configuration
- `model_config.yaml`: Model-specific configurations
- `preprocessing_config.yaml`: Data preprocessing configurations

### Adding New Models
To add a new model:

1. Update the `model_config.yaml` file with the new model's parameters
2. The pipeline will automatically include it in the training and evaluation process

### Feature Engineering
To customize feature engineering:

1. Modify the `preprocessing_config.yaml` file
2. Add new feature transformations in `src/features/feature_engineering.py`

## Project Structure

```
student_performance_ml_pipeline/
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies
├── run.sh                      # Script to run the pipeline
├── config/                     # Configuration files
├── data/                       # Data directory
├── models/                     # Saved models
├── notebooks/                  # Notebooks for exploration
├── tests/                      # Unit tests
└── src/                        # Source code
```

## Testing

Run the tests to ensure all components are working correctly:

```bash
python -m pytest tests/
```

## Results

The pipeline outputs evaluation metrics and predictions in the `results/` directory. Visualizations are also generated to help interpret the results.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
