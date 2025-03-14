# Student Performance Prediction ML Pipeline

This machine learning pipeline predicts student performance (`final_score`) using features derived from student demographics, behaviors, and academic background. The pipeline is modular, configurable, and designed for scalability.

---

## 📋 Project Overview

This project implements an end-to-end machine learning pipeline with the following stages:

1. **Data Loading**: Load raw data from CSV files or databases.
2. **Data Cleaning**: Handle missing values, duplicates, and outliers.
3. **Data Preprocessing**: Encode categorical variables, normalize numerical features, and handle imbalanced datasets.
4. **Feature Engineering**: Generate new features such as interaction terms and indices.
5. **Model Training**: Train multiple ML models with hyperparameter tuning.
6. **Model Evaluation**: Evaluate models using metrics like RMSE, MAE, and R².
7. **Model Selection**: Select the best-performing model based on evaluation metrics.
8. **Prediction**: Generate predictions on unseen data.

---

## 📊 Dataset Features

The dataset includes the following features:

### Categorical Features
- **System Location Code**: Location of the system (e.g., Zone_A, Zone_B).
- **Previous Cycle Plant Type**: Plant type from the previous cycle.
- **Plant Type**: Current plant type (e.g., Fruiting Vegetables, Leafy Greens).
- **Plant Stage**: Growth stage of the plant (e.g., maturity, seedling).

### Numerical Features
- **Temperature Sensor (°C)**: Temperature readings from sensors.
- **Humidity Sensor (%)**: Humidity levels recorded by sensors.
- **Light Intensity Sensor (lux)**: Light intensity measured in lux.
- **CO2 Sensor (ppm)**: Carbon dioxide levels in parts per million.
- **EC Sensor (dS/m)**: Electrical conductivity of the soil or water.
- **O2 Sensor (ppm)**: Oxygen levels in parts per million.
- **Nutrient N Sensor (ppm)**: Nitrogen levels in parts per million.
- **Nutrient P Sensor (ppm)**: Phosphorus levels in parts per million.
- **Nutrient K Sensor (ppm)**: Potassium levels in parts per million.
- **pH Sensor**: pH levels of the soil or water.
- **Water Level Sensor (mm)**: Water levels measured in millimeters.

### Datetime Features
- **sleep_time**: Time the system enters a low-power state.
- **wake_time**: Time the system resumes operation.

---

## ⚙️ Configuration

The pipeline is highly configurable through YAML files located in the `config/` directory:

### Preprocessing Configuration (`preprocessing_config.yaml`)
- **Missing Value Handling**: Strategies include mean, median, or mode imputation.
- **Categorical Encoding**: Options include one-hot encoding, label encoding, and frequency encoding.
- **Scaling**: Normalize numerical features using StandardScaler, MinMaxScaler, or RobustScaler.
- **Outlier Handling**: Options include IQR-based filtering or Z-score thresholds.
- **Feature Engineering**:
  - **Nutrient Balance Index**: Combines nitrogen, phosphorus, and potassium levels.
  - **Environmental Stress Index**: Derived from temperature, humidity, light intensity, CO2, and O2 levels.
  - **Water Quality Score**: Combines pH, EC, and water level measurements.

### Model Configuration (`model_config.yaml`)
- **Algorithms**: Includes Random Forest, XGBoost, LightGBM, and ElasticNet.
- **Hyperparameters**: Configurable for grid search or random search.
- **Adaptive Classifier**: Dynamically selects models based on temperature ranges.

### General Configuration (`config.yaml`)
- **Data Paths**: Paths for raw, processed, and feature-engineered data.
- **Logging**: Configurable logging levels and output formats.
- **Experiment Tracking**: Save metrics, models, and visualizations for analysis.

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/student_performance_ml_pipeline.git
   cd student_performance_ml_pipeline
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Running the Pipeline
To run the entire ML pipeline with the default configuration:
```bash
./run.sh
```

To run it with a custom configuration:
```bash
./run.sh --config config/custom_config.yaml
```

---

## 🛠️ Customization

### Adding New Models
1. Update the `model_config.yaml` file with the new model's parameters.
2. The pipeline will automatically include it in the training and evaluation process.

### Feature Engineering
1. Modify the `preprocessing_config.yaml` file to add new transformations.
2. Implement custom transformations in `src/features/feature_engineering.py`.

---

## 📂 Project Structure

```
student_performance_ml_pipeline/
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies
├── run.sh                      # Script to run the pipeline
├── config/                     # Configuration files
│   ├── config.yaml             # General pipeline configuration
│   ├── preprocessing_config.yaml # Preprocessing configuration
│   └── model_config.yaml       # Model-specific configurations
├── data/                       # Data directory
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│   └── features/               # Feature-engineered data
├── models/                     # Saved models
├── notebooks/                  # Notebooks for exploration
├── results/                    # Results and visualizations
├── src/                        # Source code
│   ├── data_loader.py          # Module for loading data
│   ├── data_cleaner.py         # Module for cleaning data
│   ├── data_preprocessor.py    # Module for preprocessing data
│   ├── feature_engineering.py  # Module for feature engineering
│   └── model_trainer.py        # Module for training models
├── tests/                      # Unit tests
└── pipeline.log                # Log file
```

---

## ✅ Testing

Run the tests to ensure all components are working correctly:
```bash
python -m pytest tests/
```

---

## 📈 Results

The pipeline outputs evaluation metrics and predictions in the `results/` directory. Key metrics include:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination

Visualizations are also generated to help interpret the results.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
