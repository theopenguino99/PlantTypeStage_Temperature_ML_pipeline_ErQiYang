# 🌱 Plant Type-Stage/ Temperature Prediction ML Pipeline

**Name:** ER Qi Yang  
**Email:** e0148703@u.nus.edu

## 🎯 Project Focus

This machine learning pipeline predicts temperature conditions ("Temperature Sensor (°C)") within a farm's closed environment, ensuring optimal plant growth. Additionally, it contains models to categorize the combined ("Plant Type-Stage") based on sensor data, aiding in strategic planning and resource allocation.

## 📋 Project Overview

This project implements an end-to-end machine learning pipeline with the following stages:

1. **Data Loading**: Load raw data from CSV files or databases
2. **Data Cleaning**: Handle missing values, duplicates, and outliers
3. **Data Preprocessing**: Encode categorical variables, normalize numerical features
4. **Feature Engineering**: Generate new features such as interaction terms and indices
5. **Model Training**: Train multiple ML models with hyperparameter tuning
6. **Model Evaluation**: Evaluate models using metrics like RMSE, MAE, and R²
7. **Model Selection**: Select the best-performing model based on evaluation metrics
8. **Prediction**: Generate predictions on unseen data

## 📊 Dataset Features

### Categorical Features
- **System Location Code**: Location of the system (e.g., Zone_A, Zone_B)
- **Previous Cycle Plant Type**: Plant type from the previous cycle
- **Plant Type**: Current plant type (e.g., Fruiting Vegetables, Leafy Greens)
- **Plant Stage**: Growth stage of the plant (e.g., maturity, seedling)

### Numerical Features
| Feature | Description |
|---------|-------------|
| Temperature Sensor (°C) | Temperature readings from sensors |
| Humidity Sensor (%) | Humidity levels recorded by sensors |
| Light Intensity Sensor (lux) | Light intensity measured in lux |
| CO2 Sensor (ppm) | Carbon dioxide levels in parts per million |
| EC Sensor (dS/m) | Electrical conductivity of the soil or water |
| O2 Sensor (ppm) | Oxygen levels in parts per million |
| Nutrient N Sensor (ppm) | Nitrogen levels in parts per million |
| Nutrient P Sensor (ppm) | Phosphorus levels in parts per million |
| Nutrient K Sensor (ppm) | Potassium levels in parts per million |
| pH Sensor | pH levels of the soil or water |
| Water Level Sensor (mm) | Water levels measured in millimeters |

## 🔭 Exploratory Data Analysis

Our EDA revealed several data quality issues:

### Data Cleanliness Issues
- **ppm Sensor in string format**: Some readings contain spaces and "ppm" text
- **Inconsistent capitalization**: Plant Type and Plant Stage contain mixed case
- **Negative Values**: Temperature and O2 sensor values contain negative values
  
  ![Negative Values Scatter Plot](image-3.png)
  
  Analysis shows these negative values follow similar distributions to positive values:
  
  ![Distribution Analysis 1](image-1.png)
  ![Distribution Analysis 2](image-2.png)

### Feature Correlation
After initial cleaning, we identified high correlation between Nutrient P and N (>0.7), suggesting potential for creating a Nutrient Balance index:

![Correlation Heatmap](image.png)

### Data Quality Issues
- **Duplicates**: 7,489 duplicate records found in the original 57,489 records
- **Null values**: Several features have missing values as shown below:

| Feature | Missing Values |
|---------|---------------|
| System Location Code | 0 |
| Previous Cycle Plant Type | 0 |
| Plant Type | 0 |
| Plant Stage | 0 |
| Temperature Sensor (°C) | 8,689 |
| Humidity Sensor (%) | 38,867 |
| Light Intensity Sensor (lux) | 4,278 |
| CO2 Sensor (ppm) | 0 |
| EC Sensor (dS/m) | 0 |
| O2 Sensor (ppm) | 0 |
| Nutrient N Sensor (ppm) | 9,974 |
| Nutrient P Sensor (ppm) | 5,698 |
| Nutrient K Sensor (ppm) | 3,701 |
| pH Sensor | 0 |
| Water Level Sensor (mm) | 8,642 |

## ⚙️ Configuration

The pipeline is highly configurable through YAML files located in the `config/` directory:

### Preprocessing Configuration (`preprocessing_config.yaml`)
- **Missing Value Handling**: Mean, median, or mode imputation
- **Categorical Encoding**: One-hot encoding, label encoding, or frequency encoding
- **Scaling**: StandardScaler, MinMaxScaler, or RobustScaler
- **Outlier Handling**: IQR-based filtering or Z-score thresholds
- **Feature Engineering**:
  - **Nutrient Balance Index**: Combines N, P, K levels
  - **Environmental Stress Index**: Derived from temperature, humidity, light, CO2, and O2
  - **Water Quality Score**: Combines pH, EC, and water level measurements

### Model Configuration (`model_config.yaml`)
- **Algorithms**: Random Forest, XGBoost, LightGBM, ElasticNet, MLP, and Ensemble models
- **Hyperparameters**: Configurable for grid search or random search

### General Configuration (`config.yaml`)
- **Data Paths**: Locations for raw, processed, and feature-engineered data
- **Logging**: Configurable logging levels and output formats
- **Experiment Tracking**: Save metrics, models, and visualizations

## 🧹 Data Cleaning and Processing

| Class | Actions |
|-------|---------|
| **DataCleaner** | • Extract and convert nutrient values from strings to float<br>• Standardize case in Plant Type and Plant Stage<br>• Convert negative sensor values to positive<br>• Remove duplicates<br>• Impute missing values<br>• Handle outliers using IQR |
| **FeatureSelector** | • Create "Plant Type-Stage" target for classification<br>• Select features via variance, correlation, or importance |
| **FeatureEngineer** | • Create Nutrient Balance Index<br>• Create Environmental Stress Index<br>• Create Water Quality Score |
| **FeatureProcessor** | • Drop user-specified columns<br>• Encode categorical data<br>• Scale numerical data |

## 🧠 Model Selection Rationale

Our model selection addresses both regression (temperature prediction) and classification (Plant Type-Stage) tasks:

### Regression Models

1. **Linear Regression**
   - **Why**: Simple, interpretable baseline model
   - **Use Case**: Establishing linear relationships between sensor data and temperature

2. **Ridge Regression**
   - **Why**: Handles multicollinearity with L2 regularization
   - **Use Case**: Stabilizing model with highly correlated features (e.g., nutrient sensors)

3. **Random Forest**
   - **Why**: Captures non-linear relationships and feature interactions
   - **Use Case**: Modeling complex environmental interactions and providing feature importance

4. **Gradient Boosting**
   - **Why**: Builds models sequentially, correcting previous errors
   - **Use Case**: Handling heterogeneous sensor data with high accuracy

5. **XGBoost**
   - **Why**: Optimized implementation of gradient boosting
   - **Use Case**: Efficiently handling large datasets with missing values

6. **Multi-Layer Perceptron (MLP)**
   - **Why**: Captures highly complex non-linear relationships
   - **Use Case**: Modeling intricate interactions between numerous sensor readings

### Classification Models

7. **Ensemble Models**
   - **Why**: Combines predictions from multiple models
   - **Use Case**: Improving "Plant Type-Stage" classification accuracy and robustness

### Model Selection Strategy
- **Baseline Models**: Linear regression, Gradient Boosting, XGBoost, and Random Forest
- **Advanced Models**: Ensemble learning and deep learning for complex patterns

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/theopenguino99/aiip5-Er-Qi-Yang-227J.git
   cd aiip5-Er-Qi-Yang-227J
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Pipeline
To run the entire ML pipeline with the default configuration:
```bash
./run.sh
```

## 🛠️ Customization

### Adding New Models
1. Update the `model_config.yaml` file with the new model's parameters
2. The pipeline will automatically include it in the training and evaluation process

### Feature Engineering
1. Modify the `preprocessing_config.yaml` file to add new transformations
2. Implement custom transformations in `src/features/feature_engineering.py`

## 📂 Project Structure

```
student_performance_ml_pipeline/
├── README.md                               # Project documentation
├── requirements.txt                        # Dependencies
├── run.sh                                  # Script to run the pipeline
├── eda.ipynb                               # Notebook for Exploratory Data Analysis
├── config/                                 # Configuration files
│   ├── config.yaml                         # General pipeline configuration
│   ├── preprocessing_config.yaml           # Preprocessing configuration
│   └── model_config.yaml                   # Model-specific configurations
├── data/                                   # Data directory
│   ├── agri.db                             # Raw data
│   └── processed/                          # Processed data
├── models/                                 # Saved models (in .pkl format)
├── results/                                # Results and visualizations
└── src/                                    # Source code
    ├── config_loader.py                    # Module for loading configurations
    ├── data_loader.py                      # Module for loading data
    ├── data_cleaner.py                     # Module for cleaning data
    ├── data_preprocessor.py                # Module for preprocessing data
    ├── feature_engineering.py              # Module for feature engineering
    ├── feature_selection.py                # Module for feature selection
    ├── model_trainer.py                    # Module for training models
    ├── model_evaluator.py                  # Module for evaluating models
    ├── temperature_regression_models.py    # Module for regression models
    ├── plant_type_stage_classification.py  # Module for classification models
    └── main.py                             # Main module to execute
```

## 📈 Results

The pipeline outputs evaluation metrics and predictions in the `results/` directory.

### Regression Metrics
- **RMSE (Root Mean Squared Error)**: Measures error magnitude, sensitive to outliers
- **MSE (Mean Squared Error)**: Penalizes larger errors more heavily
- **MAE (Mean Absolute Error)**: Average absolute error, less sensitive to outliers
- **R² (Coefficient of Determination)**: Explains variance explained by the model (0-1)

### Classification Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1-Score**: Harmonic mean of precision and recall, useful for imbalanced datasets

## 🤝 Contributing

1. Fork the repository
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
5. Open a Pull Request
