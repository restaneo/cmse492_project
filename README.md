# cmse492_project
Project Overview

This project predicts combined fuel economy (MPG) for US vehicles using machine learning techniques applied to EPA (Environmental Protection Agency) vehicle data. The goal is to develop accurate predictive models that can estimate vehicle fuel efficiency based on various vehicle characteristics such as engine specifications, drivetrain type, and manufacturer details.

Author: James Restaneo

Course: CMSE 492 

Institution: Michigan State University

GitHub Repository: https://github.com/restaneo/cmse492_project

Problem Statement
Vehicle fuel economy is a critical factor for consumers, policymakers, and environmental planners. Accurate prediction of fuel economy can help:

Consumers make informed purchasing decisions
Manufacturers optimize vehicle design for efficiency
Policymakers develop effective environmental regulations
Environmental planners estimate transportation emissions

This project uses machine learning to predict combined MPG (city + highway) based on vehicle characteristics, providing insights into which features most significantly impact fuel efficiency.
Dataset
Source
The dataset is based on vehicle testing data from the US Environmental Protection Agency (EPA). The EPA conducts comprehensive fuel economy testing at the National Vehicle and Fuel Emissions Laboratory in Ann Arbor, Michigan, under controlled conditions to provide standardized fuel economy estimates.
Dataset Characteristics

Size: 2,500 vehicle records
Time Period: Model years 2020-2024
Features: 12 variables including:

Categorical: Make, Model, Transmission Type, Drive Type, Fuel Type
Numerical: Year, Engine Displacement (L), Number of Cylinders, City MPG, Highway MPG, Combined MPG, CO2 Emissions


Target Variable: Combined MPG (comb_mpg)
Data Quality: No missing values

Key Statistics

Average Combined MPG: 25.4 MPG
Range: 12.2 - 112.7 MPG
Engine Displacement Range: 1.0 - 6.0 L
Cylinder Range: 3 - 12 cylinders


Prerequisites

Python 3.8 or higher
pip package manager

Installation

Clone the repository:

bash   git clone https://github.com/restaneo/cmse492_project.git
   cd cmse492_project

Install required packages:

bash   pip install -r requirements.txt

Verify installation:

bash   python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All packages installed successfully!')"
Dependencies
The project requires the following Python packages:

pandas (>=2.0.0) - Data manipulation and analysis
numpy (>=1.24.0) - Numerical computing
scikit-learn (>=1.3.0) - Machine learning algorithms
matplotlib (>=3.7.0) - Data visualization
seaborn (>=0.12.0) - Statistical data visualization
jupyter (>=1.0.0) - Interactive notebooks

Project Workflow
1. Data Acquisition and Preprocessing

Load EPA vehicle data from CSV files
Perform data quality checks (missing values, outliers, data types)
Split data into training, validation, and test sets (70/15/15)
Feature engineering: encode categorical variables, create interaction features
Feature scaling: standardization for numerical features

2. Exploratory Data Analysis

Distribution analysis of target variable (combined MPG)
Correlation analysis between features
Visualization of relationships between engine characteristics and fuel economy
Analysis of fuel economy by vehicle make and fuel type
Identification of outliers and anomalies

3. Model Development
The project implements three models of increasing complexity:
Baseline Model (Simple):

Linear Regression with basic features
Serves as performance benchmark
Current Test R²: 0.027 (needs improvement with feature engineering)

Intermediate Model (Planned):

Random Forest Regressor
Ensemble method with decision trees
Handles non-linear relationships and feature interactions

Advanced Model (Planned):

Gradient Boosting (XGBoost/LightGBM)
Deep Neural Network
Hyperparameter tuning via grid search/random search

4. Model Evaluation

Primary Metric: Root Mean Squared Error (RMSE)
Secondary Metrics: Mean Absolute Error (MAE), R² Score
Validation: 5-fold cross-validation
Comparison: Performance comparison across all models
Interpretation: Feature importance analysis using SHAP values

Current Progress

HW08

Dataset loaded and explored (2,500 vehicles)
Created 5 comprehensive visualizations
Baseline Linear Regression model trained
Baseline performance: Test R² = 0.027, RMSE = 23.51 MPG


Next Steps:

Feature engineering and advanced preprocessing
Implementation of intermediate and advanced models
Hyperparameter tuning and model optimization
Comprehensive model comparison and interpretation



Key Findings (Preliminary)

Strong negative correlation between engine displacement and fuel economy (r = -0.43)
Electric and hybrid vehicles show significantly higher fuel economy (80-100+ MPG equivalent)
Number of cylinders inversely correlates with fuel economy
Wide variation in fuel economy across manufacturers (top performers: Tesla, Toyota, Honda)
Baseline model demonstrates room for improvement through feature engineering and advanced algorithms

Future Enhancements

Incorporate additional features (vehicle weight, aerodynamics)
Time series analysis of fuel economy trends over years
Multi-output prediction (simultaneous prediction of city and highway MPG)
Model deployment as a web application for real-time predictions
Integration with real-world vehicle databases

Contributing
This is an academic project for CMSE 492. However, suggestions and feedback are welcome via issues or pull requests.
License
This project is part of academic coursework at Michigan State University. 

Acknowledgments

Data Source: US Environmental Protection Agency (EPA) - FuelEconomy.gov
Course: CMSE 492 
Institution: Michigan State University, Department of Computational Mathematics, Science and Engineering
Instructor: Luciano Silvestri

Contact
For questions or collaboration inquiries, please contact:

Email: restaneo@msu.edu
GitHub: restaneo
