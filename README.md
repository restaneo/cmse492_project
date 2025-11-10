# cmse492_project
# Vehicle Fuel Economy Prediction Using Machine Learning

**Author:** James Restaneo  
**Email:** restaneo@msu.edu  
**Course:** CMSE 492 - Computational Data Science
**Semester:** Fall 2025  
**Institution:** Michigan State University

## Project Overview

This project develops and compares three machine learning models (Ridge Regression, Random Forest, XGBoost) to predict combined fuel economy (MPG) for vehicles using EPA's 2020-2024 dataset. The goal is to improve upon a baseline linear regression model (R²=0.027) by implementing advanced techniques including feature engineering, hyperparameter optimization, and model interpretability through SHAP analysis.

## Repository Contents

### Proposal Documents
- `Restaneo_James_CMSE492_ProjectProposal.pdf` - Complete project proposal (6 pages)
- `Restaneo_James_CMSE492_ProjectProposal.tex` - LaTeX source code for proposal

### Visualization Figures
- `01_target_distribution.png` - Combined MPG distribution histogram
- `02_engine_characteristics.png` - Engine displacement and cylinder analysis
- `03_make_and_fuel_type.png` - Top manufacturers and fuel type distribution
- `04_correlation_heatmap.png` - Feature correlation matrix
- `05_baseline_model_performance.png` - Baseline model results
- `gantt_chart.png` - Project timeline (November 4 - December 8, 2025)

### Code
- `generate_proposal_figures.py` - Python script to generate all data visualizations
- `preliminary_analysis.ipynb` - Jupyter notebook with exploratory data analysis (if available)

### Data
- `vehicles_2024.csv` - EPA fuel economy dataset (2,500 vehicles, 2020-2024 model years)

## Dataset Information

**Source:** EPA Fuel Economy Data (https://www.fueleconomy.gov)  
**Size:** 2,500 vehicle records  
**Years:** 2020-2024  
**Features:** 83 total (categorical and numerical)  
**Target Variable:** Combined MPG (city/highway)  
**Range:** 12-136 MPG

### Key Features
- **Categorical:** Make, model, transmission type, drive configuration, fuel type, vehicle class
- **Numerical:** Engine displacement (L), cylinders, city MPG, highway MPG, annual fuel cost, CO₂ emissions (g/mi)

## Preliminary Analysis Results

### Baseline Model Performance
Simple linear regression using only 3 features (year, displacement, cylinders):
- **Test R²:** 0.027 (2.7% variance explained)
- **Test RMSE:** 23.51 MPG
- **Test MAE:** ~18-20 MPG

This poor performance demonstrates significant room for improvement and justifies the need for:
1. Advanced feature engineering (one-hot encoding, polynomial features)
2. Non-linear models (Random Forest, XGBoost)
3. Regularization techniques (Ridge regression)
4. Hyperparameter optimization

### Key Findings
- Strong negative correlation between engine displacement and fuel economy (r = -0.82)
- Strong negative correlation between cylinder count and fuel economy (r = -0.76)
- Very strong negative correlation between CO₂ emissions and fuel economy (r = -0.98)
- Electric and hybrid vehicles show dramatically higher efficiency (80-100+ MPG)

## Proposed Methodology

### Models
1. **Ridge Regression** - Linear model with L2 regularization (interpretable baseline)
2. **Random Forest** - Ensemble of decision trees (captures non-linear relationships)
3. **XGBoost** - Gradient boosting (state-of-the-art performance)

### Evaluation
- **Primary Metric:** RMSE (Root Mean Squared Error)
- **Secondary Metrics:** R² (coefficient of determination), MAE (Mean Absolute Error)
- **Validation:** 5-fold cross-validation, stratified by vehicle class
- **Interpretability:** SHAP (SHapley Additive exPlanations) values

### Success Criteria
- Test RMSE < 12 MPG (49% improvement over baseline)
- Test R² > 0.75 (explains at least 75% of variance)
- Complete pipeline executes in < 2 hours on standard hardware

## Project Timeline

- **Week 1 (Nov 4-10):** Data preprocessing and feature engineering
- **Week 2 (Nov 11-17):** Ridge Regression implementation
- **Week 3 (Nov 18-24):** Random Forest development
- **Week 4 (Nov 25-Dec 1):** XGBoost implementation
- **Week 5 (Dec 2-8):** Presentation, SHAP analysis, and final report

## How to Generate Figures

```bash
# Ensure you have vehicles_2024.csv in the same directory
python generate_proposal_figures.py
```

This will create all 5 data visualization figures (01-05) used in the proposal.

### Requirements
```bash
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0

## Expected Contributions

1. **Open-Source Tools:** Reproducible Python codebase addressing the lack of publicly available fuel economy prediction models
2. **Empirical Evidence:** Quantitative comparison of Ridge, Random Forest, and XGBoost on real-world data
3. **Interpretable Insights:** SHAP analysis revealing which specifications most influence fuel economy
4. **Methodological Template:** End-to-end ML workflow documentation for future projects

## Acknowledgments

This project proposal was developed with assistance from Claude (Anthropic), an AI assistant used for literature research, data exploration planning, and document preparation.

## References

1. US EPA. "Fuel Economy Data." https://www.fueleconomy.gov/feg/download.shtml (2024)
2. Chen, T., Guestrin, C. "XGBoost: A Scalable Tree Boosting System." Proc. 22nd ACM SIGKDD (2016)
3. Breiman, L. "Random Forests." Machine Learning 45(1), 5-32 (2001)
4. Lundberg, S.M., Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." Advances in NIPS (2017)

## License

This project is for academic purposes as part of CMSE 492 at Michigan State University.

## Contact

**James Restaneo**  
restaneo@msu.edu  
Michigan State University  
Department of Computational Mathematics, Science and Engineering

Contact
For questions or collaboration inquiries, please contact:

Email: restaneo@msu.edu
GitHub: restaneo
