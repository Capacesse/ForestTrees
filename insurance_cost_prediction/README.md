Medical Insurance Cost Prediction: Dual-Model Approach
This project addresses the challenge of predicting individual medical insurance charges using a dual-model strategy focused on balancing high predictive accuracy with clear, actionable feature interpretability.

Primary Goal: Predict charges using an XGBoost Regressor.
Secondary Goal: Interpret the top factors using a regularised Lasso Regression baseline model.

🚀 Project Pipeline Structure
The solution follows a rigorous, three-stage pipeline to ensure reproducibility:
1. Data & Preprocessing (src/): Raw data is loaded, log-transformed (to normalise the skewed target variable), and then put through a Scikit-learn pipeline for feature scaling and encoding.
2. Training (src/train.py): The automated script trains both the Lasso (baseline) and XGBoost (final) models and saves all artifacts.
3. Prediction (notebooks/03_Prediction.ipynb): Loads the final XGBoost pipeline to make and evaluate predictions.

📈 Key Findings & Model Strategy
We adopted a dual-model approach to meet both the prediction and interpretation goals of the hackathon:
Best for Prediction (Accuracy)
* Model: XGBoost Regressor
* Key Metric: Low Root Mean Squared Error (RMSE) on the original dollar scale.
* Key Insight: This model automatically captures the complex, non-linear Smoker-BMI interaction within the data, leading to the best overall predictive performance.

Best for Interpretation
* Model: Lasso Regression
* Key Metric: Clear, traceable coefficients.
* Key Insight: The Lasso model highlights Smoker status and Age as the largest linear drivers of insurance cost, which is ideal for feature analysis.

📂 Repository Contents
This repository is organised into four main directories:
* data/: Contains the raw source data for the project.
   * Key File: insurance.csv
* src/: Holds all the production-ready Python scripts that make up the reusable machine learning pipeline.
   * Key Files: preprocessing.py, train.py
* models/: Stores all trained and saved model pipelines.
   * Key Files: xgboost_final_model.joblib, lasso_baseline_model.joblib, linear_regression_pca_pipeline.joblib
* notebooks/: Contains the interactive Jupyter Notebooks for analysis, training documentation, and demonstration.
   * Key Files: 01_EDA_Feature_Analysis.ipynb, 02_Training_Colab.ipynb, 03_Prediction.ipynb

▶️ How to Run the Project
To reproduce the full pipeline and predictions, follow these steps:
1. Setup
Clone the repository and install the necessary libraries:
# Clone the repository
git clone [YOUR-REPO-URL-HERE]
cd [YOUR-REPO-NAME]

# Install dependencies
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn

2. Run the Training Script
The train.py script executes the complete pipeline, trains both models, and saves the final XGBoost model to the models/directory.
python src/train.py

3. Review Analysis and Prediction
The notebooks serve as the deliverables:
* Interpretation & Baseline: Review notebooks/01_EDA_Feature_Analysis.ipynb to see the log-transform, feature engineering, and the Lasso model's coefficients that drive the interpretation section of the report.
* Prediction Demo: Run notebooks/03_Prediction.ipynb to load the saved XGBoost model and test its performance with new, hypothetical patient data.
