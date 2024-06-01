# Advanced Real Estate Price Prediction Using Weighted Least Squares Regression

Objective:
This project aims to develop a robust predictive model for real estate prices using a Weighted Least Squares (WLS) regression approach, focusing on minimizing the Root Mean Square Error (RMSE) to achieve accurate predictions.

Methodology:
Data Preparation:

Dataset: Utilized a dataset containing 500 entries with 45 variables, including features like lot area, building type, overall quality, and sale price.
Missing Values: Addressed missing values in categorical variables by treating them as a separate category.
Transformations: Applied Box-Cox transformations to normalize the data and handle non-linear patterns.
Feature Engineering:

Created new features such as 'NewArea' and 'YearBuiltandRemod' to capture combined effects.
Generated interaction terms based on visual inspection of data patterns.
Model Development:

Initial Regression: Performed Ordinary Least Squares (OLS) regression to identify key predictors and assess residual patterns.
Weighted Least Squares (WLS): Implemented WLS regression to account for heteroscedasticity, using residual variances from the OLS model as weights.
Model Evaluation:

Evaluated model performance using R-squared and adjusted R-squared metrics.
Analyzed residuals to ensure variance stability and identify any remaining patterns or outliers.
Predictions:

Applied the trained model to test data, using consistent data processing and transformations.
Transformed predictions back to the original scale using inverse Box-Cox transformation for final submission.
Key Findings:
The WLS regression model significantly improved prediction accuracy, achieving an R-squared value of 0.999, indicating a nearly perfect fit.
Interaction terms and transformed variables played a crucial role in capturing complex relationships within the data.
The model demonstrated robustness in handling heteroscedasticity, leading to reliable and stable predictions.
Conclusion:
This project successfully developed a highly accurate predictive model for real estate prices using advanced regression techniques. The approach of combining feature engineering, data normalization, and WLS regression proved effective in minimizing prediction errors, offering valuable insights for real estate market analysis.