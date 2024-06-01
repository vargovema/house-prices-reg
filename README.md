# Advanced Real Estate Price Prediction Using Weighted Least Squares Regression

## Objective
This project aims to develop a robust predictive model for real estate prices using a Weighted Least Squares (WLS) regression approach, focusing on minimizing the Root Mean Square Error (RMSE) to achieve accurate predictions.

## Methodology

### Data Preparation:

#### Dataset

Utilized a dataset containing 500 entries with 45 variables, including features like lot area, building type, overall quality, and sale price.
Missing Values: Addressed missing values in categorical variables by treating them as a separate category.
Transformations: Applied Box-Cox transformations to normalize the data and handle non-linear patterns.

```python
num_cols_tranform = ['LotArea','BsmtSF','X1stFlrSF', 'X2ndFlrSF', 'LivAreaSF', 'GarageArea', 'NewArea']

lambdas = []
for i in num_cols_tranform:
    df[i], _ = stats.boxcox(df[i]+1)
    lambdas.append(_)
```

#### Feature Engineering

Created new features such as 'NewArea' and 'YearBuiltandRemod' to capture combined effects.
Generated interaction terms based on visual inspection of data patterns.

```python
mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'A1':0}

df['ExterQual'] = df['ExterQual'].map(mapping).replace(mapping).astype('int') # Using map() function
df['ExterCond'] = df['ExterCond'].map(mapping).replace(mapping).astype('int') # Using map() function
df['BsmtQual'] = df['BsmtQual'].map(mapping).replace(mapping).astype('int') # Using map() function
df['BsmtCond'] = df['BsmtCond'].map(mapping).replace(mapping).astype('int') # Using map() function
df['FireplaceQu'] = df['FireplaceQu'].map(mapping).replace(mapping).astype('int') # Using map() function
df['GarageQual'] = df['GarageQual'].map(mapping).replace(mapping).astype('int') # Using map() function
df['GarageCond'] = df['GarageCond'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Ex': 3, 'Gd': 2, 'TA': 1, 'Fa': 0}

df['KitchenQual'] = df['KitchenQual'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'A1': 0}

df['BsmtExposure'] = df['BsmtExposure'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'A1': 0}

df['BsmtRating'] = df['BsmtRating'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Typ': 3, 'Min': 2, 'Mod': 1, 'Maj': 0}

df['Functional'] = df['Functional'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Fin': 1, 'RFn': 1, 'Unf': 0, 'A1': 0}

df['GarageFinish'] = df['GarageFinish'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Y': 1, 'N': 0}

df['CentralAir'] = df['CentralAir'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Attchd': 1, 'Other': 1, 'Detchd': 0, 'A1': 0}

df['GarageType'] = df['GarageType'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Abnorml': 1, 'Family': 0, 'Normal': 0}

df['SaleCondition_Abnorml'] = df['SaleCondition'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Abnorml': 0, 'Family': 1, 'Normal': 0}

df['SaleCondition_Family'] = df['SaleCondition'].map(mapping).replace(mapping).astype('int') # Using map() function

df.drop('SaleCondition', axis=1, inplace=True)

df['HasBsmt'] = df['BsmtSF'].apply(lambda con: False if con == 0 else True).astype('int')
df['HasGarage'] = df['GarageCars'].apply(lambda con: False if con == 0 else True).astype('int')
df['HasFireplace'] = df['Fireplaces'].apply(lambda con: False if con == 0 else True).astype('int')
df['HasPool'] = df['PoolArea'].apply(lambda con: False if con == 0 else True).astype('int')
```

<img src="out/fig1.png" alt="Example Image" width="200"/>

### Model Development

#### Initial Regression

Performed Ordinary Least Squares (OLS) regression to identify key predictors and assess residual patterns.

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              SalePrice   R-squared:                       0.777
Model:                            OLS   Adj. R-squared:                  0.768
Method:                 Least Squares   F-statistic:                     86.50
Date:                Wed, 21 Jun 2023   Prob (F-statistic):          3.11e-140
Time:                        23:31:14   Log-Likelihood:                 705.26
No. Observations:                 492   AIC:                            -1371.
Df Residuals:                     472   BIC:                            -1287.
Df Model:                          19                                         
Covariance Type:            nonrobust                                         
=====================================================================================================
                                        coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------------
const                                 5.3027      0.131     40.423      0.000       5.045       5.560
NewArea                               0.2938      0.032      9.325      0.000       0.232       0.356
YearBuiltandRemod                     0.0161      0.005      3.165      0.002       0.006       0.026
GarageCars                            0.0216      0.005      4.365      0.000       0.012       0.031
BsmtQual                              0.0129      0.004      2.931      0.004       0.004       0.022
GarageFinish                          0.0204      0.007      2.762      0.006       0.006       0.035
KitchenQual                           0.0051      0.007      0.767      0.443      -0.008       0.018
OverallCond                           0.0111      0.003      3.968      0.000       0.006       0.017
LotArea                               0.0220      0.005      4.304      0.000       0.012       0.032
BldgType_Twnhs                       -0.0361      0.016     -2.257      0.024      -0.067      -0.005
Zone_RL                               0.0132      0.009      1.546      0.123      -0.004       0.030
Zone_FV                               0.0421      0.015      2.793      0.005       0.012       0.072
LotConfig_Inside                     -0.0088      0.006     -1.421      0.156      -0.021       0.003
FireplaceQu                           0.0049      0.002      2.796      0.005       0.001       0.008
Zone_C                               -0.0866      0.036     -2.411      0.016      -0.157      -0.016
Foundation_PConc                      0.0004      0.008      0.056      0.956      -0.015       0.016
OverallQual:SaleCondition_Abnorml     0.0132      0.006      2.248      0.025       0.002       0.025
NewArea:SaleCondition_Abnorml        -0.0173      0.008     -2.059      0.040      -0.034      -0.001
OverallQual:ExterQual                 0.0055      0.001      5.016      0.000       0.003       0.008
NewArea:ExterQual                    -0.0052      0.003     -1.891      0.059      -0.011       0.000
==============================================================================
Omnibus:                        0.840   Durbin-Watson:                   2.021
Prob(Omnibus):                  0.657   Jarque-Bera (JB):                0.944
Skew:                          -0.077   Prob(JB):                        0.624
Kurtosis:                       2.850   Cond. No.                     1.52e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.52e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
```

<img src="out/fig7.png" alt="Example Image" width="700"/>

#### Weighted Least Squares (WLS)

Implemented WLS regression to account for heteroscedasticity, using residual variances from the OLS model as weights.

```
                            WLS Regression Results                            
==============================================================================
Dep. Variable:              SalePrice   R-squared:                       0.999
Model:                            WLS   Adj. R-squared:                  0.999
Method:                 Least Squares   F-statistic:                 2.678e+04
Date:                Wed, 21 Jun 2023   Prob (F-statistic):               0.00
Time:                        23:31:14   Log-Likelihood:                 1006.3
No. Observations:                 492   AIC:                            -1973.
Df Residuals:                     472   BIC:                            -1889.
Df Model:                          19                                         
Covariance Type:            nonrobust                                         
=====================================================================================================
                                        coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------------
const                                 5.3089      0.016    326.153      0.000       5.277       5.341
NewArea                               0.2901      0.004     79.726      0.000       0.283       0.297
YearBuiltandRemod                     0.0162      0.001     16.845      0.000       0.014       0.018
GarageCars                            0.0213      0.001     21.111      0.000       0.019       0.023
BsmtQual                              0.0121      0.001     11.625      0.000       0.010       0.014
GarageFinish                          0.0207      0.001     20.077      0.000       0.019       0.023
KitchenQual                           0.0043      0.001      3.799      0.000       0.002       0.007
OverallCond                           0.0118      0.000     27.498      0.000       0.011       0.013
LotArea                               0.0229      0.001     26.202      0.000       0.021       0.025
BldgType_Twnhs                       -0.0371      0.003    -10.933      0.000      -0.044      -0.030
Zone_RL                               0.0117      0.001      9.625      0.000       0.009       0.014
Zone_FV                               0.0398      0.003     13.294      0.000       0.034       0.046
LotConfig_Inside                     -0.0076      0.001     -8.031      0.000      -0.009      -0.006
FireplaceQu                           0.0050      0.000     15.095      0.000       0.004       0.006
Zone_C                               -0.0961      0.010     -9.961      0.000      -0.115      -0.077
Foundation_PConc                      0.0023      0.002      1.471      0.142      -0.001       0.005
OverallQual:SaleCondition_Abnorml     0.0127      0.001     17.802      0.000       0.011       0.014
NewArea:SaleCondition_Abnorml        -0.0164      0.001    -15.617      0.000      -0.019      -0.014
OverallQual:ExterQual                 0.0056      0.000     35.712      0.000       0.005       0.006
NewArea:ExterQual                    -0.0054      0.000    -13.383      0.000      -0.006      -0.005
==============================================================================
Omnibus:                     2122.086   Durbin-Watson:                   1.924
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               77.429
Skew:                           0.022   Prob(JB):                     1.54e-17
Kurtosis:                       1.057   Cond. No.                     2.91e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.91e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
```

<img src="out/fig8.png" alt="Example Image" width="700"/>


## Model Evaluation

Evaluated model performance using R-squared and adjusted R-squared metrics.
Analyzed residuals to ensure variance stability and identify any remaining patterns or outliers.

### Predictions

Applied the trained model to test data, using consistent data processing and transformations.
Transformed predictions back to the original scale using inverse Box-Cox transformation for final submission.


## Key Findings

The WLS regression model significantly improved prediction accuracy.
Interaction terms and transformed variables played a crucial role in capturing complex relationships within the data.
The model demonstrated robustness in handling heteroscedasticity, leading to reliable and stable predictions.


## Conclusion

This project successfully developed a highly accurate predictive model for real estate prices using advanced regression techniques. The approach of combining feature engineering, data normalization, and WLS regression proved effective in minimizing prediction errors, offering valuable insights for real estate market analysis.
