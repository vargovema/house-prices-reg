# %% [markdown]
# # Weighted Least Square regression (Model 1)

# %% [markdown]
# This model produces the lowest scoring predictions based on RMSE.

# %% [markdown]
# Loading the needed libraries:

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy import special
import statsmodels.api as sm

# %%
df = pd.read_csv("data/train.csv")

df.head()

# %%
print(df.info())

# %% [markdown]
# ## Missing values and encodings of categorical variables

# %% [markdown]
# In general, we see that only categorical variables have missing values. This data has missing values only for properties which lack a particular feature. Hence, the missing values are treated as a separate category and endoded as 0 so when a property misses a particular feature, the information is captured in the intercept.

# %%
nan_count = df.isna().sum()
print(nan_count)

# %%
df['Type'] = df['Type'].astype('object')

# %%
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int','float']).columns.tolist()

# %%
df[categorical_cols] = df[categorical_cols].fillna(value='A1')

# %% [markdown]
# Based on the unique values in the categorical columns, mapping for these is created in order to encode them to numerical levels.

# %%
# Print the unique values in categorical columns
for col in categorical_cols:
    df[col] = df[col].astype('category')
    unique_values = df[col].unique()
    print("Unique values in", col, ":", unique_values)

# %%
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

# %% [markdown]
# Creating dummies for certain features of the properties which might be useful for further interaction effetcs.

# %%
df['HasBsmt'] = df['BsmtSF'].apply(lambda con: False if con == 0 else True).astype('int')
df['HasGarage'] = df['GarageCars'].apply(lambda con: False if con == 0 else True).astype('int')
df['HasFireplace'] = df['Fireplaces'].apply(lambda con: False if con == 0 else True).astype('int')
df['HasPool'] = df['PoolArea'].apply(lambda con: False if con == 0 else True).astype('int')

# %% [markdown]
# ## Visual inspection of patterns in the data

# %%
sns.pairplot(df.drop('Id', axis=1), y_vars=df.columns.drop(['SalePrice','Id']), x_vars=['SalePrice'])
plt.savefig('out/fig1.png', dpi=300, bbox_inches='tight')
plt.show

# %% [markdown]
# Due to the high correlations among the possible regressors, some numerical variables, that are assumed to be important, are going to be combined in order to reduce dimensionality of the data.

# %%
df['YearBuiltandRemod'] = df['YearBuilt'] + df['YearRemodAdd']
df['NewArea'] = df['BsmtSF'] + df['X1stFlrSF'] + df['X2ndFlrSF']

# %% [markdown]
# From the plot above, we see that there are some non-linear patterns in the data. For this reason, box-cox transformation is going to be used in order to normalise the data and make the patterns more linear. Box-cox is also applied to the response variable and for every variable, an optimal lambda is used such that the distributions approach normal distribution.

# %%
df['SalePrice'], _ = stats.boxcox(df['SalePrice'])
response_lambda = _

# %%
num_cols_tranform = ['LotArea','BsmtSF','X1stFlrSF', 'X2ndFlrSF', 'LivAreaSF', 'GarageArea', 'NewArea']

lambdas = []
for i in num_cols_tranform:
    df[i], _ = stats.boxcox(df[i]+1)
    lambdas.append(_)

# %%
numerical_cols_plot = numerical_cols[0:22] + ['YearBuiltandRemod','NewArea']

# %%
g=sns.pairplot(df[numerical_cols_plot+['SalePrice','SaleCondition_Abnorml']], y_vars=numerical_cols_plot, x_vars=['SalePrice'],hue='SaleCondition_Abnorml')
# Add regression line to each scatter plot
for ax in g.axes.flat:
    if ax.get_ylabel() in numerical_cols_plot:
        sns.regplot(x=ax.get_xlabel(), y=ax.get_ylabel(), data=df[numerical_cols_plot+['SalePrice','SaleCondition_Abnorml']], ax=ax, scatter=False, color='red', ci=95)

# Display the plot
plt.savefig('out/fig2.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
g=sns.pairplot(df[numerical_cols_plot+['SalePrice','ExterQual']], y_vars=numerical_cols_plot, x_vars=['SalePrice'],hue='ExterQual')
# Add regression line to each scatter plot
for ax in g.axes.flat:
    if ax.get_ylabel() in numerical_cols_plot:
        sns.regplot(x=ax.get_xlabel(), y=ax.get_ylabel(), data=df[numerical_cols_plot+['SalePrice','ExterQual']], ax=ax, scatter=False, color='red', ci=95)

# Display the plot
plt.savefig('out/fig3.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# From the plots above, we see that the abnormal sale condition catches some outliers when it comes to the effects of NewArea and OverallQual. Moreover, we that ExterQual can be used for similar interactions.

# %%
categorical_cols = df.select_dtypes(include=['category']).columns.tolist()

# %%
# Create dummy variables for categorical variable
df_model_dummies = pd.get_dummies(df, columns=categorical_cols)

# Compute the correlation matrix
corr_dummy = df_model_dummies.corr().round(decimals=2)

# %%
mask = np.zeros_like(corr_dummy)
mask[np.triu_indices_from(mask)] = True

# Plot the correlation matrix as a heatmap
plt.figure(figsize = (40,30))
# Set up the annotation font size
annot_font = {'fontsize': 8}
ax = sns.heatmap(corr_dummy, annot=True, vmin=-1, vmax=1, center=0, cbar=True, mask=mask, square=False,  annot_kws=annot_font,
            cmap=sns.diverging_palette(20, 220, n=200))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, horizontalalignment='right')
plt.savefig('out/fig4.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
correlation_threshold = 0.8
highly_correlated_pairs = []

# Iterate over the correlation matrix
for i in range(len(corr_dummy.columns)):
    for j in range(i + 1, len(corr_dummy.columns)):
        if abs(corr_dummy.iloc[i, j]) >= correlation_threshold:
            pair = (corr_dummy.columns[i], corr_dummy.columns[j])
            highly_correlated_pairs.append(pair)
print(highly_correlated_pairs)

# %% [markdown]
# From the correlation plot, the following numerical variables are chosen for the regression: 'OverallQual', 'NewArea', 'YearBuiltandRemod' 'GarageCars', 'BsmtQual', 'GarageFinish', 'KitchenQual', 'LotArea', 'OverallCond'.  'OverallCond' is chosen mainly due to the fact that it has small correlation with other variables which means it is supposed capture more new information. 'LotArea' is chosen because it captures additional information about area of the whole property that is not captured in 'NewArea'.

# %% [markdown]
# In order to avoid vast diffferences in the scales of the data, some variables need to be rescaled.

# %%
scaler = StandardScaler()
df_model_dummies[['YearBuilt','YearRemodAdd','GarageArea','YearBuiltandRemod']] = scaler.fit_transform(df_model_dummies[['YearBuilt','YearRemodAdd','GarageArea','YearBuiltandRemod']])

# %% [markdown]
# In order to choose dummy variables that could be used in the model, a regression with only the numerical variables chosen above is going to be fitted and the residuals are going to be analyzed for further correlation.

# %%
y = df_model_dummies['SalePrice']
X = df_model_dummies.drop(['Id', 'SalePrice'], axis=1)

X[X.select_dtypes(include=['bool','category']).columns.tolist()] = X[X.select_dtypes(
    include=['bool','category']).columns.tolist()].astype(int) 

# %%
X = X[['NewArea','YearBuiltandRemod','GarageCars','BsmtQual','GarageFinish','KitchenQual','LotArea','OverallCond']]

X_train = X
y_train = y


# %%
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

# Print the p-values
print(model.summary())

# %%
# Get the predicted values
y_pred_train = model.fittedvalues

# Calculate the residuals
residuals = model.resid

# Assuming 'y_pred' contains the predicted values and 'residuals' contains the residuals
plt.scatter(y_pred_train, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('out/fig5.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
reg_analysis = pd.DataFrame(df_model_dummies)
reg_analysis.loc[:, 'Residuals'] = residuals

# %%
# Compute the correlation matrix
corr_reg_analysis = reg_analysis.corr().round(decimals=2)

# %%
mask = np.zeros_like(corr_reg_analysis)
mask[np.triu_indices_from(mask)] = True

# Plot the correlation matrix as a heatmap
plt.figure(figsize = (40,30))
# Set up the annotation font size
annot_font = {'fontsize': 8}
ax = sns.heatmap(corr_reg_analysis, annot=True, vmin=-1, vmax=1, center=0, cbar=True, mask=mask, square=False,  annot_kws=annot_font,
            cmap=sns.diverging_palette(20, 220, n=200))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, horizontalalignment='right')
plt.savefig('out/fig6.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# Based on the correlations above, the following dummies are going to be included in the model: 'BldgType_Twnhs', 'Zone_RL', 'Zone_FV', 'LotConfig_Inside', 'FireplaceQu', 'Zone_C', 'Foundation_PConc'.

# %% [markdown]
# In order to fit the model, we need to create new variables for the interaction effects observed from the plots. Moreover, the two most extereme observations of SalePrice and NewArea at each tail are going to excluded from the model. This is due to the reason that we could observe in the plots that they inroduce large noise. 

# %%
y = df_model_dummies['SalePrice']
X = df_model_dummies.drop(['Id', 'SalePrice'], axis=1)

X[X.select_dtypes(include=['bool','category']).columns.tolist()] = X[X.select_dtypes(
    include=['bool','category']).columns.tolist()].astype(int) 

# %%
X['OverallQual:SaleCondition_Abnorml'] = X['OverallQual'] * X['SaleCondition_Abnorml'] 
X['NewArea:SaleCondition_Abnorml'] = X['NewArea'] * X['SaleCondition_Abnorml'] 
X['OverallQual:ExterQual'] = X['OverallQual'] * X['ExterQual'] 
X['NewArea:ExterQual'] = X['NewArea'] * X['ExterQual'] 

X = X[['NewArea','YearBuiltandRemod','GarageCars','BsmtQual','GarageFinish','KitchenQual','OverallCond','LotArea',
       'BldgType_Twnhs','Zone_RL','Zone_FV','LotConfig_Inside','FireplaceQu','Zone_C','Foundation_PConc',
       'OverallQual:SaleCondition_Abnorml','NewArea:SaleCondition_Abnorml','OverallQual:ExterQual','NewArea:ExterQual']]

X_train = X
y_train = y

X_train = X_train[(y_train>6.71) & (y_train<7.42)]
y_train = y_train[(y_train>6.71) & (y_train<7.42)]

y_train = y_train[(X_train['NewArea']>4.2) & (X_train['NewArea']<4.8)]
X_train = X_train[(X_train['NewArea']>4.2) & (X_train['NewArea']<4.8)]


# %%
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

# Print the p-values
print(model.summary())

# %%
# Get the predicted values
y_pred_train = model.fittedvalues

# Calculate the residuals
residuals = model.resid

# Assuming 'y_pred' contains the predicted values and 'residuals' contains the residuals
plt.scatter(y_pred_train, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('out/fig7.png', dpi=300, bbox_inches='tight')
plt.show()


# %% [markdown]
# From the residuals, we see that the variance is more or less similar across different predicted values, except for some outliers in the predictions. In order to control for those, WLS regression is going to be used in order to put lower weights on these kind of observations.

# %%
estimated_variances = np.square(residuals)
weights = 1 / (estimated_variances)

# Fit the WLS regression model with heteroscedasticity adjustment
model = sm.WLS(y_train, sm.add_constant(X_train), weights=weights).fit()

# Print the summary of the regression results
print(model.summary())

# %%
# Get the predicted values
#y_pred_train = lasso.predict(X_train)
y_pred_train = model.fittedvalues

# Calculate the residuals
residuals = model.resid

# Assuming 'y_pred' contains the predicted values and 'residuals' contains the residuals
plt.scatter(y_pred_train, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('out/fig8.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Creating predicitons for the test data

# %% [markdown]
# The same data processing techniques need to be applied to the test data.

# %%
# Predictions
df_test = pd.read_csv("data/test.csv")

# %%
categorical_cols = df_test.select_dtypes(include=['object']).columns.tolist()
df_test[categorical_cols] = df_test[categorical_cols].fillna(value='A1')

for col in categorical_cols:
    df_test[col] = df_test[col].astype('category')

# %%
df_test['Type'] = df_test['Type'].astype('category')

mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'A1':0}

df_test['ExterQual'] = df_test['ExterQual'].map(mapping).replace(mapping).astype('int') # Using map() function
df_test['ExterCond'] = df_test['ExterCond'].map(mapping).replace(mapping).astype('int') # Using map() function
df_test['BsmtQual'] = df_test['BsmtQual'].map(mapping).replace(mapping).astype('int') # Using map() function
df_test['BsmtCond'] = df_test['BsmtCond'].map(mapping).replace(mapping).astype('int') # Using map() function
df_test['FireplaceQu'] = df_test['FireplaceQu'].map(mapping).replace(mapping).astype('int') # Using map() function
df_test['GarageQual'] = df_test['GarageQual'].map(mapping).replace(mapping).astype('int') # Using map() function
df_test['GarageCond'] = df_test['GarageCond'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Ex': 3, 'Gd': 2, 'TA': 1, 'Fa': 0}

df_test['KitchenQual'] = df_test['KitchenQual'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'A1': 0}

df_test['BsmtExposure'] = df_test['BsmtExposure'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'A1': 0}

df_test['BsmtRating'] = df_test['BsmtRating'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Typ': 3, 'Min': 2, 'Mod': 1, 'Maj': 0}

df_test['Functional'] = df_test['Functional'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Fin': 1, 'RFn': 1, 'Unf': 0, 'A1': 0}

df_test['GarageFinish'] = df_test['GarageFinish'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Y': 1, 'N': 0}

df_test['CentralAir'] = df_test['CentralAir'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Attchd': 1, 'Other': 1, 'Detchd': 0, 'A1': 0}

df_test['GarageType'] = df_test['GarageType'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Abnorml': 1, 'Family': 0, 'Normal': 0}

df_test['SaleCondition_Abnorml'] = df_test['SaleCondition'].map(mapping).replace(mapping).astype('int') # Using map() function

mapping = {'Abnorml': 0, 'Family': 1, 'Normal': 0}

df_test['SaleCondition_Family'] = df_test['SaleCondition'].map(mapping).replace(mapping).astype('int') # Using map() function

df_test.drop('SaleCondition', axis=1, inplace=True)

df_test['HasBsmt'] = df_test['BsmtSF'].apply(lambda con: False if con == 0 else True).astype('int')
df_test['HasGarage'] = df_test['GarageCars'].apply(lambda con: False if con == 0 else True).astype('int')
df_test['HasFireplace'] = df_test['Fireplaces'].apply(lambda con: False if con == 0 else True).astype('int')
df_test['HasPool'] = df_test['PoolArea'].apply(lambda con: False if con == 0 else True).astype('int')

df_test['NewArea'] = df_test['BsmtSF'] + df_test['X1stFlrSF'] + df_test['X2ndFlrSF']
df_test['YearBuiltandRemod'] = df_test['YearBuilt'] + df_test['YearRemodAdd']

lambdas_test = []
j=0
for i in num_cols_tranform:
    df_test[i] = stats.boxcox(df_test[i]+1, lambdas[j])
    j = j+ 1

df_test[['YearBuilt','YearRemodAdd','GarageArea','YearBuiltandRemod']] = scaler.fit_transform(
    df_test[['YearBuilt','YearRemodAdd','GarageArea','YearBuiltandRemod']])

categorical_cols = df_test.select_dtypes(include=['category']).columns.tolist()

df_test_dummies = pd.get_dummies(df_test, columns=categorical_cols)

# %%
X_test = df_test_dummies.drop(['Id'], axis=1)

X_test[X_test.select_dtypes(include=['bool','category']).columns.tolist()] = X_test[X_test.select_dtypes(
    include=['bool','category']).columns.tolist()].astype(int) 

# %%
X_test['OverallQual:SaleCondition_Abnorml'] = X_test['OverallQual'] * X_test['SaleCondition_Abnorml'] 
X_test['NewArea:SaleCondition_Abnorml'] = X_test['NewArea'] * X_test['SaleCondition_Abnorml'] 
X_test['OverallQual:ExterQual'] = X_test['OverallQual'] * X_test['ExterQual'] 
X_test['NewArea:ExterQual'] = X_test['NewArea'] * X_test['ExterQual'] 

X_test = X_test[['NewArea','YearBuiltandRemod','GarageCars','BsmtQual','GarageFinish','KitchenQual','OverallCond','LotArea',
       'BldgType_Twnhs','Zone_RL','Zone_FV','LotConfig_Inside','FireplaceQu','Zone_C','Foundation_PConc',
       'OverallQual:SaleCondition_Abnorml','NewArea:SaleCondition_Abnorml','OverallQual:ExterQual','NewArea:ExterQual']]

# %%
y_pred_test = model.predict(sm.add_constant(X_test))

data = {'Id': df_test_dummies['Id'], 'Predicted': y_pred_test}

df_submission = pd.DataFrame(data)

# %% [markdown]
# The predicted values need to be trasformed back to the original scale using inverse Box-Cox tranformation with the same lambda that was used for the response variable in the train data.

# %%
df_submission['Predicted'] = special.inv_boxcox(df_submission['Predicted'].astype('float'), response_lambda)

df_submission.head()

# %% [markdown]
# 


