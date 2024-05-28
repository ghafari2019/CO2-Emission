# CO2-Emission


```python
import numpy as np
import pandas as pd
import xlrd 
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import style

# Load the dataset
path = r"C:\Users\ghafari\Desktop\project_x_new\data_pannels\data_pannel_Urg.csv"
dfft = pd.read_csv(path)

# Extract individual panels
location = dfft[dfft['year'] == 'year'].index.values.astype(int)
each_panel = [dfft[location[i] + 1:location[i + 1]] for i in range(len(location) - 1) if i < len(location)]
list_spx_co_new = each_panel

list_column = ['spx_co2', 'msci_co2', 'urg_co2', 'ref_co2']
coli = list_column[2]
year = 2017  

def train_test_split(data, coli, year):
    data.index = data['year'].astype(int)
    x = data.drop(columns=['year', 'ISIN', coli])
    x_train, x_test = x.loc[x.index < year], x.loc[x.index >= year]
    y = data[[coli]]
    y_train, y_test = y.loc[y.index < year], y.loc[y.index >= year]
    return x_train, x_test, y_train, y_test 

# Initialize lists to hold training and testing data
X_train = []
X_test = []
Y_train = []
Y_test = []

# Split data into training and testing sets
for i in range(1, len(list_spx_co_new)):
    dfa = list_spx_co_new[i]  
    x_train, x_test, y_train, y_test = train_test_split(dfa, coli, 2020)
    X_train.append(x_train)
    X_test.append(x_test)
    Y_train.append(y_train)
    Y_test.append(y_test)
    
# Concatenate training and testing data
X_train = pd.concat(X_train)
Y_train = pd.concat(Y_train)
X_test = pd.concat(X_test)
Y_test = pd.concat(Y_test)

# Train a RandomForestRegressor model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, Y_train)

# Calculate feature importances
featureImp = []
for feat, importance in zip(X_train.columns, rf.feature_importances_):  
    featureImp.append([feat, importance * 100])

fT_df = pd.DataFrame(featureImp, columns=['Feature', 'Importance'])
df_feature_import = fT_df.sort_values('Importance', ascending=False)

# Print feature importances
print(df_feature_import)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(df_feature_import['Feature'], df_feature_import['Importance'], color='skyblue')
plt.xlabel('Importance (%)')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()
```
### Implications

- **Model Diversity**: The diverse range of models used in the ensemble, from linear models to tree-based models and instance-based models like KNN, contribute different perspectives on the data, improving overall performance.
- **Ensemble Benefits**: The results clearly indicate the benefit of using ensemble techniques. The Super Learner model's superior performance demonstrates that it effectively integrates information from all base models to make more accurate predictions.
- **Practical Use**: For practical applications, using a stacked ensemble model like the Super Learner would be advantageous due to its higher accuracy and robustness compared to individual models.

Overall, the stacked ensemble approach proves to be a powerful method for improving predictive performance in machine learning tasks.


```python

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy import sqrt
from matplotlib import pyplot as plt

# Generate synthetic regression data
X, y = make_regression(n_samples=1000, n_features=100, noise=0.5)
X, X_val, y, y_val = train_test_split(X, y, test_size=0.50, random_state=42)
print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)

# Define a list of base models
def get_models():
    models = list()
    models.append(LinearRegression())
    models.append(ElasticNet())
    models.append(SVR(gamma='scale'))
    models.append(DecisionTreeRegressor())
    models.append(KNeighborsRegressor())
    models.append(AdaBoostRegressor())
    models.append(BaggingRegressor(n_estimators=10))
    models.append(RandomForestRegressor(n_estimators=10))
    models.append(ExtraTreesRegressor(n_estimators=10))
    return models

# Generate out-of-fold predictions
def get_out_of_fold_predictions(X, y, models):
    meta_X, meta_y = list(), list()
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(X):
        fold_yhats = list()
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        meta_y.extend(test_y)
        for model in models:
            model.fit(train_X, train_y)
            yhat = model.predict(test_X)
            fold_yhats.append(yhat.reshape(len(yhat), 1))
        meta_X.append(np.hstack(fold_yhats))
    return np.vstack(meta_X), np.asarray(meta_y)

# Fit base models
def fit_base_models(X, y, models):
    for model in models:
        model.fit(X, y)

# Fit meta model
def fit_meta_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Evaluate models
def evaluate_models(X, y, models):
    for model in models:
        yhat = model.predict(X)
        mse = mean_squared_error(y, yhat)
        print('%s: RMSE %.3f' % (model.__class__.__name__, sqrt(mse)))

# Make predictions with stacked model
def super_learner_predictions(X, models, meta_model):
    meta_X = list()
    for model in models:
        yhat = model.predict(X)
        meta_X.append(yhat.reshape(len(yhat), 1))
    meta_X = np.hstack(meta_X)
    return meta_model.predict(meta_X)

# Get models
models = get_models()

# Generate out-of-fold predictions
meta_X, meta_y = get_out_of_fold_predictions(X, y, models)
print('Meta ', meta_X.shape, meta_y.shape)

# Fit base models
fit_base_models(X, y, models)

# Fit the meta model
meta_model = fit_meta_model(meta_X, meta_y)

# Evaluate base models
evaluate_models(X_val, y_val, models)

# Evaluate meta model
yhat = super_learner_predictions(X_val, models, meta_model)
print('Super Learner: RMSE %.3f' % (sqrt(mean_squared_error(y_val, yhat))))



```


