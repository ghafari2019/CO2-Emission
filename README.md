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


