from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree, export_text
from sklearn.metrics import mean_squared_error
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import opendatasets as od
import os
from zipfile import ZipFile

import numpy as np  # linear algebra
import pandas as pd  # data processing
import seaborn as sns
sns.set(style="ticks", color_codes=True)

# ----------------------------------Data loading----------------------------------#
"""
dataset_url = 'https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting'
od.download('https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting')
os.listdir('walmart-recruiting-store-sales-forecasting')
"""
train_zip_file = ZipFile(
    'walmart-recruiting-store-sales-forecasting/train.csv.zip')
features_zip_file = ZipFile(
    'walmart-recruiting-store-sales-forecasting/features.csv.zip')
test_zip_file = ZipFile(
    'walmart-recruiting-store-sales-forecasting/test.csv.zip')
sample_submission_zip_file = ZipFile(
    'walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')


train_df = pd.read_csv(train_zip_file.open('train.csv'))
features_df = pd.read_csv(features_zip_file.open('features.csv'))
stores_df = pd.read_csv(
    'walmart-recruiting-store-sales-forecasting/stores.csv')
test_df = pd.read_csv(test_zip_file.open('test.csv'))
submission_df = pd.read_csv(
    sample_submission_zip_file.open('sampleSubmission.csv'))


dataset = train_df.merge(stores_df, how='left').merge(features_df, how='left')
test_dataset = test_df.merge(
    stores_df, how='left').merge(features_df, how='left')


# print(dataset.describe(include='all'))

# Checking the NaN percentage
# print(dataset.isnull().mean() * 100)

"""
corr = dataset.corr()
f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={'shrink': .5})
plt.show()
"""

# ----------------------------------Data manipulation----------------------------------#

dataset[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] = dataset[[
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].fillna(0)
dataset['Year'] = pd.to_datetime(dataset['Date']).dt.year
dataset['Month'] = pd.to_datetime(dataset['Date']).dt.month
dataset['Week'] = pd.to_datetime(dataset['Date']).dt.isocalendar().week
dataset = dataset.drop(
    columns=["Date", "CPI", "Fuel_Price", 'Unemployment', 'Temperature'])

# Move target column to the end
df = dataset.pop('Weekly_Sales')
dataset['Weekly_Sales'] = df

# Identify input and target columns
input_cols, target_col = dataset.columns[:-1], dataset.columns[-1]
inputs_df, targets = dataset[input_cols].copy(), dataset[target_col].copy()

# Identify numerical and categorical columns
numeric_cols = dataset[input_cols].select_dtypes(
    include=np.number).columns.tolist()
categorical_cols = dataset[input_cols].select_dtypes(
    include='object').columns.tolist()

# Impute and scale numerical columns
imputer = SimpleImputer().fit(inputs_df[numeric_cols])
inputs_df[numeric_cols] = imputer.transform(inputs_df[numeric_cols])
scaler = MinMaxScaler().fit(inputs_df[numeric_cols])
inputs_df[numeric_cols] = scaler.transform(inputs_df[numeric_cols])


# One-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(
    inputs_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
inputs_df[encoded_cols] = encoder.transform(inputs_df[categorical_cols])
# print(inputs_df.head())

# Split the data into train and validation sets
train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    inputs_df[numeric_cols + encoded_cols], targets, test_size=0.25, random_state=42)

# print(train_inputs.shape, val_inputs.shape, train_targets.shape, val_targets.shape)

"""
# ----------------------------------DECISION TREE REGRESSOR----------------------------------#
# ----------------------------------Model training----------------------------------#
tree = DecisionTreeRegressor(random_state=0)


start_time = time.time()

tree.fit(train_inputs, train_targets)

end_time = time.time()
elapsed_time = end_time - start_time
# print(f"Execution time: {elapsed_time} seconds")

# ----------------------------------Model evaluation----------------------------------#

tree_train_preds = tree.predict(train_inputs)

tree_train_rmse = mean_squared_error(
    train_targets, tree_train_preds, squared=False)

tree_val_preds = tree.predict(val_inputs)

tree_val_rmse = mean_squared_error(val_targets, tree_val_preds, squared=False)

# print('Train RMSE: {}, Validation RMSE: {}'.format(tree_train_rmse, tree_val_rmse))

# ----------------------------------Decsion tree visualisation----------------------------------#

sns.set_style('darkgrid')

plt.figure(figsize=(30, 15))
# plot_tree(tree, feature_names=train_inputs.columns, max_depth=3, filled=True)
# plt.show()

# tree structure
tree_text = export_text(tree, feature_names=list(train_inputs.columns))
# print(tree_text[:2000])


# Important features

tree_importances = tree.feature_importances_

tree_importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': tree_importances
}).sort_values('importance', ascending=False)

# print(tree_importance_df)

plt.title('Decision Tree Feature Importance')
sns.barplot(data=tree_importance_df.head(10), x='importance', y='feature')
# plt.show()
"""

# ---------------------------------- RANDOM FOREST REGRESSOR ----------------------------------#
# ----------------------------------Model training----------------------------------#
"""rf1 = RandomForestRegressor(random_state=0, n_estimators=10)

start_time = time.time()

rf1.fit(train_inputs, train_targets)

end_time = time.time()
elapsed_time = end_time - start_time"""
# print(f"Execution time: {elapsed_time} seconds")

# ----------------------------------Model evaluation----------------------------------#

"""rf1_train_preds = rf1.predict(train_inputs)

rf1_train_rmse = mean_squared_error(
    train_targets, rf1_train_preds, squared=False)

rf1_val_preds = rf1.predict(val_inputs)

rf1_val_rmse = mean_squared_error(val_targets, rf1_val_preds, squared=False)

print('Train RMSE: {}, Validation RMSE: {}'.format(rf1_train_rmse, rf1_val_rmse))"""

# Predict a given instance
# print(val_targets)
# print(rf1.predict(val_inputs.iloc[0:5]))
# print(val_targets.iloc[0:5])

# ----------------------------------Hyper parameter tuning----------------------------------#


def test_params(**params):
    model = RandomForestRegressor(
        random_state=0, n_jobs=-1, n_estimators=16, **params).fit(train_inputs, train_targets)
    train_rmse = mean_squared_error(model.predict(
        train_inputs), train_targets, squared=False)
    val_rmse = mean_squared_error(model.predict(
        val_inputs), val_targets, squared=False)
    return train_rmse, val_rmse


def test_param_and_plot(param_name, param_values):
    train_errors, val_errors = [], []
    for value in param_values:
        params = {param_name: value}
        train_rmse, val_rmse = test_params(**params)
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)
    plt.figure(figsize=(10, 6))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(['Training', 'Validation'])
    plt.show()


# test_param_and_plot('min_samples_leaf', [1, 2, 3, 4, 5])
# print(test_params(min_samples_leaf=5))

# test_param_and_plot('max_leaf_nodes', [20, 25, 30, 35, 40, 50])
# test_params(max_leaf_nodes=20)

# test_param_and_plot('max_depth', [5, 10, 15, 20, 25])
# test_params(max_depth=10)

# ----------------------------------Training best model----------------------------------#
rf2 = RandomForestRegressor(
    n_estimators=16, random_state=0, min_samples_leaf=1)

rf2.fit(train_inputs, train_targets)

rf2_train_preds = rf2.predict(train_inputs)

rf2_train_rmse = mean_squared_error(
    train_targets, rf2_train_preds, squared=False)

rf2_val_preds = rf2.predict(val_inputs)

rf2_val_rmse = mean_squared_error(val_targets, rf2_val_preds, squared=False)

print('Train RMSE: {}, Validation RMSE: {}'.format(rf2_train_rmse, rf2_val_rmse))


# ----------------------------------Feature Importance----------------------------------#

"""rf2_importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': rf2.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(data=rf2_importance_df, x='importance', y='feature')
plt.show()"""


# ----------------------------------Make the test set----------------------------------#
"""test_dataset[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']] = test_dataset[[
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].fillna(0)
test_dataset['Year'] = pd.to_datetime(test_dataset['Date']).dt.year
test_dataset['Month'] = pd.to_datetime(test_dataset['Date']).dt.month
test_dataset['Week'] = pd.to_datetime(
    test_dataset['Date']).dt.isocalendar().week
test_dataset = test_dataset.drop(
    columns=["Date", "CPI", "Fuel_Price", 'Unemployment', 'Temperature'])

test_dataset[numeric_cols] = imputer.transform(test_dataset[numeric_cols])
test_dataset[numeric_cols] = scaler.transform(test_dataset[numeric_cols])
test_dataset[encoded_cols] = encoder.transform(test_dataset[categorical_cols])

test_inputs = test_dataset[numeric_cols + encoded_cols]

test_preds = rf2.predict(test_inputs)

# See the RMSE of the test set

print(mean_squared_error(
    submission_df['Weekly_Sales'], test_preds, squared=False))

# See the predictions
print(test_preds)
# See the actual values of weekly sales in the test set
print(submission_df['Weekly_Sales'])"""
