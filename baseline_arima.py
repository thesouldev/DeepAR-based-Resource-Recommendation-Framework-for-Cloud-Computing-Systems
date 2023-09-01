# -*- coding: utf-8 -*-
"""
This script is designed to model a machine learning predictor using ARIMA. 
It was originally created in Google Colab, and this version has been converted to a standalone Python script.
To run this in Google Colab:
1. Upload this script to Google Colab.
2. Ensure the necessary data file (`df_scaled.csv`) is present in the specified Google Drive path.
3. Execute the script.

Original file link:
    https://colab.research.google.com/drive/1hk9HT3Hr4WrRquJ8GD8iwO04u2H6kv09
"""

# Import Google Drive library for Colab
from google.colab import drive

# Mount Google Drive to access files
drive.mount('/content/gdrive')

# Define the directory where the dataset and output files are/will be stored
log_dir = "/content/gdrive/My Drive/workload_predictor_vm"

# Import necessary libraries and modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import statsmodels.api as sm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the preprocessed data from the drive
scaled_df = pd.read_csv(log_dir + '/df_scaled.csv')
print(scaled_df.columns)

# ARIMA (AutoRegressive Integrated Moving Average) modeling

# Splitting the data into training and testing sets
X = scaled_df['CPU usage [%]']
size = int(len(X) * 0.66)
train, test = X[0:size].reset_index(drop=True), X[size:len(X)].reset_index(drop=True)
history = [x for x in train]
predictions = list()

# Training and predicting with ARIMA model
for t in range(len(test)):
    model = sm.tsa.arima.ARIMA(history, order=(10,0,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# Evaluating the ARIMA model
mse = mean_squared_error(test, predictions)
mae = mean_absolute_error(test, predictions)
rmse = sqrt(mse)
r2 = r2_score(test, predictions)
print('Test MSE: %.3f' % mse)
print('Test MAE: %.3f' % mae)
print('Test RMSE: %.3f' % rmse)
print('Test R2 score: %.3f' % r2)

# Plotting the actual vs predicted CPU usage
sns.set_style("whitegrid")
test_list = list(test)
predictions_list = list(predictions)
range_values = range(len(test_list))
plt.figure(figsize=(12, 6))
plt.plot(range_values, test_list, label='Actual', color='b')
plt.plot(range_values, predictions_list, label='Predicted', color='r')
plt.legend(loc='upper left')
plt.title('ARIMA: Actual vs Predicted CPU usage [MHZ]')
plt.ylabel('CPU usage [MHZ]')
plt.xlabel('Index')
plt.grid(True)
plt.savefig(log_dir + '/arima_act_pred.png')
plt.show()