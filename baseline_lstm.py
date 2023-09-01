# -*- coding: utf-8 -*-
"""
This script is designed to model a machine learning predictor using LSTM.
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
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the preprocessed data from the drive
scaled_df = pd.read_csv(log_dir + '/df_scaled.csv')
print(scaled_df.columns)

# LSTM (Long Short-Term Memory) modeling

# Normalize the CPU usage data for LSTM training
data = scaled_df['CPU usage [MHZ]'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data)

# Split dataset into training and testing sets
train_size = int(len(dataset) * 0.7)
train, test = dataset[:train_size], dataset[train_size:]

# Convert dataset into LSTM acceptable format
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

# Reshape input for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define and compile the LSTM model
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=20, batch_size=1, verbose=2)

# Predict using the LSTM model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Convert predictions back to original scale
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Compute performance metrics
train_mae = mean_absolute_error(Y_train[0], train_predict[:, 0])
test_mae = mean_absolute_error(Y_test[0], test_predict[:, 0])
train_mse = mean_squared_error(Y_train[0], train_predict[:, 0])
test_mse = mean_squared_error(Y_test[0], test_predict[:, 0])
train_rmse = math.sqrt(train_mse)
test_rmse = math.sqrt(test_mse)
train_r2 = r2_score(Y_train[0], train_predict[:, 0])
test_r2 = r2_score(Y_test[0], test_predict[:, 0])

# Print metrics
print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Train MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Train R2: {train_r2:.2f}")
print(f"Test R2: {test_r2:.2f}")

# Plot actual vs predicted CPU usage
plt.figure(figsize=(10, 6))
plt.plot(Y_test[0], 'b', label='Actual')
plt.plot(test_predict, 'g', alpha=0.7, label='Predicted')
plt.title('LSTM: Actual vs Predicted CPU usage [MHZ]')
plt.ylabel('CPU usage [MHZ]')
plt.xlabel('Steps')
plt.legend()
plt.savefig(log_dir + '/lstm_act_pred.png')
plt.show()
