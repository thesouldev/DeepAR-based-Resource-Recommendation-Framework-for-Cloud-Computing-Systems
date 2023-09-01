# -*- coding: utf-8 -*-
"""
Description:
This script is designed to predict CPU workloads using the DeepAR algorithm on AWS SageMaker.
It fetches data, processes it, and trains a DeepAR model to forecast CPU usage.

Instructions for AWS SageMaker setup:
1. Create a SageMaker notebook instance.
2. Once the notebook instance is active, open Jupyter or JupyterLab.
3. Upload this script.
4. Ensure the necessary IAM roles are set up with permissions for SageMaker and S3.
5. You might need to install required libraries/packages if not already available.
6. Run the cells of the notebook/script sequentially.

"""

# Initial setup: Importing necessary libraries and packages
import warnings
warnings.filterwarnings('ignore')

# Set up the S3 bucket and prefix for SageMaker
bucket = 'workload-prediction-v1'
prefix = 'sagemaker/test'

# Set up IAM role and SageMaker session
import sagemaker
import sagemaker.predictor
import boto3
import s3fs
import re
from sagemaker import get_execution_role
import json
import math
from os import path
import sagemaker.amazon.common as smac

role = get_execution_role()
sagemaker_session = sagemaker.Session()
region = boto3.Session().region_name
smclient = boto3.Session().client('sagemaker')

# Define data paths on S3
s3_data_path = "{}/{}/data".format(bucket, prefix)
s3_output_path = "{}/{}/output".format(bucket, prefix)

# Setting up the SageMaker DeepAR container
containers = {
    'us-west-2': '156387875391.dkr.ecr.us-west-2.amazonaws.com/forecasting-deepar:latest',
}
region = boto3.Session().region_name
image_name = containers.get(region)
if image_name is None:
    raise ValueError(f"Container image not found for the region: {region}")

# Importing necessary libraries for data processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import time
import json
import glob

# Data fetching: Uncomment these lines if you need to download the dataset for the first time
# !wget http://gwa.ewi.tudelft.nl/fileadmin/pds/trace-archives/grid-workloads-archive/datasets/gwa-t-12/rnd.zip
# import zipfile
# with zipfile.ZipFile("rnd.zip","r") as zip_ref:
#    zip_ref.extractall("targetdir")

# Data processing: Loading and concatenating datasets
files = glob.glob(os.path.join('targetdir/rnd/2013-7', "*.csv"))

files_first200 = files[:150]
dfs = [pd.read_csv(fp, sep = ';\t').assign(VM=os.path.basename(fp).split('.')[0]) for fp in files_first200]
df = pd.concat(dfs, ignore_index=True)

files2 = glob.glob(os.path.join('targetdir/rnd/2013-8', "*.csv"))
files2_first200 = files2[:150]
dfs2 = [pd.read_csv(fp, sep = ';\t').assign(VM=os.path.basename(fp).split('.')[0]) for fp in files2_first200]
df2 = pd.concat(dfs2, ignore_index=True)

files3 = glob.glob(os.path.join('targetdir/rnd/2013-9', "*.csv"))
files3_first200 = files3[:150]
dfs3 = [pd.read_csv(fp, sep = ';\t').assign(VM=os.path.basename(fp).split('.')[0]) for fp in files3_first200]
df3 = pd.concat(dfs3, ignore_index=True)

newdat = df.append(df2)
newerdat = newdat.append(df3)
concatenated_df = newerdat

concatenated_df.head()



# Formatting
concatenated_df['Timestamp'] = pd.to_datetime(concatenated_df['Timestamp [ms]'], unit = 's')
concatenated_df.describe()
concatenated_df['weekday'] = concatenated_df['Timestamp'].dt.dayofweek
concatenated_df['weekend'] = ((concatenated_df.weekday) // 5 == 1).astype(float)

# Feature engineering with the date
concatenated_df['month']=concatenated_df.Timestamp.dt.month
concatenated_df['day']=concatenated_df.Timestamp.dt.day
concatenated_df.set_index('Timestamp',inplace=True)
concatenated_df["CPU usage prev"] = concatenated_df['CPU usage [%]'].shift(1)
concatenated_df["CPU_diff"] = concatenated_df['CPU usage [%]'] - concatenated_df["CPU usage prev"]
concatenated_df["received_prev"] = concatenated_df['Network received throughput [KB/s]'].shift(1)
concatenated_df["received_diff"] = concatenated_df['Network received throughput [KB/s]']- concatenated_df["received_prev"]
concatenated_df["transmitted_prev"] = concatenated_df['Network transmitted throughput [KB/s]'].shift(1)
concatenated_df["transmitted_diff"] = concatenated_df['Network transmitted throughput [KB/s]']- concatenated_df["transmitted_prev"]

concatenated_df["start"] = concatenated_df.index
concatenated_df['target'] = concatenated_df['CPU usage [MHZ]']

df2 = concatenated_df.groupby('VM').resample('1min')['target'].mean().to_frame()
df2.reset_index(level=0, inplace=True)

df2.head()

df2 = df2.fillna(method='ffill')

df2.to_csv('df2.csv', index=False)

df3 = concatenated_df.groupby('VM').resample('1min')['CPU capacity provisioned [MHZ]'].mean().to_frame()
df3.reset_index(level=0, inplace=True)
df3 = df3.fillna(method='ffill')
df3.head()

df3.to_csv('df3.csv', index=False)

# df2 = pd.read_csv('df2.csv')

freq = "1min"
context_length = 30
prediction_length = 30

def series_to_obj(ts, cat=None):
    obj = {"start": str(ts.index[0]), "target": list(ts)}
    if cat is not None:
        obj["cat"] = cat
    return obj

def series_to_jsonline(ts, cat=None):
    return json.dumps(series_to_obj(ts, cat))

time_series_test=[]
vm_index_range = df2['VM'].unique()
for i in vm_index_range:
    newseries = df2[df2['VM'] == i]['target']
    newseries.index.name = None
    newseries.index = pd.to_datetime(newseries.index)
    time_series_test.append(newseries)

time_series_training=[]
vm_index_range = df2['VM'].unique()
for i in vm_index_range:
    newseries = df2[df2['VM'] == i]['target']
    newseries.index.name = None
    newseries.index = pd.to_datetime(newseries.index)
    time_series_training.append(newseries[:-prediction_length])

time_series_test_pro=[]
vm_index_range = df3['VM'].unique()
for i in vm_index_range:
    newseries = df3[df3['VM'] == i]['CPU capacity provisioned [MHZ]']
    newseries.index.name = None
    newseries.index = pd.to_datetime(newseries.index)
    time_series_test_pro.append(newseries)

print(len(time_series_test), len(time_series_training), len(time_series_test_pro))

s3filesystem = s3fs.S3FileSystem()

encoding = "utf-8"

with s3filesystem.open(s3_data_path + "/test/test_data.json", 'wb') as fp:
    for ts in time_series_test:
        fp.write(series_to_jsonline(ts).encode(encoding))
        fp.write('\n'.encode(encoding))

with s3filesystem.open(s3_data_path + "/train/train_data.json", 'wb') as fp:
    for ts in time_series_training:
        fp.write(series_to_jsonline(ts).encode(encoding))
        fp.write('\n'.encode(encoding))

estimator = sagemaker.estimator.Estimator(
    sagemaker_session=sagemaker_session,
    image_uri=image_name,
    role=role,
    instance_count=1,
    instance_type='ml.c4.xlarge',
    base_job_name='test-demo-deepar',
    output_path="s3://" + s3_output_path
)

hyperparameters  = {
    "time_freq": freq,
    "context_length": context_length,
    "prediction_length": prediction_length,
    "num_cells": "64",
    "num_layers": "3",
    "likelihood": "gaussian",
    "epochs": "30",
    "mini_batch_size": "32",
    "learning_rate": "0.001",
    "dropout_rate": "0.05",
    "early_stopping_patience": "10"
}

estimator.set_hyperparameters(**hyperparameters)

data_channels = {
    "train": "s3://{}/train/".format(s3_data_path),
    "test": "s3://{}/test/".format(s3_data_path)
}

estimator.fit(inputs=data_channels)

job_name = estimator.latest_training_job.name
print(job_name)
endpoint_name = sagemaker_session.endpoint_from_job(
    job_name=job_name,
    initial_instance_count=1,
    instance_type='ml.m4.xlarge',
    image_uri=image_name,
    role=role
)

class DeepARPredictor(sagemaker.predictor.RealTimePredictor):

    def set_prediction_parameters(self, freq, prediction_length):
        self.freq = freq
        self.prediction_length = prediction_length

    def predict(self, ts, cat=None, encoding="utf-8", num_samples=100, quantiles=["0.1", "0.5", "0.9"]):
        prediction_times = [x.index[-1] + pd.Timedelta(1, unit=x.index.freq.name) for x in ts]
        req = self.__encode_request(ts, cat, encoding, num_samples, quantiles)
        res = super(DeepARPredictor, self).predict(req, initial_args={"ContentType": "application/json"})
        # res = super(DeepARPredictor, self).predict(req)
        return self.__decode_response(res, prediction_times, encoding)

    def __encode_request(self, ts, cat, encoding, num_samples, quantiles):
        instances = [series_to_obj(ts[k], cat[k] if cat else None) for k in range(len(ts))]
        configuration = {"num_samples": num_samples, "output_types": ["quantiles"], "quantiles": quantiles}
        http_request_data = {"instances": instances, "configuration": configuration}
        return json.dumps(http_request_data).encode(encoding)

    def __decode_response(self, response, prediction_times, encoding):
        response_data = json.loads(response.decode(encoding))
        list_of_df = []
        for k in range(len(prediction_times)):
            prediction_index = pd.date_range(start=prediction_times[k], freq=self.freq, periods=self.prediction_length)
            list_of_df.append(pd.DataFrame(data=response_data['predictions'][k]['quantiles'], index=prediction_index))
        return list_of_df

predictor = DeepARPredictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session,
    content_type="application/json"
)
predictor.set_prediction_parameters(freq, prediction_length)

new_time_series_training = []
for ts in time_series_training:
    new_time_series_training.append(ts.asfreq('T'))

new_time_series_test = []
for ts in time_series_test:
    new_time_series_test.append(ts.asfreq('T'))

new_time_series_test_pro = []
for ts in time_series_test_pro:
    new_time_series_test_pro.append(ts.asfreq('T'))

list_of_df  = predictor.predict(new_time_series_training[1:2]) # predicted forecast
actual_data = new_time_series_test[1:2] # full data set
actual_provisioned = new_time_series_test_pro[1:2]

print(actual_provisioned)


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.figure(figsize=(12,6))

for k in range(len(list_of_df)):
    actual_data[k][-prediction_length-context_length:].plot(label='Actual',linewidth = 2.5)
    p10 = list_of_df[k]['0.1']
    p90 = list_of_df[k]['0.9']
    plt.fill_between(p10.index, p10, p90, color='grey', alpha=0.3, label='80% Confidence Interval')
    list_of_df[k]['0.5'].plot(label='Prediction Median', color = 'blue', linewidth = 2.5)
    (list_of_df[k]['0.9']+100).plot(label='Suggested Provision', color = 'green', linewidth = 2.5)

    plt.title("DeepAR Model Prediction")
    plt.ylabel("CPU usage [MHz]")
    plt.xlabel("Time")
    plt.yticks()
    plt.xticks()
    plt.legend(loc='upper left')

    print(actual_provisioned[k][-prediction_length-context_length:].mean())
    print((list_of_df[k]['0.9']+100).mean())
    print(list_of_df[k]['0.5'].mean())

plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# These lists will hold all actual and predicted values
y_true = []
y_pred = []

for k in range(len(list_of_df)):
    # Plotting code here...

    # Append actual values to y_true
    y_true.extend(actual_data[k][-prediction_length-context_length:].tolist())

    # Append predicted values to y_pred
    y_pred.extend(list_of_df[k]['0.5'].tolist())

# Now that we have all actual and predicted values, we can compute the metrics:
y_true = y_true[30:]
print(len(y_true), len(y_pred))
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('Root Mean Squared Error:', rmse)
print('R2 Score:', r2)

print(len(new_time_series_training))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")

fig, axs = plt.subplots(4, 1, figsize=(8, 15))
axs = axs.ravel() # Ravel turns a matrix into a vector, which is easier to iterate

for i in range(4):
    pred  = predictor.predict(new_time_series_training[i:i+1]) # predicted forecast
    act = new_time_series_test[i:i+1] # full data set
    act_pro = new_time_series_test_pro[1:2]

    for k in range(len(pred)):
        y_true = []
        y_pred = []
        y_true.extend(act[k][-prediction_length-context_length:].tolist())
        y_pred.extend(pred[k]['0.5'].tolist())
        y_true = y_true[30:]
        print(len(y_true), len(y_pred))
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        ape = np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))
        mape = np.mean(ape) * 100
        print("------>", i, i+1)
        print('Mean Squared Error:', mse)
        print('Mean Absolute Error:', mae)
        print('Root Mean Squared Error:', rmse)
        print('R2 Score:', r2)
        print('Mean Absolute Percentage Error (MAPE):', mape)

        act[k][-prediction_length-context_length:].plot(label='Actual', linewidth=2.5, ax=axs[i])
        p10 = pred[k]['0.1']
        p90 = pred[k]['0.9']
        axs[i].fill_between(p10.index, p10, p90, color='grey', alpha=0.3, label='80% Confidence Interval')
        pred[k]['0.5'].plot(label='Prediction Median', color='blue', linewidth=2.5, ax=axs[i])
        (pred[k]['0.9']+100).plot(label='Weighted Provision', color='green', linewidth=2.5, ax=axs[i])
                # Calculate and plot the mean of 0.9 percentile
        mean_90 = np.mean(pred[k]['0.9']+100)
        axs[i].axhline(mean_90, color='purple', linestyle='--', label='Mean [0.9 Percentile of Weighted Provision]')

        axs[i].set_title(f"DeepAR Model Prediction Test#{i+1} (MAPE : {round(mape)}%)")
        axs[i].set_ylabel("CPU usage [MHz]")
        axs[i].set_xlabel("Time")
        axs[i].legend(loc='upper left')

plt.tight_layout()
plt.savefig("deepar_pred.png")
plt.show()

print(len(y_true), len(y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")

fig, axs = plt.subplots(4, 1, figsize=(8, 15))
axs = axs.ravel() # Ravel turns a matrix into a vector, which is easier to iterate

for i in range(4):
    pred  = predictor.predict(new_time_series_training[i:i+1]) # predicted forecast
    act = new_time_series_test[i:i+1] # full data set
    act_pro = new_time_series_test_pro[i:i+1]

    for k in range(len(pred)):

        labels = ['Traditional Approach', 'Our Approach']
        total = [act_pro[k][-prediction_length-context_length:].mean(), mean_90]
        usage = [act[k][-prediction_length-context_length:].mean(), act[k][-prediction_length-context_length:].mean()]

        unused = [total[0] - usage[0], total[1] - usage[1]]

        axs[i].bar(labels, usage, label='CPU Usage', color='b')
        axs[i].bar(labels, unused, label='CPU Unused', bottom=usage, color='orange')

        axs[i].set_ylabel('CPU')
        axs[i].set_title(f'CPU Comparison: Traditional vs Our Approach (Test#{i+1})')
        axs[i].legend()


plt.tight_layout()
plt.savefig("deepar_comparison_test.png")
plt.show()

# Now that we have all actual and predicted values, we can compute the metrics:
y_true = y_true[3000:]
print(len(y_true), len(y_pred))
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('Root Mean Squared Error:', rmse)
print('R2 Score:', r2)

