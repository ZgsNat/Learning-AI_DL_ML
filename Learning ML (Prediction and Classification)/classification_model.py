import pandas as pd
from ydata_profiling import ProfileReport
import os
from sklearn.model_selection import train_test_split
# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct path to data file
data_path = os.path.join(current_dir, "diabetes.csv")
data = pd.read_csv(data_path)
# data = pd.read_csv("diabetes.csv")
# print(data.describe())
# print(data.info())
# print(data.corr())
# profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
# profile.to_file("diabetes.html")
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)