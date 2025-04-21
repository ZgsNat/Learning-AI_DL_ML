import pandas as pd
from ydata_profiling import ProfileReport
import os
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