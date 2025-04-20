import pandas as pd
from ydata_profiling import ProfileReport

data = pd.read_csv("diabetes.csv")
# print(data.describe())
# print(data.info())
# print(data.corr())
profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
# profile.to_file("diabetes.html")
