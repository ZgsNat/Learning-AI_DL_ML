import pandas as pd
from ydata_profiling import ProfileReport
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = SVC(probability=True)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# for i, j in zip(y_test, y_pred):
#     print(f"True: {i}, Predicted: {j}")
print("Model prediction: ", y_pred)
print("Model trained")
print("Model score: ", model.score(x_test, y_test))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))