import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

data = pd.read_csv(
    "heart_disease.data",
    header=None,
    names=columns,
    na_values="?"
)

data["ca"] = data["ca"].fillna(data["ca"].median())
data["thal"] = data["thal"].fillna(data["thal"].median())

data["target"] = (data["target"] > 0).astype(int)

print(data.head())
print(data.isnull().sum())
print(data["target"].value_counts())

X = data.drop("target", axis=1)
y = data["target"]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
report = classification_report(y_test, predictions)

print("Classification Report:")
print(report)
joblib.dump(model, "health_risk_model.pkl")

print("Model saved successfully!")