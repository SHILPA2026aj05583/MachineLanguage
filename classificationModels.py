import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef
)

# 1. Load Dataset
df = pd.read_csv('bank-full.csv', sep=';')

# 2. Features and Target separation
X = df.drop(columns=['y'])
y = (df['y'] == 'yes').astype(int)

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
#cat_cols = X.select_dtypes(include=['object']).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

# 3. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Scaling & Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# 5. Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbor Classifier": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes Classifier (Gaussian)": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

# 6. Model Training & Evaluation Loop
results = []
for name, model in models.items():
    model.fit(X_train_prep, y_train)
    y_pred = model.predict(X_test_prep)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_prep)[:, 1]
    else:
        y_proba = model.decision_function(X_test_prep)
        
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "MCC Score": matthews_corrcoef(y_test, y_pred)
    })

# Output Results
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

import joblib

joblib.dump(
    {
        "preprocessor": preprocessor,
        "models": models,
        "feature_columns": X.columns.tolist()
    },
    "model_bundle.joblib"
)

# Export the held-out test data, including its labels
test_data = X_test.copy()
test_data["y"] = y_test
test_data.to_csv("test_data.csv", index=False)
