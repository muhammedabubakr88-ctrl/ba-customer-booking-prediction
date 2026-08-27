"""
Task 2 - Step 2: Train Random Forest, cross-validate, evaluate
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json

df = pd.read_csv('outputs/prepared_data.csv')

X = df.drop(columns=['booking_complete'])
y = df['booking_complete']

# Train/test split (stratified, since target is imbalanced)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# ---- 5-fold cross-validation on training set ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(rf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

print("=" * 50)
print("5-FOLD CROSS-VALIDATION RESULTS (training set)")
print("=" * 50)
cv_summary = {}
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    cv_summary[metric] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"{metric:12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# ---- Fit on full training set, evaluate on held-out test set ----
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

test_metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_proba),
}

print("\n" + "=" * 50)
print("HELD-OUT TEST SET RESULTS")
print("=" * 50)
for k, v in test_metrics.items():
    print(f"{k:12s}: {v:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Booked', 'Booked']))

# ---- Feature importance ----
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)
print(importances.to_string(index=False))

# Save everything for later use (plotting, slide)
importances.to_csv('outputs/feature_importance.csv', index=False)

results = {
    'cv_summary': cv_summary,
    'test_metrics': test_metrics,
    'confusion_matrix': cm.tolist(),
    'baseline_accuracy_if_predict_majority': float((y == 0).mean())
}
with open('outputs/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved outputs/feature_importance.csv and outputs/results.json")