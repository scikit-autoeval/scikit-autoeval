# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================
# RegressionNoiseEvaluator Example
# Comparing estimated vs. real performance with cross-validation
# ==============================================================

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from skeval.evaluators.regression_noise import RegressionNoiseEvaluator


# ======================
# 1. Load datasets
# ======================
df_geriatrics = pd.read_csv('./skeval/datasets/geriatria-controle-alzheimerLabel.csv')
df_neurology = pd.read_csv('./skeval/datasets/neurologia-controle-alzheimerLabel.csv')

# ======================
# 2. Separate features and target
# ======================
X1, y1 = df_geriatrics.drop(columns=['Alzheimer']), df_geriatrics['Alzheimer']
X2, y2 = df_neurology.drop(columns=['Alzheimer']), df_neurology['Alzheimer']

# ======================
# 3. Define model pipeline
# ======================
model = make_pipeline(
    KNNImputer(n_neighbors=5),
    RandomForestClassifier(n_estimators=300, random_state=42)
)

# ======================
# 4. Initialize RegressionNoiseEvaluator
# ======================
scorers = {
    'accuracy': accuracy_score,
    'f1_macro': lambda y, p: f1_score(y, p, average='macro')
}

evaluator = RegressionNoiseEvaluator(
    model=model,
    scorer=scorers,
    n_splits=5,
    verbose=True
)

# ======================
# 5. Fit evaluator with injected noise
# ======================
print("\n[INFO] Training evaluator with noise injection...")
evaluator.fit([X1, X2], [y1, y2], start_noise=0, end_noise=50, step_noise=25)

# ======================
# 6. Train/test split for Geriatrics dataset
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.5, random_state=42, stratify=y1
)

final_model = model.fit(X_train, y_train)
evaluator.model = final_model  # important: set trained model

# ======================
# 7. Estimate performance using evaluator
# ======================
print("\n[INFO] Estimating metrics on test set...")
estimated_scores = evaluator.estimate(X_test)

# ======================
# 8. Compute real performance with cross-validation
# ======================
print("\n[INFO] Computing real performance using cross-validation...")
real_scores = {}
for metric_name, scorer_fn in scorers.items():
    cv_score = cross_val_score(
        model, X1, y1, scoring='accuracy' if metric_name == 'accuracy' else 'f1_macro',
        cv=5
    ).mean()
    real_scores[metric_name] = cv_score

# ======================
# 9. Side-by-side comparison
# ======================
print("\n===== Estimated vs. Real Performance =====")
for metric in scorers.keys():
    print(f"{metric:<10} -> Real (CV): {real_scores[metric]:.4f} | Estimated: {estimated_scores[metric]:.4f}")
