# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================
# ConfidenceThresholdEvaluator Example
# Comparing estimated vs. real performance with cross-validation
# ==============================================================

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from skeval.evaluators.confidence import ConfidenceThresholdEvaluator


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
# 4. Initialize ConfidenceThresholdEvaluator
# ======================
scorers = {
    'accuracy': accuracy_score,
    'f1_macro': lambda y, p: f1_score(y, p, average='macro')
}

evaluator = ConfidenceThresholdEvaluator(
    model=model,
    scorer=scorers,
    threshold=0.65,
    verbose=False
)

# ======================
# 5. Train/test split for Geriatrics dataset
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.5, random_state=42, stratify=y1
)

# ======================
# 6. Fit model and evaluator
# ======================
evaluator.fit(X_train, y_train)

# ======================
# 7. Estimate performance using evaluator
# ======================
estimated_scores = evaluator.estimate(X_test)

# ======================
# 8. Compute real performance using cross-validation
# ======================
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
