# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================
# AgreementEvaluator Example
# Comparing estimated vs. real agreement with cross-validation
# ==============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from ..evaluators.agreement import AgreementEvaluator


# ======================
# 1. Load datasets
# ======================
df_geriatrics = pd.read_csv('./skeval/datasets/geriatria-controle-alzheimerLabel.csv')

# ======================
# 2. Separate features and target
# ======================
X1, y1 = df_geriatrics.drop(columns=['Alzheimer']).values, df_geriatrics['Alzheimer'].values

# ======================
# 3. Define model pipelines
# ======================
model = make_pipeline(
    KNNImputer(n_neighbors=5),
    RandomForestClassifier(n_estimators=300, random_state=42)
)

sec_model = make_pipeline(
    KNNImputer(n_neighbors=5),
    GaussianNB()
)

# ======================
# 4. Define scorers and evaluator
# ======================
scorers = {
    'accuracy': accuracy_score,
    'f1_macro': lambda y, p: f1_score(y, p, average='macro')
}

evaluator = AgreementEvaluator(
    model=model,
    sec_model=sec_model,
    scorer=scorers,
    verbose=True
)

# ======================
# 5. Fit evaluator and estimate agreement
# ======================
evaluator.fit(X1, y1)
estimated_scores = evaluator.estimate(X1)

# ======================
# 6. Compute real agreement via cross-validation
# ======================
cv = KFold(n_splits=5, shuffle=True, random_state=42)
real_scores = {m: [] for m in scorers.keys()}

for train_idx, test_idx in cv.split(X1):
    X_train, X_test = X1[train_idx], X1[test_idx]
    y_train, y_test = y1[train_idx], y1[test_idx]

    m1 = model.fit(X_train, y_train)
    m2 = sec_model.fit(X_train, y_train)

    pred1 = m1.predict(X_test)
    pred2 = m2.predict(X_test)

    # Agreement vector: 1 when predictions match, 0 otherwise
    agreement = (pred1 == pred2).astype(int)
    true = np.ones_like(agreement)  # ideal perfect agreement = all ones

    for name, fn in scorers.items():
        real_scores[name].append(fn(true, agreement))

# Average over folds
real_scores = {k: np.mean(v) for k, v in real_scores.items()}

# ======================
# 7. Side-by-side comparison
# ======================
print("\n===== Estimated vs. Real Agreement =====")
for metric in scorers.keys():
    print(f"{metric:<10} -> Real (CV): {real_scores[metric]:.4f} | Estimated: {estimated_scores[metric]:.4f}")
