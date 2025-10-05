# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================
# RegressionEvaluator Example
# Comparing estimated vs. real performance with cross-validation
# ==============================================================
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from ..evaluators import RegressionEvaluator

# =====================================
# 1. Load datasets
# =====================================
geriatrics = pd.read_csv("./skeval/datasets/geriatria-controle-alzheimerLabel.csv")
neurology = pd.read_csv("./skeval/datasets/neurologia-controle-alzheimerLabel.csv")

# =====================================
# 2. Separate features and target
# =====================================
X1, y1 = geriatrics.drop(columns=["Alzheimer"]), geriatrics["Alzheimer"]
X2, y2 = neurology.drop(columns=["Alzheimer"]), neurology["Alzheimer"]

# =====================================
# 3. Define pipeline (KNNImputer + RandomForest)
# =====================================
model = make_pipeline(
    KNNImputer(n_neighbors=5),
    RandomForestClassifier(n_estimators=300, random_state=42)
)

# =====================================
# 4. Define scorers and evaluator
# =====================================
scorers = {
    "accuracy": accuracy_score,
    "f1_macro": lambda y, p: f1_score(y, p, average="macro")
}

evaluator = RegressionEvaluator(
    model=model,
    scorer=scorers,
    n_splits=4,
    verbose=True
)

# =====================================
# 5. Fit evaluator using multiple datasets
# =====================================
evaluator.fit([X1, X2], [y1, y2])
final_model = model.fit(X1, y1)

# =====================================
# 6. Estimate scores for new dataset
# =====================================
estimated_scores = evaluator.estimate(X1)

# =====================================
# 7. Compute real scores via cross-validation
# =====================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    model, X1, y1, cv=cv,
    scoring={'accuracy': 'accuracy', 'f1_macro': 'f1_macro'},
    return_train_score=False
)
real_scores = {
    'accuracy': cv_results['test_accuracy'].mean(),
    'f1_macro': cv_results['test_f1_macro'].mean()
}

# =====================================
# 8. Compare estimated vs. real cross-validation results
# =====================================
print("\n=== Comparison: Estimated vs. Real Cross-Validation Scores ===")
for metric in scorers.keys():
    print(f"{metric:<10} | Estimated: {estimated_scores[metric]:.4f} | Real (CV): {real_scores[metric]:.4f}")
