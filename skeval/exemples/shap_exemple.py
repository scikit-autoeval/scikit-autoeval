# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================
# ShapEvaluator Example
# ==============================================================

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from skeval.evaluators.shap import ShapEvaluator
from skeval.utils import get_CV_and_real_scores 

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

evaluator = ShapEvaluator(
    model=model,
    scorer=scorers,
    verbose=False
)

# =====================================
# 5. Fit evaluator on geriatrics data
# =====================================
evaluator.fit(X1, y1)

# =====================================
# 6. Estimate performance (train on X1, estimate on X2)
# =====================================
# Transform data using only the preprocessing part of the pipeline
imputer = evaluator.model.named_steps["knnimputer"]
X1_proc = evaluator.model[:-1].transform(X1.values)
X2_proc = evaluator.model[:-1].transform(X2.values)

final_model = evaluator.model[-1]
estimated_scores = evaluator.estimate(X2_proc, X_train=X1_proc, y_train=y1)

# =====================================
# 7. Compute real and CV performance
# =====================================
scores_dict = get_CV_and_real_scores(model=model, scorers=scorers, X_train=X1, y_train=y1, X_test=X2, y_test=y2)
cv_scores = scores_dict['cv_scores']
real_scores = scores_dict['real_scores']

# =====================================
# 8. Final comparison
# =====================================
print("\n===== CV (intra-domain) vs. Estimated vs. Real (train Geriatrics -> test Neurology) =====")
for metric in scorers.keys():
    print(
        f"{metric:<10} -> CV: {cv_scores[metric]:.4f} | "
        f"Estimated: {estimated_scores[metric]:.4f} | "
        f"Real: {real_scores[metric]:.4f}"
    )


print("\n===== Absolute Error w.r.t. Real Performance =====")
for metric in scorers.keys():
    err_est = abs(real_scores[metric] - estimated_scores[metric])
    err_cv  = abs(real_scores[metric] - cv_scores[metric])
    print(
        f"{metric:<10} -> |Real - Estimated|: {err_est:.4f} | "
        f"|Real - CV|: {err_cv:.4f}"
    )
