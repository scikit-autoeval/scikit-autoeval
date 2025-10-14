# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================
# AgreementEvaluator Example
# ==============================================================

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from ..evaluators.agreement import AgreementEvaluator
from ..utils import get_CV_and_real_scores

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
# 3. Define model pipelines
# ======================
model = make_pipeline(
    KNNImputer(n_neighbors=5),
    GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
)

sec_model = make_pipeline(
    KNNImputer(n_neighbors=5),
    SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)
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
    verbose=False
)

# ======================
# 5. Fit evaluator and estimate agreement
# ======================
evaluator.fit(X1, y1)
estimated_scores = evaluator.estimate(X2)


# ======================
# 6. Cross-Validation and Real Performance
# ======================
scores_dict = get_CV_and_real_scores(model=model, scorers=scorers, X_train=X1, y_train=y1, X_test=X2, y_test=y2)
cv_scores = scores_dict['cv_scores']
real_scores = scores_dict['real_scores']

# ======================
# 7. Side-by-side comparison
# ======================
print("\n===== CV (intra-domain) vs. Estimated vs. Real (train Geriatrics -> test Neurology) =====")
for metric in scorers.keys():
    print(
        f"{metric:<10} -> CV: {cv_scores[metric]:.4f} | "
        f"Estimated: {estimated_scores[metric]:.4f} | "
        f"Real: {real_scores[metric]:.4f}"
    )

# ======================
# 8. Absolute error comparison (distance to Real)
# ======================
print("\n===== Absolute Error w.r.t. Real Performance =====")
for metric in scorers.keys():
    err_est = abs(real_scores[metric] - estimated_scores[metric])
    err_cv  = abs(real_scores[metric] - cv_scores[metric])
    print(
        f"{metric:<10} -> |Real - Estimated|: {err_est:.4f} | "
        f"|Real - CV|: {err_cv:.4f}"
    )
