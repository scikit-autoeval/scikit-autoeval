# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================
# ConfidenceThresholdEvaluator Example
# ==============================================================

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from ..evaluators.confidence import ConfidenceThresholdEvaluator
from ..utils import get_CV_and_real_scores

# ======================
# 1. Load datasets
# ======================
df_geriatrics = pd.read_csv('./skeval/datasets/geriatria-controle-alzheimerLabel.csv')
df_neurology  = pd.read_csv('./skeval/datasets/neurologia-controle-alzheimerLabel.csv')

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
# 5. Fit evaluator on geriatrics dataset (learn internal behavior)
# ======================
evaluator.fit(X1, y1)

# ======================
# 6. Estimated performance (apply evaluator to neurology)
# ======================
estimated_scores = evaluator.estimate(X2)

# ======================
# 7. Cross-Validation and Real Performance
#  
# ======================
scores_dict = get_CV_and_real_scores(model=model, scorers=scorers, X_train=X1, y_train=y1, X_test=X2, y_test=y2)
cv_scores = scores_dict['cv_scores']
real_scores = scores_dict['real_scores']

# ======================
# 8. Side-by-side comparison
# ======================
print("\n===== CV (intra-domain) vs. Estimated vs. Real (train Geriatrics -> test Neurology) =====")
for metric in scorers.keys():
    print(
        f"{metric:<10} -> CV: {cv_scores[metric]:.4f} | "
        f"Estimated: {estimated_scores[metric]:.4f} | "
        f"Real: {real_scores[metric]:.4f}"
    )

# ======================
# 9. Absolute error comparison (distance to Real)
# ======================
print("\n===== Absolute Error w.r.t. Real Performance =====")
for metric in scorers.keys():
    err_est = abs(real_scores[metric] - estimated_scores[metric])
    err_cv  = abs(real_scores[metric] - cv_scores[metric])
    print(
        f"{metric:<10} -> |Real - Estimated|: {err_est:.4f} | "
        f"|Real - CV|: {err_cv:.4f}"
    )
