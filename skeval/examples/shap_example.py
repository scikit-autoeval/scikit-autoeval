# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================
# ShapEvaluator Example
# ==============================================================

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from skeval.evaluators.shap import ShapEvaluator
from skeval.utils import get_cv_and_real_scores, print_comparison


def run_shap_eval(verbose=False):

    geriatrics = pd.read_csv("./skeval/datasets/geriatria-controle-alzheimerLabel.csv")
    neurology = pd.read_csv("./skeval/datasets/neurologia-controle-alzheimerLabel.csv")

    x_geriatria, y_geriatria = geriatrics.drop(columns=["Alzheimer"]), geriatrics["Alzheimer"]
    x_neurologia = neurology.drop(columns=["Alzheimer"])

    # Define pipeline
    model = make_pipeline(KNNImputer(n_neighbors=5), XGBClassifier())

    # Define scorers and evaluator
    scorers = {
        "accuracy": accuracy_score,
        "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
    }

    evaluator = ShapEvaluator(
        model=model,
        scorer=scorers,
        verbose=False,
        inner_clf=XGBClassifier(random_state=42),
    )

    # Fit evaluator
    evaluator.fit(x_geriatria, y_geriatria)

    # Estimate performance
    estimated_scores = evaluator.estimate(x_neurologia)

    # =====================================
    # 7. Compute real and CV performance
    # =====================================
    train_data = X1, y1
    test_data = X2, y2
    scores_dict = get_cv_and_real_scores(
        model=model, scorers=scorers, train_data=train_data, test_data=test_data
    )
    cv_scores = scores_dict["cv_scores"]
    real_scores = scores_dict["real_scores"]

    if verbose:
        print_comparison(scorers, cv_scores, estimated_scores, real_scores)

    return {"cv": cv_scores, "estimated": estimated_scores, "real": real_scores}


if __name__ == "__main__":
    results = run_shap_eval(verbose=True)
