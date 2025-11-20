# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================
# RegressionEvaluator Example
# ==============================================================
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from skeval.evaluators import RegressionEvaluator
from skeval.utils import get_cv_and_real_scores, print_comparison


def run_regression_eval(verbose=False):
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
    model = RandomForestClassifier(n_estimators=180, random_state=42)

    # =====================================
    # 4. Define scorers and evaluator
    # =====================================
    scorers = {
        "accuracy": accuracy_score,
        "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
    }

    evaluator = RegressionEvaluator(model=model, scorer=scorers, verbose=False)

    # =====================================
    # 5. Fit evaluator using multiple datasets
    # =====================================
    evaluator.fit([X1, X2], [y1, y2], n_splits=4)

    # =====================================
    # 6. Estimate scores for new dataset
    # =====================================
    estimated_scores = evaluator.estimate(X2)

    # ======================
    # 7. Cross-Validation and Real Performance
    #
    # ======================
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
    results = run_regression_eval(verbose=True)
