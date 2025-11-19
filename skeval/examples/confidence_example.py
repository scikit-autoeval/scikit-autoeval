# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from skeval.evaluators.confidence import ConfidenceThresholdEvaluator
from skeval.utils import get_cv_and_real_scores, print_comparison


def run_confidence_eval(verbose=False):
    # ======================
    # 1. Load datasets
    # ======================
    df_geriatrics = pd.read_csv(
        "./skeval/datasets/geriatria-controle-alzheimerLabel.csv"
    )
    df_neurology = pd.read_csv(
        "./skeval/datasets/neurologia-controle-alzheimerLabel.csv"
    )

    # ======================
    # 2. Separate features and target
    # ======================
    X1, y1 = df_geriatrics.drop(columns=["Alzheimer"]), df_geriatrics["Alzheimer"]
    X2, y2 = df_neurology.drop(columns=["Alzheimer"]), df_neurology["Alzheimer"]

    # ======================
    # 3. Define model pipeline
    # ======================
    model = make_pipeline(
        KNNImputer(n_neighbors=4),
        RandomForestClassifier(n_estimators=300, random_state=42),
    )

    # ======================
    # 4. Initialize evaluator
    # ======================
    scorers = {
        "accuracy": accuracy_score,
        "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
    }

    evaluator = ConfidenceThresholdEvaluator(
        model=model, scorer=scorers, threshold=0.65, verbose=False
    )

    # ======================
    # 5. Fit evaluator
    # ======================
    evaluator.fit(X1, y1)

    # ======================
    # 6. Estimated performance
    # ======================
    estimated_scores = evaluator.estimate(X2)

    # ======================
    # 7. CV and Real performance
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
    results = run_confidence_eval(verbose=True)
