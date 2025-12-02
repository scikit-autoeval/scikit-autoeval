# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================
# AgreementEvaluator Example
# ==============================================================

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from skeval.evaluators.agreement import AgreementEvaluator
from skeval.utils import get_cv_and_real_scores, print_comparison


def run_agreement_eval(verbose=False):
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
    # 3. Define model pipelines
    # ======================
    model = make_pipeline(KNNImputer(n_neighbors=5), XGBClassifier())

    sec_model = make_pipeline(
        KNNImputer(n_neighbors=5),
        XGBClassifier(),
    )
    # rede neural
    # sec_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # ======================
    # 4. Define scorers and evaluator
    # ======================
    scorers = {
        "accuracy": accuracy_score,
        "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
    }

    evaluator = AgreementEvaluator(
        model=model, sec_model=sec_model, scorer=scorers, verbose=False
    )

    # ======================
    # 5. Fit evaluator and estimate agreement
    # ======================
    evaluator.fit(X1, y1)
    estimated_scores = evaluator.estimate(X2)

    # ======================
    # 6. Cross-Validation and Real Performance
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
    results = run_agreement_eval(verbose=True)
