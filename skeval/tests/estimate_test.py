import os
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from evaluators.confidence import ConfidenceThresholdEvaluator

def load_dataset(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Alzheimer"])
    y = df["Alzheimer"]
    return X, y

def test_estimate(train_csv, test_csv, threshold=0.7):
    train_path = os.path.join("datasets", train_csv)
    test_path = os.path.join("datasets", test_csv)

    X_train, y_train = load_dataset(train_path)
    X_test, _ = load_dataset(test_path)

    model = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    evaluator = ConfidenceThresholdEvaluator(
        estimator=model,
        threshold=threshold,
        limit_to_top_class=True
    )

    evaluator.fit(X_train, y_train)

    estimate_result = evaluator.estimate(X_test)

    print(f"Estimate from model trained on '{train_csv}' and tested on '{test_csv}':")
    print(estimate_result)
    print("-" * 60)

test_estimate("geriatria-controle-alzheimerLabel.csv", "neurologia-controle-alzheimerLabel.csv")
test_estimate("neurologia-controle-alzheimerLabel.csv", "geriatria-controle-alzheimerLabel.csv")
