import pandas as pd
import pytest
import os
import sys
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluators.confidence import ConfidenceThresholdEvaluator
from metrics.comparison import score_error

@pytest.fixture
def datasets():
    def load_dataset(path):
        df = pd.read_csv(path)
        X = df.drop(columns=["Alzheimer"])
        y = df["Alzheimer"]
        return X, y

    X_geri, y_geri = load_dataset("datasets/geriatria-controle-alzheimerLabel.csv")
    X_neuro, y_neuro = load_dataset("datasets/neurologia-controle-alzheimerLabel.csv")
    return X_geri, y_geri, X_neuro, y_neuro

@pytest.fixture
def evaluator():
    model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000))
    return ConfidenceThresholdEvaluator(
        estimator=model,
        scorer={"acc": accuracy_score, "mse": mean_squared_error},
        threshold=0.7
    )

def run_score_error(evaluator, X_train, y_train, X_test, y_test):
    evaluator.fit(X_train, y_train)
    y_pred = evaluator.estimator.predict(X_test)

    real_score = {
        "acc": accuracy_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred)
    }

    estimated_score = evaluator.estimate(X_test)

    return score_error(
        real_scores=real_score,
        estimated_scores=estimated_score,
        comparator={"acc": mean_absolute_error, "mse": mean_squared_error},
        verbose=False
    )

def test_geriatria_to_neurologia(evaluator, datasets):
    X_geri, y_geri, X_neuro, y_neuro = datasets
    errors = run_score_error(evaluator, X_geri, y_geri, X_neuro, y_neuro)
    
    assert "acc" in errors
    assert "mse" in errors
    assert isinstance(errors["acc"], float)
    assert isinstance(errors["mse"], float)

def test_neurologia_to_geriatria(evaluator, datasets):
    X_geri, y_geri, X_neuro, y_neuro = datasets
    errors = run_score_error(evaluator, X_neuro, y_neuro, X_geri, y_geri)

    assert "acc" in errors
    assert "mse" in errors
    assert isinstance(errors["acc"], float)
    assert isinstance(errors["mse"], float)
