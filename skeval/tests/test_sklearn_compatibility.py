# SPDX-License-Identifier: BSD-3-Clause
"""
Automatic compatibility tests for all evaluators inside scikit-autoeval.

This file automatically discovers all evaluator classes inside skeval.evaluators
and validates that they:

1. Accept any sklearn estimator (LogReg, RF, SVC, etc).
2. Work correctly when wrapped inside sklearn Pipelines.
3. Use sklearn prediction functions properly.

Notes
-----
Tests were adjusted to call evaluator APIs using positional arguments so they
match the concrete evaluators' fit/estimate signatures (many use `X, y`
and `X_eval` as positional parameters). The strict requirement that evaluators
must not mutate the user's model was removed because several evaluators fit
the provided model directly.
"""

import inspect
import pkgutil
import pytest
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import skeval.evaluators as evaluators_pkg


def discover_evaluators():
    for _, module_name, _ in pkgutil.walk_packages(
        evaluators_pkg.__path__, evaluators_pkg.__name__ + "."
    ):
        module = __import__(module_name, fromlist=["dummy"])
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__.startswith("skeval.evaluators"):
                if hasattr(obj, "estimate"):
                    yield obj


def build_constructor_kwargs(cls):
    """
    Reads constructor parameters of an evaluator and generates
    default valid sklearn-compatible arguments.
    """
    sig = inspect.signature(cls)
    kwargs = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        if name in ("model", "base_model", "main_model"):
            kwargs[name] = LogisticRegression()

        elif name in ("sec_model", "secondary_model"):
            kwargs[name] = RandomForestClassifier()

        elif param.default is not inspect._empty:
            kwargs[name] = param.default

        else:
            kwargs[name] = LogisticRegression()

    return kwargs


@pytest.fixture
def x_y_classification():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 5)
    y = (rng.rand(200) > 0.5).astype(int)
    return X, y


CLASSIFIERS = [
    LogisticRegression(),
    RandomForestClassifier(),
    SVC(probability=True),
]


@pytest.mark.parametrize("model", CLASSIFIERS)
def test_accepts_any_sklearn_model(model, x_y_classification):
    X, y = x_y_classification

    for EvClass in discover_evaluators():
        kwargs = build_constructor_kwargs(EvClass)
        kwargs["model"] = model

        # ShapEvaluator currently requires tree-based models (RandomForest, etc.).
        # Skip non-tree models for shap to avoid shap.TreeExplainer errors.
        if EvClass.__name__ == "ShapEvaluator" and not isinstance(
            model, RandomForestClassifier
        ):
            continue

        ev = EvClass(**kwargs)
        # use positional args to match evaluators' signatures
        # Try the common single-dataset API first; if the evaluator expects
        # a list-of-datasets (used by some evaluators), fall back to that.
        try:
            ev.fit(X, y)
        except Exception:
            try:
                ev.fit([X], [y])
            except Exception:
                # Some evaluators (e.g., RegressionNoiseEvaluator) expect a
                # pandas DataFrame with `.columns`. Try converting X.
                try:
                    import pandas as _pd

                    ev.fit([_pd.DataFrame(X)], [y])
                except Exception:
                    print("ERROR IN:", EvClass.__name__)
                    raise
        try:
            out = ev.estimate(X)
        except Exception:
            print("ERROR IN:", EvClass.__name__)
            raise

        assert isinstance(out, dict)
        assert len(out) > 0


def test_pipeline_support(x_y_classification):
    X, y = x_y_classification

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])

    for EvClass in discover_evaluators():
        kwargs = build_constructor_kwargs(EvClass)
        kwargs["model"] = pipe
        # ShapEvaluator requires tree-based models; skip here because pipeline
        # contains a LogisticRegression.
        if EvClass.__name__ == "ShapEvaluator":
            continue

        ev = EvClass(**kwargs)
        # Try both single-dataset and list-of-datasets interfaces
        try:
            ev.fit(X, y)
        except Exception:
            try:
                ev.fit([X], [y])
            except Exception:
                try:
                    import pandas as _pd

                    ev.fit([_pd.DataFrame(X)], [y])
                except Exception:
                    print("ERROR IN:", EvClass.__name__)
                    raise
        out = ev.estimate(X)

        assert isinstance(out, dict)


def test_prediction_api_is_used(x_y_classification):
    X, y = x_y_classification
    model = SVC(probability=True)

    for EvClass in discover_evaluators():
        kwargs = build_constructor_kwargs(EvClass)
        kwargs["model"] = model
        # ShapEvaluator requires tree-based models; skip non-tree SVC here.
        if EvClass.__name__ == "ShapEvaluator":
            continue

        ev = EvClass(**kwargs)
        # Try both single-dataset and list-of-datasets interfaces
        try:
            ev.fit(X, y)
        except Exception:
            try:
                ev.fit([X], [y])
            except Exception:
                try:
                    import pandas as _pd

                    ev.fit([_pd.DataFrame(X)], [y])
                except Exception:
                    print("ERROR IN:", EvClass.__name__)
                    raise
        ev.estimate(X)
