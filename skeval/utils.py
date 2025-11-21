# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, Callable, Dict, Mapping, Tuple
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer


def check_is_fitted(model: Any) -> None:
    """Check if the model has been fitted.

    This method uses `sklearn.utils.validation.check_is_fitted` to verify
    that the underlying model has been fitted. It raises a `RuntimeError`
    if the model is not fitted.

    Raises
    ------
    RuntimeError
        If the model has not been fitted yet.
    """
    if not hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
        raise ValueError("The model must implement predict_proba or decision_function.")
    sklearn_check_is_fitted(model)


def get_cv_and_real_scores(
    model: Any,
    scorers: Mapping[str, Callable[[Any, Any], float]],
    train_data: Tuple[Any, Any],
    test_data: Tuple[Any, Any],
) -> Dict[str, Dict[str, float]]:
    """Compute cross-validation and real scores for a given model and datasets.

    Parameters
    ----------
    evaluator : object
        An evaluator instance with an `estimate` method.
    model : object
        A fitted model with `fit` and `predict` methods.
    x_train : array-like
        The training features.
    y_train : array-like
        The training target labels.
    x_test : array-like
        The test features.
    y_test : array-like
        The test target labels.

    Returns
    -------
    dict
        A dictionary containing 'cv_scores' and 'real_scores'.
    """
    x_train, y_train = train_data
    x_test, y_test = test_data
    cv_scores = {}
    scoring = {name: make_scorer(fn) for name, fn in scorers.items()}
    cv_result = cross_validate(
        model, x_test, y_test, scoring=scoring, cv=5, return_train_score=False
    )
    for metric_name in scorers.keys():
        cv_key = f"test_{metric_name}"
        cv_scores[metric_name] = cv_result[cv_key].mean()
    model.fit(x_train, y_train)
    y_pred_real = model.predict(x_test)

    real_scores = {
        metric_name: scorer_fn(y_test, y_pred_real)
        for metric_name, scorer_fn in scorers.items()
    }

    return {"cv_scores": cv_scores, "real_scores": real_scores}


def print_comparison(
    scorers: Mapping[str, Any],
    cv_scores: Mapping[str, float],
    estimated_scores: Mapping[str, float],
    real_scores: Mapping[str, float],
) -> None:
    """
    Print a formatted comparison between cross-validation (intra-domain), estimated, and
    real performance scores for a set of metrics, and display the absolute errors of the
    CV and estimated scores with respect to the real scores.
    Parameters
    ----------
    scorers : Mapping[str, Any]
        A mapping whose keys are metric names (strings). Only the keys are used to
        determine which metrics to display; values (e.g., scorer callables) are not
        inspected by this function.
    cv_scores : Mapping[str, float]
        Mapping from metric name to the cross-validation (intra-domain) score.
    estimated_scores : Mapping[str, float]
        Mapping from metric name to the estimated cross-domain score.
    real_scores : Mapping[str, float]
        Mapping from metric name to the observed real score on the target domain.
    Returns
    -------
    None
    """
    print("\n===== CV vs. Estimated vs. Real =====")
    for metric in scorers.keys():
        print(
            f"{metric:<10} -> CV: {cv_scores[metric]:.4f} | "
            f"Estimated: {estimated_scores[metric]:.4f} | "
            f"Real: {real_scores[metric]:.4f}"
        )

    print("\n===== Absolute Error w.r.t. Real Performance =====")
    for metric in scorers.keys():
        err_est = abs(real_scores[metric] - estimated_scores[metric])
        err_cv = abs(real_scores[metric] - cv_scores[metric])
        print(
            f"{metric:<10} -> |Real - Estimated|: {err_est:.4f} | "
            f"|Real - CV|: {err_cv:.4f}"
        )
