# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

def check_is_fitted(model):
    """Check if the model has been fitted.

    This method uses `sklearn.utils.validation.check_is_fitted` to verify
    that the underlying model has been fitted. It raises a `RuntimeError`
    if the model is not fitted.

    Raises
    ------
    RuntimeError
        If the model has not been fitted yet.
    """
    return sklearn_check_is_fitted(model)

def get_CV_and_real_scores(model, scorers, X_train, y_train, X_test, y_test):
    """Compute cross-validation and real scores for a given model and datasets.

    This function computes the cross-validation scores on the test dataset
    (using cross_validate) and the real scores on the test dataset after
    fitting the model on X_train/y_train.

    Parameters
    ----------
    evaluator : object
        An evaluator instance with an `estimate` method.
    model : object
        A fitted model with `fit` and `predict` methods.
    X_train : array-like
        The training features.
    y_train : array-like
        The training target labels.
    X_test : array-like
        The test features.
    y_test : array-like
        The test target labels.

    Returns
    -------
    dict
        A dictionary containing 'cv_scores' and 'real_scores'.
    """
    
    # ======================
    # 1. Cross-Validation (intra-domain)
    #  This is the standard CV estimate: train/validate 
    # ======================
    cv_scores = {}
    scoring = {name: make_scorer(fn) for name, fn in scorers.items()}
    cv_result = cross_validate(
        model,
        X_test,
        y_test,
        scoring=scoring,
        cv=5,
        return_train_score=False
    )
    for metric_name in scorers.keys():
        cv_key = f"test_{metric_name}"
        cv_scores[metric_name] = cv_result[cv_key].mean()

    # ======================
    # 2. Real performance (train on geriatrics -> test on neurology)
    # ======================
    model.fit(X_train, y_train)
    y_pred_real = model.predict(X_test)

    real_scores = {
        metric_name: scorer_fn(y_test, y_pred_real)
        for metric_name, scorer_fn in scorers.items()
    }
    
    return {
        'cv_scores': cv_scores,
        'real_scores': real_scores
    }