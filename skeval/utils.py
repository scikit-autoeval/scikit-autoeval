# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

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

    This function computes the cross-validation scores on the training dataset
    and the real scores on the test dataset using the provided evaluator and model.

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
    cv_scores = {metric: [] for metric in scorers.keys()}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, _ in kf.split(X_train):
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test) 
        
        for metric in scorers.keys():
            cv_scores[metric].append(scorers[metric](y_test, y_pred_fold))

    cv_scores = {metric: np.mean(values) for metric, values in cv_scores.items()}

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