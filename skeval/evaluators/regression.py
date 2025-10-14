# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ..base import BaseEvaluator
from ..utils import check_is_fitted

class RegressionEvaluator(BaseEvaluator):
    """Regression-based evaluator for classification models.

    This evaluator estimates the performance of a classification model (e.g.,
    accuracy, precision) without requiring labeled test data. It works by
    training a meta-regressor (e.g., Random Forest) that learns to map
    meta-features, extracted from a classifier's probability distributions,
    to its true performance score on unseen data.

    To use this evaluator, it must first be fitted on a collection of diverse
    datasets using the same model type to learn this mapping robustly.

    Parameters
    ----------
    model : object
        An unfitted classifier instance that will be used as the base model.
        Clones of this model will be trained during the `fit` process and
        the same model type must later be fitted manually before `estimate`.
    meta_regressor : object, default=None
        A regression model implementing `fit` and `predict`. If None, a
        `RandomForestRegressor` with 500 trees is used for each scorer.
    n_splits : int, default=5
        Number of random splits per dataset to generate multiple meta-examples.
    scorer : callable or dict of str -> callable, default=accuracy_score
        The performance metric(s) to estimate. If a dictionary is provided, a
        separate meta-regressor will be trained to estimate each metric.
    verbose : bool, default=False
        If True, prints informational messages during the training of the
        meta-regressors and during estimation.

    Attributes
    ----------
    meta_regressors_ : dict
        A dictionary mapping each scorer's name to its fitted meta-regressor
        instance. This attribute is populated after the `fit` method is called.
    
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.metrics import accuracy_score
    >>> from skeval.evaluators.regression import RegressionEvaluator
    >>>
    >>> iris = load_iris()
    >>> X_list, y_list = [iris.data], [iris.target]
    >>> evaluator = RegressionEvaluator(
    ...     model=LogisticRegression(max_iter=1000),
    ...     scorer=accuracy_score,
    ...     verbose=False
    ... )
    >>> evaluator.fit(X_list, y_list)
    RegressionEvaluator(...)
    >>> # Train a final model manually
    >>> final_model = LogisticRegression(max_iter=1000).fit(iris.data, iris.target)
    >>> evaluator.model = final_model
    >>> estimated_scores = evaluator.estimate(iris.data)
    >>> print(estimated_scores)
    {'score': ...}
    """
    def __init__(self, model, scorer=accuracy_score, verbose=False, meta_regressor=None, n_splits=5):
        super().__init__(model=model, scorer=scorer, verbose=verbose)

        self.meta_regressor = meta_regressor or RandomForestRegressor(n_estimators=500, random_state=42)
        self.n_splits = n_splits

    def fit(self, X, y):
        """Trains the internal meta-regressor(s) using a single model type.

        This method builds a meta-dataset to train the evaluator. For each dataset,
        it performs multiple random splits to increase the number of meta-examples.

        Parameters
        ----------
        X : list of array-like
            A list of datasets (features) used to train the meta-model.
        y : list of array-like
            A list of labels corresponding to `X`.

        Returns
        -------
        self : object
            The fitted evaluator instance.
        """
        scorers_names = self._get_scorer_names()
        meta_features = []
        meta_targets = {name: [] for name in scorers_names}

        for X_i, y_i in zip(X, y):
            for split in range(self.n_splits):
                est = clone(self.model)

                stratify_y = y_i if len(np.unique(y_i)) > 1 else None
                X_train_meta, X_holdout_meta, y_train_meta, y_holdout_meta = train_test_split(
                    X_i, y_i, test_size=0.33, random_state=42 + split, stratify=stratify_y
                )

                est.fit(X_train_meta, y_train_meta)
                feats = self._extract_metafeatures(est, X_holdout_meta)
                y_pred_holdout = est.predict(X_holdout_meta)

                meta_features.append(feats.flatten())

                if isinstance(self.scorer, dict):
                    for name in scorers_names:
                        func = self.scorer[name]
                        score = func(y_holdout_meta, y_pred_holdout)
                        meta_targets[name].append(score)
                elif callable(self.scorer):
                    score = self.scorer(y_holdout_meta, y_pred_holdout)
                    meta_targets['score'].append(score)

        meta_features = np.array(meta_features)
        self.meta_regressors_ = {}

        for name in scorers_names:
            meta_y = np.array(meta_targets[name])
            reg = clone(self.meta_regressor)
            reg.fit(meta_features, meta_y)
            self.meta_regressors_[name] = reg

            if self.verbose:
                print(f"[INFO] Meta-regressor for '{name}' has been trained.")
        
        self.model.fit(X[0], y[0])
        return self


    def estimate(self, X_eval):
        """Estimates the performance of the current model on unlabeled data.

        The model assigned to `self.model` must already be a fitted classifier
        (manually trained by the user). This method extracts meta-features from
        its predictions on the unlabeled data `X_eval` and uses the pre-trained
        meta-regressor(s) to predict the performance scores.

        Parameters
        ----------
        X_eval : array-like of shape (n_samples, n_features)
            The unlabeled input data.

        Returns
        -------
        dict
            A dictionary with the estimated scores, where keys are the names
            of the scorers and values are the predicted performance scores.
        
        Raises
        ------
        RuntimeError
            If the `estimate` method is called before the evaluator has been
            fitted with the `fit` method.
        ValueError
            If `self.model` does not implement `predict_proba`.
        """
        check_is_fitted(self.model)
        
        if not hasattr(self, "meta_regressors_"):
            raise RuntimeError("The evaluator has not been fitted yet. Call 'fit' before 'estimate'.")
            
        feats = self._extract_metafeatures(self.model, X_eval)
        scores = {}
        
        for name, reg in self.meta_regressors_.items():
            estimated_score = reg.predict(feats)[0]
            scores[name] = estimated_score
            if self.verbose:
                print(f"[INFO] Estimated {name}: {estimated_score:.4f}")
        return scores

    def _extract_metafeatures(self, estimator, X):
        """Extracts meta-features from a fitted model's predicted probabilities.

        The extracted features include the mean and standard deviation of the
        maximum prediction probabilities (confidence) and the mean and standard
        deviation of the entropy of the probability distributions.

        Parameters
        ----------
        estimator : fitted classifier
            The classifier from which to extract prediction probabilities.
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        ndarray
            A 2D numpy array of shape (1, n_meta_features) containing the
            extracted features.
        
        Raises
        ------
        ValueError
            If the provided estimator does not have a `predict_proba` method.
        """
        if not hasattr(estimator, "predict_proba"):
            raise ValueError("The estimator must implement predict_proba.")
            
        probas = estimator.predict_proba(X)
        max_probs = np.max(probas, axis=1)
        eps = 1e-12
        entropy = -np.sum(probas * np.log(probas + eps), axis=1)
        
        features = {
            "mean_conf": np.mean(max_probs),
            "std_conf": np.std(max_probs),
            "mean_entropy": np.mean(entropy),
            "std_entropy": np.std(entropy)
        }
        return np.array(list(features.values())).reshape(1, -1)