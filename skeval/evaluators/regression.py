# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from skeval.base import BaseEvaluator


class RegressionBasedEvaluator(BaseEvaluator):
    """
    Regression-based evaluator for classification models.
    
    This evaluator estimates the accuracy (or another performance metric) of a
    classification model without requiring labeled test data. It does so by
    extracting meta-features from the model's predicted probability
    distributions (e.g., mean confidence, entropy) and feeding them into a
    regression meta-model that has been trained to map such features to real
    performance values.

    Parameters
    ----------
    model : object
        Any model with `fit`, `predict`, and `predict_proba` methods.
    scorer : callable or dict of str -> callable, default=accuracy_score
        An evaluation function or a dictionary of multiple evaluation functions.
    meta_regressor : object, default=None
        A regression model implementing `fit` and `predict`. If None, a
        RandomForestRegressor with 100 trees is used.
    verbose : bool, default=False
        If True, prints additional information during training and estimation.

    Attributes
    ----------
    model : object
        The classification model passed in the constructor.
    scorer : callable or dict
        The scoring function(s) passed in the constructor.
    meta_regressor : object
        The regression model trained to predict classifier accuracy.
    verbose : bool
        The verbosity level.
    """

    def __init__(self, model, scorer=accuracy_score, meta_regressor=None, verbose=False):
        self.model = model
        self.scorer = scorer
        self.meta_regressor = meta_regressor or RandomForestRegressor(n_estimators=100, random_state=42)
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fits the base classification model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target labels.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.model.fit(X, y)
        return self

    def estimate(self, X):
        """
        Estimates the model's performance on unlabeled data using
        regression-based meta-features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Unlabeled input data.

        Returns
        -------
        dict
            Dictionary containing the estimated performance scores.
        """
        feats = self.__extract_metafeatures(X)
        acc_estimate = self.meta_regressor.predict(feats)[0]

        if self.verbose:
            print("[INFO] Extracted meta-features:", feats.flatten())
            print("[INFO] Estimated accuracy:", acc_estimate)

        # Support for multiple scorers
        if isinstance(self.scorer, dict):
            scores = {name: func([acc_estimate], [acc_estimate]) for name, func in self.scorer.items()}
            if self.verbose:
                print("[INFO] Estimated scores:", scores)
            return scores
        elif callable(self.scorer):
            score = self.scorer([acc_estimate], [acc_estimate])
            if self.verbose:
                print("[INFO] Estimated score:", score)
            return {"score": score}
        else:
            raise ValueError("'scorer' must be a callable or a dict of callables.")

    def __extract_metafeatures(self, X):
        """
        Extracts meta-features from the model's predicted probabilities.

        Features include:
        - mean confidence (average max probability per sample)
        - std confidence
        - mean entropy
        - std entropy

        Parameters
        ----------
        X : array-like
            Input data to extract meta-features from.

        Returns
        -------
        ndarray of shape (1, n_features)
            Extracted meta-features.
        """
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
        else:
            raise ValueError("The model must implement predict_proba.")

        max_probs = np.max(probas, axis=1)

        eps = 1e-12
        entropy = -np.sum(probas * np.log(probas + eps), axis=1)

        features = {
            "mean_conf": np.mean(max_probs),
            "std_conf": np.std(max_probs),
            "mean_entropy": np.mean(entropy),
            "std_entropy": np.std(entropy),
        }

        return np.array(list(features.values())).reshape(1, -1)
