# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from sklearn.metrics import accuracy_score

from skeval.base import BaseEvaluator
from skeval.utils import check_is_fitted

class ConfidenceThresholdEvaluator(BaseEvaluator):
    """
    Confidence-based evaluator for classification models.
    
    This evaluator filters the predictions of a classification model based on a
    confidence threshold. Only predictions with a confidence greater than or
    equal to the specified threshold are considered for scoring.

    Parameters
    ----------
    model : object
        Any model with `fit`, `predict`, and `predict_proba` or
        `decision_function` methods.
    scorer : callable or dict of str -> callable, default=accuracy_score
        An evaluation function or a dictionary of multiple evaluation functions.
    threshold : float, default=0.8
        The minimum confidence required to include a prediction in the calculation.
    limit_to_top_class : bool, default=True
        If True, uses only the probability of the top class as the confidence score.
    verbose : bool, default=False
        If True, prints additional information during evaluation.

    Attributes
    ----------
    model : object
        The model passed in the constructor.
    scorer : callable or dict
        The scoring function(s) passed in the constructor.
    threshold : float
        The confidence threshold.
    limit_to_top_class : bool
        Indicates if confidence is based solely on the highest probability class.
    verbose : bool
        The verbosity level.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.metrics import precision_score, recall_score
    >>> # Sample data
    >>> X_train = np.array([[1], [2], [3], [4], [5], [6]])
    >>> y_train = np.array([0, 0, 0, 1, 1, 1])
    >>> X_test = np.array([[0.5], [0.8], [3.5], [5.5]])
    >>> # 1. Create and train a standard classifier
    >>> classifier = LogisticRegression(solver='liblinear', random_state=0)
    >>> classifier.fit(X_train, y_train)
    LogisticRegression(random_state=0, solver='liblinear')
    >>> # 2. Create the evaluator with the trained classifier
    >>> scorings = {'precision': precision_score, 'recall': recall_score}
    >>> conf_eval = ConfidenceThresholdEvaluator(
    ...     model=classifier,
    ...     scorer=scorings,
    ...     threshold=0.9
    ... )
    >>> # 3. The fit method is for compatibility, not strictly needed here
    >>> #    since the model is already trained.
    >>> conf_eval.fit(X_train, y_train)
    ConfidenceThresholdEvaluator(...)
    >>> # 4. Estimate the performance on the test set
    >>> #    The evaluator will internally call classifier.predict_proba(X_test)
    >>> scores = conf_eval.estimate(X_test)
    >>> for score_name, value in scores.items():
    ...     print(f"{score_name}: {value:.2f}")
    precision: 1.00
    recall: 1.00
    """
    
    def __init__(self, model, scorer=accuracy_score, verbose=False, threshold=0.65, limit_to_top_class=True):
        super().__init__(model=model, scorer=scorer, verbose=verbose)
        
        self.threshold = threshold
        self.limit_to_top_class = limit_to_top_class

    def fit(self, X, y):
        """
        Fits the model to the training data.

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
        if self.verbose:
                print(f"[INFO] Model has been trained.")
        self.model.fit(X, y)
        return self

    def estimate(self, X_eval):
        """
        Estimates scores based on the confidence threshold.

        This method calculates the prediction confidences, filters out those
        that do not meet the threshold, and then computes the score(s)
        specified in the `scorer`.

        Parameters
        ----------
        X_eval : array-like of shape (n_samples, n_features)
            Input data for which to estimate scores.

        Returns
        -------
        dict
            A dictionary with estimated scores for each scorer.

            If no predictions pass the threshold, it returns 0.0 for each scorer.
        """
        check_is_fitted(self.model)
   
        conf, correct = self.__get_confidences_and_correct(X_eval)
        
        if self.verbose:
            print("[INFO] Confidences:", conf)
            print("[INFO] Passed threshold:", correct)

        if not np.any(correct):
            if self.verbose:
                print("[INFO] No predictions passed the threshold.")
            return {name: 0.0 for name in self._get_scorer_names()}

        y_pred = self.model.predict(X_eval)
        y_estimated = [y_pred[i] if c == 1 else (y_pred[i]+1)%2 for i, c in enumerate(correct)]
        y_estimated = [int(y) for y in y_estimated]

        if self.verbose:
            print("[INFO] y_pred:", y_pred)
            print("[INFO] y_estimated:", y_estimated)

        if isinstance(self.scorer, dict):
            scores = {
                name: func(y_estimated, y_pred)
                for name, func in self.scorer.items()
            }
            if self.verbose:
                print("[INFO] Estimated scores:", scores)
            return scores
        elif callable(self.scorer):
            score = self.scorer(y_estimated, y_pred)
            if self.verbose:
                print("[INFO] Estimated score:", score)
            return {'score': score}
        else:
            raise ValueError("'scorer' must be a callable or a dict of callables.")

    def __get_confidences_and_correct(self, X):
        """
        Computes confidence scores and applies the confidence threshold.

        This private method extracts prediction confidences from the model,
        either from `predict_proba` or `decision_function`. It then compares
        these confidences against the `threshold` to determine which predictions
        are considered to meet the confidence criteria.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        confidences : ndarray
            An array of confidence scores for each sample.
        correct : ndarray of bool
            A boolean array indicating `True` for samples where confidence
            is greater than or equal to the `threshold`.
        
        Raises
        ------
        ValueError
            If the model does not implement `predict_proba` or
            `decision_function` methods.
        """
        if not (hasattr(self.model, 'predict_proba') or hasattr(self.model, 'decision_function')):
            raise ValueError("The model must implement predict_proba or decision_function.")
        
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
            conf = np.max(probas, axis=1) if self.limit_to_top_class else probas
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X)
            conf = np.max(decision, axis=1) if decision.ndim > 1 else np.abs(decision)
        else:
            raise ValueError("The model must implement predict_proba or decision_function.")
        
        correct = conf >= self.threshold
        return conf, correct