# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, Callable, Dict, List, Mapping, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score

from skeval.base import BaseEvaluator
from skeval.utils import check_is_fitted


class ConfidenceThresholdEvaluator(BaseEvaluator):
    """Confidence-based evaluator for classification models.

    This evaluator filters predictions from a classification model according to
    a confidence threshold. Only predictions whose confidence (top-class
    probability, or other chosen score) is greater than or equal to the given
    threshold are treated as "trusted"; the remaining predictions are flipped
    (binary case) to build an expected label vector used for metric estimation.

    Parameters
    ----------
    model : object
        Any classifier implementing ``fit``, ``predict`` and either
        ``predict_proba`` or ``decision_function``.
    scorer : callable or dict of str -> callable, default=accuracy_score
        Single scoring function or mapping of metric names to callables with
        signature ``scorer(y_true, y_pred)``.
    verbose : bool, default=False
        If ``True``, prints intermediate information during fitting and
        estimation.

    Attributes
    ----------
    model : object
        The primary model evaluated.
    scorer : callable or dict
        Scoring function(s) applied to agreement-based labels.
    verbose : bool
        Verbosity flag.

    Examples
    --------
    Example using medical datasets and a RandomForest pipeline:

    >>> import pandas as pd
    >>> from sklearn.metrics import accuracy_score, f1_score
    >>> from sklearn.impute import KNNImputer
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from skeval.evaluators.confidence import ConfidenceThresholdEvaluator
    >>> from skeval.utils import get_cv_and_real_scores, print_comparison
    >>> # 1. Load datasets
    >>> df_geriatrics = pd.read_csv("geriatria.csv")
    >>> df_neurology = pd.read_csv("neurologia.csv")
    >>> # 2. Separate features and target
    >>> X1, y1 = df_geriatrics.drop(columns=["Alzheimer"]), df_geriatrics["Alzheimer"]
    >>> X2, y2 = df_neurology.drop(columns=["Alzheimer"]), df_neurology["Alzheimer"]
    >>> # 3. Define model pipeline
    >>> model = make_pipeline(
    ...     KNNImputer(n_neighbors=4),
    ...     RandomForestClassifier(n_estimators=300, random_state=42),
    ... )
    >>> # 4. Initialize evaluator with scorers
    >>> scorers = {
    ...     "accuracy": accuracy_score,
    ...     "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
    ... }
    >>> evaluator = ConfidenceThresholdEvaluator(model=model, scorer=scorers)
    >>> # 5. Fit evaluator
    >>> evaluator.fit(X1, y1)
    >>> # 6. Estimated performance (using confidence threshold)
    >>> estimated_scores = evaluator.estimate(X2, threshold=0.65, limit_to_top_class=True)
    >>> # 7. Cross-validation and real performance comparison
    >>> scores_dict = get_cv_and_real_scores(
    ...     model=model, scorers=scorers, train_data=(X1, y1), test_data=(X2, y2)
    ... )
    >>> cv_scores = scores_dict["cv_scores"]
    >>> real_scores = scores_dict["real_scores"]
    >>> print_comparison(scorers, cv_scores, estimated_scores, real_scores)
    """

    def __init__(
        self,
        model: Any,
        scorer: Union[
            Callable[..., Any], Mapping[str, Callable[..., Any]]
        ] = accuracy_score,
        verbose: bool = False,
    ) -> None:
        super().__init__(model=model, scorer=scorer, verbose=verbose)

    def fit(self, x: Any, y: Any) -> "ConfidenceThresholdEvaluator":
        """
        Fits the model to the training data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target labels.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if self.verbose:
            print("[INFO] Model has been trained.")
        self.model.fit(x, y)
        return self

    def estimate(
        self, x_eval: Any, threshold: float = 0.65, limit_to_top_class: bool = True
    ) -> Dict[str, float]:
        """
        Estimates scores based on the confidence threshold.

        This method calculates the prediction confidences, filters out those
        that do not meet the threshold, and then computes the score(s)
        specified in the `scorer`.

        Parameters
        ----------
        x_eval : array-like of shape (n_samples, n_features)
            Input data for which to estimate scores.
        threshold : float, default=0.8
            The minimum confidence required to include a prediction in the calculation.
        limit_to_top_class : bool, default=True
            If True, uses only the probability of the top class as the confidence score.

        Returns
        -------
        dict
            A dictionary with estimated scores for each scorer.

            If no predictions pass the threshold, it returns 0.0 for each scorer.
        """
        check_is_fitted(self.model)

        conf, correct = self.__get_confidences_and_correct(
            x_eval, threshold, limit_to_top_class
        )
        self._print_verbose_confidence_info(conf, correct)

        if not np.any(correct):
            return self._handle_no_confident_predictions()

        y_pred = self.model.predict(x_eval)
        y_estimated = self._build_estimated_labels(y_pred, correct)

        self._print_verbose_label_info(y_pred, y_estimated)

        return self._compute_scores(y_estimated, y_pred)

    def _print_verbose_confidence_info(
        self, conf: np.ndarray, correct: np.ndarray
    ) -> None:
        if self.verbose:
            print("[INFO] Confidences:", conf)
            print("[INFO] Passed threshold:", correct)

    def _handle_no_confident_predictions(self) -> Dict[str, float]:
        if self.verbose:
            print("[INFO] No predictions passed the threshold.")
        return {name: 0.0 for name in self._get_scorer_names()}

    def _build_estimated_labels(self, y_pred: Any, correct: np.ndarray) -> List[int]:
        y_estimated = [
            y_pred[i] if c == 1 else (y_pred[i] + 1) % 2 for i, c in enumerate(correct)
        ]
        return [int(y) for y in y_estimated]

    def _print_verbose_label_info(self, y_pred: Any, y_estimated: List[int]) -> None:
        if self.verbose:
            print("[INFO] y_pred:", y_pred)
            print("[INFO] y_estimated:", y_estimated)

    def _compute_scores(self, y_estimated: List[int], y_pred: Any) -> Dict[str, float]:
        if isinstance(self.scorer, dict):
            scores: Dict[str, float] = {
                name: float(func(y_estimated, y_pred))
                for name, func in self.scorer.items()
            }
            if self.verbose:
                print("[INFO] Estimated scores:", scores)
            return scores

        if callable(self.scorer):
            score_val = float(self.scorer(y_estimated, y_pred))
            if self.verbose:
                print("[INFO] Estimated score:", score_val)
            return {"score": score_val}
        raise ValueError("'scorer' must be a callable or a dict of callables.")

    def __get_confidences_and_correct(
        self, x: Any, threshold: float, limit_to_top_class: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes confidence scores and applies the confidence threshold.
        """
        if not (
            hasattr(self.model, "predict_proba")
            or hasattr(self.model, "decision_function")
        ):
            raise ValueError(
                "The model must implement predict_proba or decision_function."
            )

        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(x)
            conf: np.ndarray = (
                np.max(probas, axis=1) if limit_to_top_class else np.asarray(probas)
            )
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(x)
            conf = (
                np.max(decision, axis=1)
                if getattr(decision, "ndim", 1) > 1
                else np.abs(np.asarray(decision))
            )
        else:
            raise ValueError(
                "The model must implement predict_proba or decision_function."
            )

        correct = conf >= threshold
        return conf, correct
