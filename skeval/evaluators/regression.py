# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
import numpy as np

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from skeval.base import BaseEvaluator
from skeval.utils import check_is_fitted


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

    def __init__(
        self,
        model: Any,
        scorer: Union[
            Callable[..., float], Dict[str, Callable[..., float]]
        ] = accuracy_score,
        verbose: bool = False,
        meta_regressor: Optional[Any] = None,
    ) -> None:
        super().__init__(model=model, scorer=scorer, verbose=verbose)
        self.meta_regressor = meta_regressor or RandomForestRegressor(
            n_estimators=500, random_state=42
        )
        self.meta_regressors_: Dict[str, Any] = {}

    def fit(
        self, x: Sequence[Any], y: Sequence[Any], n_splits: int = 5
    ) -> "RegressionEvaluator":
        """Trains the internal meta-regressor(s) using a single model type.

        This method builds a meta-dataset to train the evaluator. For each dataset,
        it performs multiple random splits to increase the number of meta-examples.

        Parameters
        ----------
        x : list of array-like
            A list of datasets (features) used to train the meta-model.
        y : list of array-like
            A list of labels corresponding to `x`.
        n_splits : int, default=5
            Number of random splits per dataset to generate multiple meta-examples.

        Returns
        -------
        self : object
            The fitted evaluator instance.
        """
        scorers_names: List[str] = self._get_scorer_names()
        meta_targets: Dict[str, List[float]] = {name: [] for name in scorers_names}
        meta_features: List[np.ndarray] = []

        for x_i, y_i in zip(x, y):
            for split in range(n_splits):
                feats, y_holdout_meta, y_pred_holdout = self._generate_meta_example(
                    x_i=x_i, y_i=y_i, split=split
                )
                meta_features.append(feats)

                self._update_meta_targets(
                    y_true=y_holdout_meta,
                    y_pred=y_pred_holdout,
                    meta_targets=meta_targets,
                    scorers_names=scorers_names,
                )

        self.meta_regressors_ = {}

        for name in scorers_names:
            self.meta_regressors_[name] = self._fit_single_meta_regressor(
                name, np.array(meta_features), meta_targets
            )

            if self.verbose:
                print(f"[INFO] Meta-regressor for '{name}' has been trained.")

        self.model.fit(x[0], y[0])
        return self

    def estimate(self, x_eval: Any) -> Dict[str, float]:
        """Estimates the performance of the current model on unlabeled data.

        The model assigned to `self.model` must already be a fitted classifier
        (manually trained by the user). This method extracts meta-features from
        its predictions on the unlabeled data `x_eval` and uses the pre-trained
        meta-regressor(s) to predict the performance scores.

        Parameters
        ----------
        x_eval : array-like of shape (n_samples, n_features)
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
            raise RuntimeError(
                "The evaluator has not been fitted yet. Call 'fit' before 'estimate'."
            )

        feats = self._extract_metafeatures(estimator=self.model, x=x_eval)
        scores = {}

        for name, reg in self.meta_regressors_.items():
            estimated_score = reg.predict(feats)[0]
            scores[name] = estimated_score
            if self.verbose:
                print(f"[INFO] Estimated {name}: {estimated_score:.4f}")
        return scores

    def _extract_metafeatures(self, estimator: Any, x: Any) -> np.ndarray:
        """Extracts meta-features from a fitted model's predicted probabilities.

        The extracted features include the mean and standard deviation of the
        maximum prediction probabilities (confidence) and the mean and standard
        deviation of the entropy of the probability distributions.

        Parameters
        ----------
        estimator : fitted classifier
            The classifier from which to extract prediction probabilities.
        x : array-like of shape (n_samples, n_features)
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

        probas = estimator.predict_proba(x)
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

    def _generate_meta_example(
        self, x_i: Any, y_i: Any, split: int
    ) -> Tuple[np.ndarray, Any, np.ndarray]:
        """Generates a single meta-example from a dataset split."""
        est = clone(self.model)

        stratify_y = self._safe_stratify(y_i)
        x_train, x_hold, y_train, y_hold = train_test_split(
            x_i,
            y_i,
            test_size=0.33,
            random_state=42 + split,
            stratify=stratify_y,
        )
        est.fit(x_train, y_train)

        feats = self._extract_metafeatures(estimator=est, x=x_hold)
        y_pred = est.predict(x_hold)

        return feats.flatten(), y_hold, y_pred

    def _fit_single_meta_regressor(
        self, name: str, meta_features: np.ndarray, meta_targets: Dict[str, List[float]]
    ) -> Any:
        """Fits a single meta-regressor for a given scorer."""
        meta_y = np.array(meta_targets[name])
        reg = clone(self.meta_regressor)
        reg.fit(meta_features, meta_y)
        return reg

    def _update_meta_targets(
        self,
        y_true: Any,
        y_pred: Any,
        meta_targets: Dict[str, List[float]],
        scorers_names: List[str],
    ) -> None:
        """Updates the meta-targets dictionary with new scores."""
        if isinstance(self.scorer, dict):
            for name in scorers_names:
                # scorer[name] is callable
                meta_targets[name].append(float(self.scorer[name](y_true, y_pred)))
        else:
            scorer_fn = cast(Callable[..., float], self.scorer)
            meta_targets["score"].append(float(scorer_fn(y_true, y_pred)))

    def _safe_stratify(self, y: Optional[Any]) -> Optional[Any]:
        if y is None:
            return None
        y = np.asarray(y)

        uniques, counts = np.unique(y, return_counts=True)
        if len(uniques) < 2:
            return None
        if np.min(counts) < 2:
            return None

        return y
