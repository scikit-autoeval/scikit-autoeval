# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from skeval.evaluators.regression import RegressionEvaluator

class RegressionNoiseEvaluator(RegressionEvaluator):
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
    >>> from skeval.evaluators.regression import RegressionBasedEvaluator
    >>>
    >>> iris = load_iris()
    >>> X_list, y_list = [iris.data], [iris.target]
    >>> evaluator = RegressionBasedEvaluator(
    ...     model=LogisticRegression(max_iter=1000),
    ...     scorer=accuracy_score,
    ...     verbose=False
    ... )
    >>> evaluator.fit(X_list, y_list)
    RegressionBasedEvaluator(...)
    >>> # Train a final model manually
    >>> final_model = LogisticRegression(max_iter=1000).fit(iris.data, iris.target)
    >>> evaluator.model = final_model
    >>> estimated_scores = evaluator.estimate(iris.data)
    >>> print(estimated_scores)
    {'score': ...}
    """
    
    def fit(self, X, y, start_noise=10, end_noise=100, step_noise=10):
        """Trains the internal meta-regressor(s) using a single model type.

        This method builds a meta-dataset to train the evaluator. For each dataset,
        it performs multiple random splits to increase the number of meta-examples.

        Parameters
        ----------
        X : list of array-like
            A list of datasets (features) used to train the meta-model.
        y : list of array-like
            A list of labels corresponding to `X`.
        start_noise : int, default=10
            The starting percentage of label noise to introduce in the holdout set.
        end_noise : int, default=100
            The ending percentage of label noise to introduce in the holdout set.
        step_noise : int, default=10
            The step size for increasing the percentage of label noise.

        Returns
        -------
        self : object
            The fitted evaluator instance.
        """
        self._validate_noise_params(start_noise, end_noise, step_noise)

        scorer_names = self._get_scorer_names()
        meta_features = []
        meta_targets = {name: [] for name in scorer_names}

        for X_i, y_i in zip(X, y):
            self._process_single_dataset(
                X_i, y_i, start_noise, end_noise, step_noise,
                meta_features, meta_targets, scorer_names
            )

        self._train_meta_regressors(meta_features, meta_targets, scorer_names)

        # Base model fit (lógica original preservada)
        self.model.fit(X[0], y[0])
        return self
    
    def _validate_noise_params(self, start_noise, end_noise, step_noise):
        if start_noise < 0 or end_noise > 100 or step_noise <= 0:
            raise ValueError(
                "Noise parameters must satisfy: "
                "0 <= start_noise <= end_noise <= 100 and step_noise > 0."
            )

    def _process_single_dataset(
        self, X_i, y_i, start_noise, end_noise, step_noise,
        meta_features, meta_targets, scorer_names
    ):
        """Processes one dataset by generating meta-examples."""

        unique_labels = np.unique(y_i)
        stratify_y = y_i if len(unique_labels) > 1 else None

        for split in range(self.n_splits):
            self._process_single_split(
                X_i, y_i, stratify_y, split,
                start_noise, end_noise, step_noise,
                meta_features, meta_targets, scorer_names
            )

    def _process_single_split(
        self, X_i, y_i, stratify_y, split,
        start_noise, end_noise, step_noise,
        meta_features, meta_targets, scorer_names
    ):
        """Processes each train/holdout split."""
        base_model = clone(self.model)

        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X_i, y_i,
            test_size=0.33,
            random_state=42 + split,
            stratify=stratify_y
        )

        base_model.fit(X_train, y_train)
        metafeats = self._extract_metafeatures(base_model, X_holdout)

        for noise_p in range(start_noise, end_noise + 1, step_noise):
            self._generate_meta_example(
                base_model, X_holdout, y_holdout,
                metafeats, noise_p, split,
                meta_features, meta_targets, scorer_names
            )

    def _generate_meta_example(
        self, base_model, X_holdout, y_holdout, metafeats,
        noise_p, split, meta_features, meta_targets, scorer_names
    ):
        """Adds one meta-example (metafeatures + performance target)."""

        n_noisy = int(len(X_holdout) * (noise_p / 100.0))

        X_noisy = X_holdout[:n_noisy].copy()
        X_clean = X_holdout[n_noisy:].copy()

        rng = np.random.default_rng(42 + noise_p + split)

        for col in X_holdout.columns:
            X_noisy[col] = rng.permutation(X_noisy[col])

        X_concat = pd.concat([X_noisy, X_clean], axis=0)
        y_pred = base_model.predict(X_concat)

        meta_features.append(metafeats.flatten())
        self._store_meta_targets(meta_targets, scorer_names, y_holdout, y_pred)

    def _store_meta_targets(self, meta_targets, scorer_names, y_true, y_pred):
        """Stores score values for one meta-example."""

        if isinstance(self.scorer, dict):
            for name in scorer_names:
                meta_targets[name].append(self.scorer[name](y_true, y_pred))
        else:
            meta_targets["score"].append(self.scorer(y_true, y_pred))

    def _train_meta_regressors(self, meta_features, meta_targets, scorer_names):
        """Trains one regressor per scorer."""
        meta_features = np.array(meta_features)
        self.meta_regressors_ = {}

        for name in scorer_names:
            y_arr = np.array(meta_targets[name])
            reg = clone(self.meta_regressor)
            reg.fit(meta_features, y_arr)
            self.meta_regressors_[name] = reg

            if self.verbose:
                print(f"[INFO] Meta-regressor for '{name}' has been trained.")