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

    def fit(self, x, y, n_splits=5, noise_cfg=None):
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
        noise_cfg : dict, default=None
            Configuration for label noise levels:
            - 'start': int, starting percentage of label noise (inclusive).
            - 'end': int, ending percentage of label noise (inclusive).
            - 'step': int, step size for noise percentage increments.

        Returns
        -------
        self : object
            The fitted evaluator instance.
        """
        if noise_cfg is None:
            noise_cfg = {"start": 10, "end": 100, "step": 10}

        self._validate_noise_params(noise_cfg)

        scorer_names = self._get_scorer_names()

        meta_cfg = {
            "features": [],
            "targets": {name: [] for name in scorer_names},
            "scorer_names": scorer_names,
        }

        for x_i, y_i in zip(x, y):
            self._process_single_dataset((x_i, y_i), noise_cfg, meta_cfg, n_splits)

        self._train_meta_regressors(meta_cfg)

        # Base model fit (lógica original preservada)
        self.model.fit(x[0], y[0])
        return self

    def _validate_noise_params(self, noise_cfg):
        has_all_keys = all(key in noise_cfg for key in ["start", "end", "step"])

        if (
            not has_all_keys
            or noise_cfg["start"] < 0
            or noise_cfg["end"] > 100
            or noise_cfg["step"] <= 0
        ):
            raise ValueError(
                "Noise parameters must satisfy: "
                "0 <= start <= end <= 100 and step > 0."
            )

    def _process_single_dataset(self, train_data, noise_cfg, meta_cfg, n_splits):
        """Processes one dataset by generating meta-examples."""
        y_i = train_data[1]
        unique_labels = np.unique(y_i)
        stratify_y = y_i if len(unique_labels) > 1 else None

        for split in range(n_splits):
            self._process_single_split(
                train_data, stratify_y, split, (noise_cfg, meta_cfg)
            )

    def _process_single_split(self, train_data, stratify_y, split, cfg):
        """Processes each train/holdout split."""
        base_model = clone(self.model)
        noise_cfg, meta_cfg = cfg

        x_train, x_holdout, y_train, y_holdout = train_test_split(
            train_data[0],
            train_data[1],
            test_size=0.33,
            random_state=42 + split,
            stratify=stratify_y,
        )

        base_model.fit(x_train, y_train)
        metafeats = self._extract_metafeatures(base_model, x_holdout)

        for noise_p in range(
            noise_cfg["start"], noise_cfg["end"] + 1, noise_cfg["step"]
        ):
            generation_cfg = {
                "features": meta_cfg["features"],
                "targets": meta_cfg["targets"],
                "scorer_names": meta_cfg["scorer_names"],
                "metafeats": metafeats,
                "noise_p": noise_p,
            }
            self._generate_noise_meta_example(
                base_model, (x_holdout, y_holdout), split, generation_cfg
            )

    def _generate_noise_meta_example(self, base_model, holdout, split, cfg):
        """Adds one meta-example (metafeatures + performance target)."""
        x_holdout, y_holdout = holdout
        n_noisy = int(len(x_holdout) * (cfg["noise_p"] / 100.0))

        x_noisy = x_holdout[:n_noisy].copy()
        x_clean = x_holdout[n_noisy:].copy()

        rng = np.random.default_rng(42 + cfg["noise_p"] + split)

        for col in x_holdout.columns:
            x_noisy[col] = rng.permutation(x_noisy[col])

        x_concat = pd.concat([x_noisy, x_clean], axis=0)
        y_pred = base_model.predict(x_concat)

        cfg["features"].append(cfg["metafeats"].flatten())
        self._store_meta_targets(cfg["targets"], cfg["scorer_names"], y_holdout, y_pred)

    def _store_meta_targets(self, meta_targets, scorer_names, y_true, y_pred):
        """Stores score values for one meta-example."""

        if isinstance(self.scorer, dict):
            for name in scorer_names:
                meta_targets[name].append(self.scorer[name](y_true, y_pred))
        else:
            meta_targets["score"].append(self.scorer(y_true, y_pred))

    def _train_meta_regressors(self, meta_cfg):
        """Trains one regressor per scorer."""
        meta_features = np.array(meta_cfg["features"])
        self.meta_regressors_ = {}

        for name in meta_cfg["scorer_names"]:
            y_arr = np.array(meta_cfg["targets"][name])
            reg = clone(self.meta_regressor)
            reg.fit(meta_features, y_arr)
            self.meta_regressors_[name] = reg

            if self.verbose:
                print(f"[INFO] Meta-regressor for '{name}' has been trained.")
