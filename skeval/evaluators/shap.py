# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
import shap
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from xgboost import XGBClassifier


from skeval.base import BaseEvaluator
from skeval.utils import check_is_fitted


class ShapEvaluator(BaseEvaluator):
    """
    SHAP-based evaluator for supervised classification models.

    This evaluator uses SHAP values computed from a tree-based classifier to
    train a secondary model that predicts the correctness of the original
    classifier's predictions. The predicted correctness on the evaluation set
    is used to generate an expected label vector, which is then compared with
    the model predictions to estimate the chosen metric(s).

    The evaluation process follows four steps:
    (1) compute SHAP values on the training and evaluation sets,
    (2) train a correctness classifier using SHAP values as input,
    (3) predict correctness for evaluation samples,
    (4) flip labels where the model is predicted to be wrong, generating an
    "expected label" vector used to estimate metrics.

    Parameters
    ----------
    model : estimator
        A classification model implementing ``fit`` and ``predict``. For SHAP
        computation using ``TreeExplainer``. Compatible with ``sklearn.make_pipeline``.
    scorer : callable or dict of str -> callable, default=accuracy_score
        A scoring function or a dictionary mapping metric names to scoring
        functions. Scorers must follow the signature ``scorer(y_true, y_pred)``.
    verbose : bool, default=False
        If True, prints additional progress information.
    inner_clf : estimator, optional
        Classifier trained on SHAP values to estimate correctness. If None,
        defaults to ``XGBClassifier(random_state=42)``.

    Attributes
    ----------
    model : estimator
        The model provided at initialization.
    inner_clf : estimator
        The classifier used to model correctness from SHAP values.
    explainer : shap.TreeExplainer
        Object responsible for computing SHAP values.

    Notes
    -----
    *SHAP computation requirement:*
    The final estimator in ``model`` (or the estimator itself, if not a
    pipeline) must be compatible with ``shap.TreeExplainer``.

    *Estimate method:*
    The method performs multiple correctness predictions and averages the
    resulting estimated metrics. This introduces stochasticity and aims to
    approximate uncertainty in the correctness model.

    Examples
    --------
    >>> from skeval.evaluators import ShapEvaluator
    >>> from sklearn.metrics import accuracy_score
    >>> from xgboost import XGBClassifier
    >>> import pandas as pd
    >>>
    >>> df_train = pd.read_csv("dataset_train.csv")
    >>> df_test = pd.read_csv("dataset_test.csv")
    >>> x_train, y_train = df_train.drop(columns=["label"]), df_train["label"]
    >>> x_test = df_test.drop(columns=["label"])
    >>> model = XGBClassifier()
    >>>
    >>> evaluator = ShapEvaluator(model, scorer=accuracy_score)
    >>> evaluator.fit(x_train, y_train)
    >>> scores = evaluator.estimate(x_test)
    >>> print(scores)
    0.84
    """

    def __init__(
        self,
        model,
        scorer=accuracy_score,
        verbose=False,
        inner_clf=None,
    ):
        super().__init__(model=model, scorer=scorer, verbose=verbose)
        self.inner_clf = inner_clf or XGBClassifier(random_state=42)
        self.explainer = None
        self.x_train, self.y_train = None, None

    def fit(self, x=None, y=None):
        """
        Fit the model used by the evaluator.

        If `x_train` and `y_train` were provided during initialization, they
        take precedence. Otherwise, the provided `x` and `y` are stored and
        used for computing SHAP values during the estimation step.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Feature matrix used to fit the original model and to compute SHAP
            values if `x_train` is not already defined.
        y : array-like of shape (n_samples,)
            Labels corresponding to `x`. Required if no training data was
            provided at initialization.

        Returns
        -------
        self : ShapEvaluator
            The fitted evaluator instance.

        Raises
        ------
        ValueError
            If no training data is available, preventing the underlying model from being fitted.
        """

        # self.x_train = x if x is not None else self.x_train
        # self.y_train = y if y is not None else self.y_train

        if x is None or y is None:
            raise ValueError("x and y must be provided to fit the model.")
        self.x_train = x
        self.y_train = y

        self.model.fit(self.x_train, self.y_train)
        return self

    def estimate(self, x_eval, n_pred=30, train_data=None):
        """
        Estimate metric values using SHAP-based correctness prediction.

        SHAP values are computed for train and eval sets, used to train a correctness
        classifier, and the resulting correctness predictions decide when to flip
        the model predictions before scoring. Results are averaged over `n_pred`
        iterations.

        Parameters
        ----------
        x_eval : array-like
            Feature matrix for evaluation.
        n_pred : int
            Number of correctness predictions to average over.
        train_data : tuple of (array-like, array-like), optional
            Training data (X, y) used to fit the original model and compute SHAP
            values, if not already provided during `fit()`.

        Returns
        -------
        dict
            Mapping me
        """
        check_is_fitted(self.model)
        if self.x_train is None or self.y_train is None:
            if train_data is None:
                raise ValueError("Train data must be provided to compute SHAP values.")
            self.x_train, self.y_train = train_data

        shap_train_arr, shap_eval_arr = self._compute_shap_arrays(x_eval)

        pred_eval = self.model.predict(x_eval)
        scores_list = []

        for _ in range(n_pred):
            pred_train = np.random.randint(0, 2, size=len(self.y_train))
            y_right = (self.y_train == pred_train).astype(int)

            clf = self._train_correctness_model(shap_train_arr, y_right)
            pred_right = clf.predict(shap_eval_arr)

            expected_y = self._predict_expected_labels(pred_eval, pred_right)
            scores_list.append(self._evaluate_scores(expected_y, pred_eval))

        if isinstance(self.scorer, dict):
            return {k: np.mean([s[k] for s in scores_list]) for k in self.scorer}
        return {"score": np.mean([s["score"] for s in scores_list])}

    def _choose_class_shap(self, shap_vals, model):
        """
        Returns SHAP values for class 1 (or fallback class). Supports list and 3D-array formats.
        """
        try:
            idx = list(model.classes_).index(1)
        except ValueError:
            idx = -1

        if isinstance(shap_vals, list):
            return np.array(shap_vals[idx])

        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            return shap_vals[:, :, idx]

        return np.array(shap_vals)

    def _compute_shap_arrays(self, x_eval):
        model_to_explain = (
            self.model[-1] if hasattr(self.model, "steps") else self.model
        )
        self.explainer = shap.TreeExplainer(model_to_explain)

        shap_train = self.explainer.shap_values(self.x_train)
        shap_eval = self.explainer.shap_values(x_eval)

        return (
            self._choose_class_shap(shap_train, self.model),
            self._choose_class_shap(shap_eval, self.model),
        )

    def _train_correctness_model(self, shap_train_arr, y_right):
        clf = clone(self.inner_clf)
        clf.fit(shap_train_arr, y_right)
        return clf

    def _predict_expected_labels(self, pred_eval, pred_right):
        return pred_eval ^ (1 - pred_right)

    def _evaluate_scores(self, expected_y, pred_eval):
        if isinstance(self.scorer, dict):
            return {name: fn(expected_y, pred_eval) for name, fn in self.scorer.items()}
        return {"score": self.scorer(expected_y, pred_eval)}
