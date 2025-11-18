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
    X_train : array-like, optional
        Training features used to compute SHAP values.
    y_train : array-like, optional
        Training labels used to compute SHAP values.

    Attributes
    ----------
    model : estimator
        The model provided at initialization.
    inner_clf : estimator
        The classifier used to model correctness from SHAP values.
    explainer : shap.TreeExplainer
        Object responsible for computing SHAP values.
    X_train : ndarray of shape (n_samples, n_features)
        Feature matrix used to compute SHAP values for correctness learning.
    y_train : ndarray of shape (n_samples,)
        Ground-truth labels corresponding to ``X_train``.

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
    >>> X_train, y_train = df_train.drop(columns=["label"]), df_train["label"]
    >>> X_test = df_test.drop(columns=["label"])
    >>> model = XGBClassifier()
    >>>
    >>> evaluator = ShapEvaluator(model, scorer=accuracy_score)
    >>> evaluator.fit(X_train, y_train)
    >>> scores = evaluator.estimate(X_test)
    >>> print(scores)
    0.84
    """

    def __init__(self, model, scorer=accuracy_score, verbose=False, inner_clf=None, X_train=None, y_train=None):
        super().__init__(model=model, scorer=scorer, verbose=verbose)
        self.inner_clf = inner_clf or XGBClassifier(random_state=42)
        self.explainer = None
        self.X_train, self.y_train = X_train, y_train
        
    def fit(self, X=None, y=None):
        """
        Fit the model used by the evaluator.

        If `X_train` and `y_train` were provided during initialization, they
        take precedence. Otherwise, the provided `X` and `y` are stored and
        used for computing SHAP values during the estimation step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Feature matrix used to fit the original model and to compute SHAP
            values if `X_train` is not already defined.
        y : array-like of shape (n_samples,), optional
            Labels corresponding to `X`. Required if no training data was
            provided at initialization.

        Returns
        -------
        self : ShapEvaluator
            The fitted evaluator instance.

        Raises
        ------
        ValueError
            If no training data is available (neither from initialization nor
            from arguments), preventing the underlying model from being fitted.
        """
        
        self.X_train = X if X is not None else self.X_train
        self.y_train = y if y is not None else self.y_train
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("X and y must be provided to fit the model.")
            
        self.model.fit(self.X_train, self.y_train)
        return self

    def estimate(self, X_eval, n_pred=30):
        """
        Estimate metric values using SHAP-based correctness prediction.

        SHAP values are computed for train and eval sets, used to train a correctness
        classifier, and the resulting correctness predictions decide when to flip
        the model predictions before scoring. Results are averaged over `n_pred`
        iterations.

        Parameters
        ----------
        X_eval : array-like
        n_pred : int

        Returns
        -------
        dict
            Mapping me
        """
        check_is_fitted(self.model)
        if self.X_train is None or self.y_train is None:
            raise ValueError("X_train and y_train must be provided to compute SHAP values.")

        shap_train_arr, shap_eval_arr = self._compute_shap_arrays(X_eval)

        pred_eval = self.model.predict(X_eval)
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
        except Exception:
            idx = -1

        if isinstance(shap_vals, list):
            return np.array(shap_vals[idx])

        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            return shap_vals[:, :, idx]

        return np.array(shap_vals)
    
    def _compute_shap_arrays(self, X_eval):
        model_to_explain = self.model[-1] if hasattr(self.model, "steps") else self.model
        self.explainer = shap.TreeExplainer(model_to_explain)

        shap_train = self.explainer.shap_values(self.X_train)
        shap_eval = self.explainer.shap_values(X_eval)

        return (
            self._choose_class_shap(shap_train, self.model),
            self._choose_class_shap(shap_eval, self.model),
        )

    def _train_correctness_model(self, shap_train_arr, y_right):
        clf = clone(self.inner_clf)
        clf.fit(shap_train_arr, y_right)
        return clf
    
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



