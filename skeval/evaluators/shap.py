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
    classifier’s predictions. The predicted correctness on the evaluation set
    is used to generate an expected label vector, which is then compared with
    the model predictions to estimate the chosen metric(s).

    The evaluation process follows four steps:
    (1) compute SHAP values on the training and evaluation sets,
    (2) train a correctness classifier using SHAP values as input,
    (3) predict correctness for evaluation samples,
    (4) flip labels where the model is predicted to be wrong, generating an
        “expected label” vector used to estimate metrics.

    Parameters
    ----------
    model : estimator
        A classification model implementing `fit` and `predict`. For SHAP
        computation using `TreeExplainer`. Compatible with sklearn.make_pipeline
    scorer : callable or dict of str -> callable, default=accuracy_score
        A scoring function or a dictionary mapping metric names to scoring
        functions. Scorers must follow the signature `scorer(y_true, y_pred)`.
    verbose : bool, default=False
        If True, prints additional progress information.
    inner_clf : estimator, optional
        Classifier trained on SHAP values to estimate correctness. If None,
        defaults to an `XGBClassifier(random_state=42)`.
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
        Ground-truth labels corresponding to `X_train`.

    Notes
    -----
    *SHAP computation requirement:*  
    The final estimator in `model` (or the estimator itself, if not a
    pipeline) must be compatible with `shap.TreeExplainer`.

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
    >>> df_test  = pd.read_csv("dataset_test.csv")
    >>> X_train,  = df_train.drop(columns=["label"]), df["label"]
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
        self.X_train = X_train
        self.y_train = y_train
        
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
        
        if (self.X_train is None and X is not None):
            self.X_train = X

        if (self.y_train is None and y is not None):
            self.y_train = y
            
        if self.X_train is None or self.y_train is None:
            raise ValueError("X and y must be provided to fit the model.")
            
        self.model.fit(X, y)
        return self

    def estimate(self, X_eval, n_pred=30):
        """
        Estimate metric values using SHAP-based correctness prediction.

        This method computes SHAP values for the training set and the evaluation
        set, trains a correctness classifier on the training SHAP values, and
        predicts correctness for evaluation samples. Labels predicted as
        incorrect are flipped to produce an expected label vector, which is
        compared against the model's predictions to estimate the target metrics.

        Multiple iterations are performed to model uncertainty, and the final
        metric is obtained by averaging scores across all iterations.

        Parameters
        ----------
        X_eval : array-like of shape (n_eval_samples, n_features)
            Evaluation feature matrix used to generate predictions and SHAP
            values.
        n_pred : int, default=30
            Number of estimation iterations. In each iteration, a correctness
            classifier is retrained and used to produce an estimated metric.

        Returns
        -------
        dict
            A dictionary mapping metric names to averaged estimated values.
            If a single scorer was provided, the dictionary contains a single
            entry with key ``'score'``.

        Raises
        ------
        ValueError
            If training data (`X_train`, `y_train`) is missing, preventing SHAP
            computation.
        ValueError
            If the provided scorer is neither a callable nor a dictionary of
            callables.
        """

        check_is_fitted(self.model)
        if self.X_train is None or self.y_train is None:
            raise ValueError("X_train and y_train must be provided to compute SHAP values for the training set.")

        model_to_explain = self.model[-1] if hasattr(self.model, "steps") else self.model
        self.explainer = shap.TreeExplainer(model_to_explain)

        X_train_proc = self.X_train
        X_eval_proc = X_eval

        shap_train = self.explainer.shap_values(X_train_proc)
        shap_eval = self.explainer.shap_values(X_eval_proc)

        shap_train_arr = self._choose_class_shap(shap_train, self.model)
        shap_eval_arr = self._choose_class_shap(shap_eval, self.model)
        
        scores_list = []

        for _ in range(n_pred):
            pred_train = np.random.randint(0, 2, size=len(self.y_train))
            pred_eval = self.model.predict(X_eval)

            y_right = np.array([1 if self.y_train[i] == pred_train[i] else 0 for i in range(len(pred_train))])

            clf = clone(self.inner_clf)
            clf.fit(shap_train_arr, y_right)

            pred_right = clf.predict(shap_eval_arr)

            expected_y = np.array([
                pred_eval[i] if pred_right[i] == 1 else ((pred_eval[i] + 1) % 2)
                for i in range(len(pred_eval))
            ])

            if isinstance(self.scorer, dict):
                iter_scores = {name: fn(expected_y, pred_eval) for name, fn in self.scorer.items()}
            elif callable(self.scorer):
                iter_scores = {'score': self.scorer(expected_y, pred_eval)}
            else:
                raise ValueError("'scorer' must be a callable or a dict of callables.")

            scores_list.append(iter_scores)

        if isinstance(self.scorer, dict):
            avg_scores = {name: np.mean([s[name] for s in scores_list]) for name in self.scorer.keys()}
        else:
            avg_scores = {'score': np.mean([s['score'] for s in scores_list])}

        return avg_scores

    def _choose_class_shap(self, shap_vals, model):
        """
        Select the SHAP values corresponding to the positive class.

        SHAP outputs differ depending on the model type:
        - For binary classifiers, SHAP may return a list where each entry
          corresponds to class-specific SHAP values.
        - Some tree-based models return arrays of shape
          (n_samples, n_features, n_classes).

        This helper method extracts the SHAP values associated with the class
        labeled ``1`` when possible. If class ``1`` is not found, the last class
        available is used as fallback.

        Parameters
        ----------
        shap_vals : list or ndarray
            SHAP values produced by `shap.TreeExplainer`. May be a list of
            per-class arrays or a 3D array with class dimension.
        model : estimator
            The original classification model, used to determine class ordering.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            SHAP value matrix associated with the selected class.

        Notes
        -----
        This function does not validate whether the model is binary or
        multiclass. The selection logic is heuristic and assumes that class
        ``1`` corresponds to the positive or target class.
        """
        if isinstance(shap_vals, list):
            try:
                idx = list(model.classes_).index(1)
            except Exception:
                idx = -1
            return np.array(shap_vals[idx])


        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            try:
                idx = list(model.classes_).index(1)
            except Exception:
                idx = -1
            return shap_vals[:, :, idx]
        return np.array(shap_vals)