# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone

from skeval.base import BaseEvaluator
from skeval.utils import check_is_fitted

class ShapEvaluator(BaseEvaluator):
    """
    SHAP-based evaluator.

    The evaluator uses SHAP values extracted from a (tree) classifier to train
    a secondary classifier that predicts whether the original classifier's
    predictions are correct. The predicted "correctness" on the evaluation set
    is then used to build an expected label vector; that vector is compared
    against the original model predictions to produce estimated metrics.

    Parameters
    ----------
    model : estimator
        A trained (or to-be-trained) classifier. For SHAP TreeExplainer a
        tree-based model (e.g. RandomForest, GradientBoosting) is expected.
    scorer : callable or dict, default=accuracy_score
        Single scoring function or dictionary of name->callable to compute
        estimated metrics.
    verbose : bool, default=False
        Enable informational prints.
    inner_clf : estimator, optional
        Classifier used to learn "correctness" from SHAP values. If None a
        RandomForestClassifier is used.

    Notes
    -----
    estimate requires both a training set (X_train, y_train) and an
    evaluation set (X_eval, optionally y_eval for debugging). The method
    signature follows estimate(X_eval, X_train=None, y_train=None, y_eval=None).
    """

    def __init__(self, model, scorer=accuracy_score, verbose=False, inner_clf=None):
        super().__init__(model=model, scorer=scorer, verbose=verbose)
        self.inner_clf = inner_clf or RandomForestClassifier(n_estimators=50, random_state=42)

    def fit(self, X, y):
        """
        Fit underlying model if required (keeps compatibility with other evaluators).

        Parameters
        ----------
        X : array-like
            Feature matrix to fit the underlying model.
        y : array-like
            Target vector.

        Returns
        -------
        self
        """
        if self.verbose:
            print("[INFO] Fitting underlying model")
        self.model.fit(X, y)
        return self

    def estimate(self, X_eval, X_train=None, y_train=None):
        """
        Estimate metrics using SHAP-based correctness prediction.

        Parameters
        ----------
        X_eval : array-like
            The evaluation features (used as X_test in the pseudo-code).
        X_train : array-like
            The training features used to compute SHAP on train (required).
        y_train : array-like
            True labels for X_train (required).
        y_eval : array-like, optional
            True labels for X_eval (optional, used only for debugging/validation).

        Returns
        -------
        dict
            Estimated metric(s). If `scorer` is a dict, returns mapping name->value,
            otherwise returns {'score': value}.
        """
        check_is_fitted(self.model)

        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided to compute SHAP values for the training set.")

        # Se for pipeline, pega o último estimador (modelo final)
        model_to_explain = self.model[-1] if hasattr(self.model, "steps") else self.model
        explainer = shap.TreeExplainer(model_to_explain)

        # Se for pipeline, aplica transformações antes de calcular SHAP
        if hasattr(self.model, "steps"):
            X_train_proc = self.model[:-1].transform(X_train)
            X_eval_proc = self.model[:-1].transform(X_eval)
        else:
            X_train_proc = X_train
            X_eval_proc = X_eval

        shap_train = explainer.shap_values(X_train_proc)
        shap_eval = explainer.shap_values(X_eval_proc)

        shap_train_arr = self._choose_class_shap(shap_train, self.model)
        shap_eval_arr = self._choose_class_shap(shap_eval, self.model)

        # converter arrays SHAP 3D para 2D
        if shap_train_arr.ndim == 3:
            shap_train_arr = shap_train_arr.mean(axis=0)
        if shap_eval_arr.ndim == 3:
            shap_eval_arr = shap_eval_arr.mean(axis=0)

        # Predições no treino e avaliação pelo classificador original
        pred_train = self.model.predict(X_train)
        pred_eval = self.model.predict(X_eval)

        # Construir alvo para o classificador interno: 1 se a previsão original estava correta, 0 caso contrário
        y_right = np.array([1 if y_train[i] == pred_train[i] else 0 for i in range(len(pred_train))])

        # Treina o classificador interno em SHAP(train) -> acerto
        min_len = min(len(shap_train_arr), len(y_right))
        shap_train_arr = shap_train_arr[:min_len]
        y_right = y_right[:min_len]
        clf = clone(self.inner_clf)
        clf.fit(shap_train_arr, y_right)

        # Prediz quais instâncias de avaliação foram corretamente previstas pelo modelo original
        pred_right = clf.predict(shap_eval_arr)
        min_len = min(len(pred_eval), len(pred_right))
        pred_eval = pred_eval[:min_len]
        pred_right = pred_right[:min_len]

        # Construir labels esperadas: se previsto como correto, mantém a previsão do modelo, caso contrário inverte a classe binária
        # (assume classificação binária com classes {0,1})
        expected_y = np.array([pred_eval[i] if pred_right[i] == 1 else ((pred_eval[i] + 1) % 2)
                            for i in range(len(pred_eval))])

        if isinstance(self.scorer, dict):
            return {name: fn(expected_y, pred_eval) for name, fn in self.scorer.items()}
        elif callable(self.scorer):
            return {'score': self.scorer(expected_y, pred_eval)}
        else:
            raise ValueError("'scorer' must be a callable or a dict of callables.")
    
    def _choose_class_shap(self, shap_vals, model):
        if isinstance(shap_vals, list):
            try:
                idx = list(model.classes_).index(1)
            except Exception:
                idx = -1
            return np.array(shap_vals[idx])
        return np.array(shap_vals)