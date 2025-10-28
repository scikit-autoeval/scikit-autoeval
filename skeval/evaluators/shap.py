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

    def __init__(self, model, scorer=accuracy_score, verbose=False, inner_clf=None, X_train=None, y_train=None):
        super().__init__(model=model, scorer=scorer, verbose=verbose)
        self.inner_clf = inner_clf or XGBClassifier()
        self.explainer = None
        self.X_train = X_train
        self.y_train = y_train
        
    def fit(self, X=None, y=None):
        """
        Fit the evaluator's model.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training labels.
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
        Estimate metrics using SHAP-based correctness prediction.

        Parameters
        ----------
        X_eval : array-like
            The evaluation features (used as X_test in the pseudo-code).
        n_pred : int, optional
            Number of predictions to generate. If None, uses 30.

        Returns
        -------
        dict
            Estimated metric(s). If `scorer` is a dict, returns mapping name->value,
            otherwise returns {'score': value}.
        """

        check_is_fitted(self.model)
        if self.X_train is None or self.y_train is None:
            raise ValueError("X_train and y_train must be provided to compute SHAP values for the training set.")

        # Se for pipeline, pega o último estimador (modelo final)
        model_to_explain = self.model[-1] if hasattr(self.model, "steps") else self.model
        self.explainer = shap.TreeExplainer(model_to_explain)

        X_train_proc = self.X_train
        X_eval_proc = X_eval

        shap_train = self.explainer.shap_values(X_train_proc)
        shap_eval = self.explainer.shap_values(X_eval_proc)

        shap_train_arr = self._choose_class_shap(shap_train, self.model)
        shap_eval_arr = self._choose_class_shap(shap_eval, self.model)
        
        # Predições no treino e avaliação pelo classificador original
        scores_list = []

        for _ in range(n_pred):
            pred_train = np.random.randint(0, 2, size=len(self.y_train))
            pred_eval = self.model.predict(X_eval)

            # Construir alvo para o classificador interno: 1 se a previsão original estava correta, 0 caso contrário
            y_right = np.array([1 if self.y_train[i] == pred_train[i] else 0 for i in range(len(pred_train))])

            # Treina o classificador interno em SHAP(train) -> acerto
            clf = clone(self.inner_clf)
            clf.fit(shap_train_arr, y_right)

            # Prediz quais instâncias de avaliação foram corretamente previstas pelo modelo original
            pred_right = clf.predict(shap_eval_arr)

            # Construir labels esperadas: se previsto como correto, mantém a previsão do modelo, caso contrário inverte a classe binária
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
        if isinstance(shap_vals, list):
            try:
                idx = list(model.classes_).index(1)
            except Exception:
                idx = -1
            return np.array(shap_vals[idx])
        return np.array(shap_vals)