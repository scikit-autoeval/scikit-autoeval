# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from skeval.base import BaseEvaluator


class AgreementEvaluator(BaseEvaluator):
    """
    Evaluator based on agreement/disagreement between the target model (M)
    and a secondary model (M').

    The agreement is computed as:
        1 if the prediction of M equals the prediction of M' for an instance,
        0 otherwise.

    The secondary model is set to Naive Bayes by default, but can be replaced
    by any other classifier.
    """

    def __init__(
        self,
        model,
        sec_model=None,
        scorer=accuracy_score,
        n_splits=10,
        use_train=True,
        verbose=False,
    ):
        self.model = model
        self.sec_model = sec_model if sec_model is not None else GaussianNB()
        self.scorer = scorer
        self.n_splits = n_splits
        self.use_train = use_train
        self.verbose = verbose
        self._pred_main = None
        self._pred_secondary = None
        self._y = None

    def _cross_val_predict(self, model, X, y):
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        preds = np.zeros(len(y), dtype=np.int64)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            clone_model = clone(model)
            clone_model.fit(X[train_idx], y[train_idx])
            preds[test_idx] = clone_model.predict(X[test_idx])

            if self.verbose:
                print(f"[AgreementEvaluator] Fold {fold}/{self.n_splits} completed.")

        return preds

    def fit(self, X, y):
        """
        Fit the evaluator by generating predictions from both models.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        """
        self._y = y 

        if self.use_train:
            self._pred_main = self._cross_val_predict(self.model, X, y)
            self._pred_secondary = self._cross_val_predict(self.sec_model, X, y)
        else:
            self.model.fit(X, y)
            self.sec_model.fit(X, y)
            self._pred_main = self.model.predict(X)
            self._pred_secondary = self.sec_model.predict(X)

        if self.verbose:
            print("[AgreementEvaluator] Fit completed.")

        return self

    def estimate(self, X):
        """
        Estimate the agreement score between the main and secondary models.

        Parameters
        ----------
        X : array-like
            Feature matrix (not used, kept for interface consistency).

        Returns
        -------
        scores : dict or float
            Agreement score(s) computed using the provided scorer(s).
        """
        if self._pred_main is None or self._pred_secondary is None:
            raise ValueError("You must call `fit` before `estimate`.")

        agreement = (self._pred_main == self._pred_secondary).astype(int)

        if isinstance(self.scorer, dict):
            return {name: metric(self._y, agreement) for name, metric in self.scorer.items()}
        else:
            return self.scorer(self._y, agreement)

