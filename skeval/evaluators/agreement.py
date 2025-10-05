# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

from ..base import BaseEvaluator


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
        scorer=accuracy_score,
        verbose=False,
        sec_model=None,
    ):
        super().__init__(model=model, scorer=scorer, verbose=verbose)

        self.sec_model = sec_model if sec_model is not None else GaussianNB()

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

        self.model.fit(X, y)
        self.sec_model.fit(X, y)
        
        if self.verbose:
            print("[INFO] Fit completed.")

        return self

    def estimate(self, X_eval):
        """
        Estimate the agreement score between the main and secondary models.

        Parameters
        ----------
        X_eval : array-like
            Feature matrix (not used, kept for interface consistency).

        Returns
        -------
        scores : dict or float
            Agreement score(s) computed using the provided scorer(s).
        """
        
        check_is_fitted(self.model)
        check_is_fitted(self.sec_model)
        
        pred_main = self.model.predict(X_eval)
        pred_secondary = self.sec_model.predict(X_eval)

        agreement = (pred_main == pred_secondary).astype(int)
        y_agreement = [p if a else 1 - p for p, a in zip(pred_main, agreement)]
        
        if isinstance(self.scorer, dict):
            return {name: metric(y_agreement, agreement) for name, metric in self.scorer.items()}
        else:
            return self.scorer(y_agreement, agreement)