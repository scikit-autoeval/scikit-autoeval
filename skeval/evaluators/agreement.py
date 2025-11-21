# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, Dict, Mapping, Optional, Union

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from skeval.base import BaseEvaluator
from skeval.utils import check_is_fitted


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
        model: Any,
        scorer: Union[Mapping[str, Any], Any] = accuracy_score,
        verbose: bool = False,
        sec_model: Optional[Any] = None,
    ) -> None:
        super().__init__(model=model, scorer=scorer, verbose=verbose)

        self.sec_model = sec_model if sec_model is not None else GaussianNB()

    def fit(self, x: Any, y: Any) -> "AgreementEvaluator":
        """
        Fit the evaluator by generating predictions from both models.

        Parameters
        ----------
        x : array-like
            Feature matrix.
        y : array-like
            Target vector.
        """

        self.model.fit(x, y)
        self.sec_model.fit(x, y)

        if self.verbose:
            print("[INFO] Fit completed.")

        return self

    def estimate(self, x_eval: Any) -> Dict[str, float]:
        """
        Estimate the agreement score between the main and secondary models.

        Parameters
        ----------
        x_eval : array-like
            Feature matrix (not used, kept for interface consistency).

        Returns
        -------
        scores : dict or float
            Agreement score(s) computed using the provided scorer(s).
        """

        check_is_fitted(self.model)
        check_is_fitted(self.sec_model)

        pred_main = self.model.predict(x_eval)
        pred_secondary = self.sec_model.predict(x_eval)

        agreement = (pred_main == pred_secondary).astype(int)
        y_agreement = [p if a else 1 - p for p, a in zip(pred_main, agreement)]

        if isinstance(self.scorer, dict):
            score: Dict[str, float] = {
                name: float(metric(y_agreement, agreement))
                for name, metric in self.scorer.items()
            }
            if self.verbose:
                print("[INFO] Estimated score:", score)

            return score
        if callable(self.scorer):
            score_val = float(self.scorer(y_agreement, agreement))
            if self.verbose:
                print("[INFO] Estimated score:", score_val)
            return {"score": score_val}
        raise ValueError("'scorer' must be a callable or a dict of callables.")
