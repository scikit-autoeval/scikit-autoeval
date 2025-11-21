# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, Dict, Mapping, Optional, Union

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from skeval.base import BaseEvaluator
from skeval.utils import check_is_fitted


class AgreementEvaluator(BaseEvaluator):
    """Agreement-based evaluator for supervised classification models.

    This evaluator compares predictions produced by a *primary* model (``model``)
    and a *secondary* model (``sec_model``) on an evaluation set. For each
    sample, an agreement indicator is defined as ``1`` when both models predict
    the same class and ``0`` otherwise. Using this indicator, an *expected label*
    vector is created by flipping the primary model's prediction when the models
    disagree. Metric(s) are then computed comparing the expected label vector
    to the agreement indicator, providing an estimate of how often the primary
    model's predictions would align with a plausible correction strategy based
    on model disagreement.

    Evaluation workflow:
    1. Fit both the primary and secondary models on the training data.
    2. Generate predictions for both models on the evaluation data.
    3. Build the agreement vector (1 = same prediction, 0 = different).
    4. Produce an expected label vector, flipping predictions where disagreement occurs.
    5. Compute the chosen metric(s) using the scorer(s).

    Parameters
    ----------
    model : estimator
        A classification estimator implementing ``fit`` and ``predict``.
        May be a single estimator or a pipeline created with
        ``sklearn.pipeline.make_pipeline``.
    scorer : callable or dict of str -> callable, default=accuracy_score
        A single scoring function or a dictionary mapping metric names to
        scoring callables. Each scorer must follow the signature
        ``scorer(y_true, y_pred)``.
    verbose : bool, default=False
        If ``True``, prints progress information during fit and estimate.
    sec_model : estimator, optional
        Secondary classification model used solely to generate comparison
        predictions. If ``None``, defaults to ``GaussianNB()``.

    Attributes
    ----------
    model : estimator
        The primary model provided at initialization.
    sec_model : estimator
        The secondary model used to create agreement signals.

    Notes
    -----
    This evaluator assumes both models output class labels directly via
    ``predict``. No probability calibration is performed. The metric(s) are
    computed on synthetic targets produced from model agreement—not against
    real ground-truth labels—so scores should be interpreted as *agreement-based
    estimates*, not actual performance metrics.

    Examples
    --------
    Basic usage with two RandomForest pipelines and multiple scorers:

    >>> import pandas as pd
    >>> from sklearn.metrics import accuracy_score, f1_score
    >>> from sklearn.impute import KNNImputer
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from skeval.evaluators.agreement import AgreementEvaluator
    >>> from skeval.utils import get_cv_and_real_scores, print_comparison
    >>> df_geriatrics = pd.read_csv("geriatria.csv")
    >>> df_neurology = pd.read_csv("neurologia.csv")
    >>> X1, y1 = df_geriatrics.drop(columns=["Alzheimer"]), df_geriatrics["Alzheimer"]
    >>> X2, y2 = df_neurology.drop(columns=["Alzheimer"]), df_neurology["Alzheimer"]
    >>> model = make_pipeline(
    ...     KNNImputer(n_neighbors=10),
    ...     RandomForestClassifier(n_estimators=50, random_state=42),
    ... )
    >>> sec_model = make_pipeline(
    ...     KNNImputer(n_neighbors=10),
    ...     RandomForestClassifier(n_estimators=100, random_state=42),
    ... )
    >>> scorers = {
    ...     "accuracy": accuracy_score,
    ...     "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
    ... }
    >>> evaluator = AgreementEvaluator(model=model, sec_model=sec_model, scorer=scorers)
    >>> evaluator.fit(X1, y1)
    >>> estimated_scores = evaluator.estimate(X2)
    >>> # Optionally compare with CV and real scores
    >>> scores_dict = get_cv_and_real_scores(
    ...     model=model, scorers=scorers, train_data=(X1, y1), test_data=(X2, y2)
    ... )
    >>> cv_scores = scores_dict["cv_scores"]
    >>> real_scores = scores_dict["real_scores"]
    >>> print_comparison(scorers, cv_scores, estimated_scores, real_scores)
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
        """Fit the evaluator by training both primary and secondary models.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Feature matrix used to fit both models.
        y : array-like of shape (n_samples,)
            Target labels corresponding to ``x``.

        Returns
        -------
        self : AgreementEvaluator
            The fitted evaluator instance.
        """

        self.model.fit(x, y)
        self.sec_model.fit(x, y)

        if self.verbose:
            print("[INFO] Fit completed.")

        return self

    def estimate(self, x_eval: Any) -> Dict[str, float]:
        """Estimate agreement-based metric values on evaluation data.

        Generates predictions from both models, constructs an agreement vector
        and an expected label vector (flipping the primary prediction when
        disagreement occurs), then applies the configured scorer(s).

        Parameters
        ----------
        x_eval : array-like of shape (n_samples, n_features)
            Evaluation feature matrix.

        Returns
        -------
        scores : dict
            If ``scorer`` is a dict, returns a mapping from metric name to
            agreement-based score. Otherwise returns ``{"score": float}``.

        Raises
        ------
        ValueError
            If ``scorer`` is neither a callable nor a dict of callables.
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
