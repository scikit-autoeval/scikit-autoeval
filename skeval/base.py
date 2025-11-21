# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, Callable, Dict, List, Mapping, Union
from sklearn.metrics import accuracy_score


class BaseEvaluator:
    """
    Base abstract class for all evaluators in scikit-autoeval.

    All evaluators should inherit from this class and implement the `estimate` method.
    """

    def __init__(
        self,
        model: Any,
        scorer: Union[
            Callable[..., Any], Mapping[str, Callable[..., Any]]
        ] = accuracy_score,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.scorer = scorer
        self.verbose = verbose

    def fit(self, x: Any, y: Any) -> "BaseEvaluator":
        """
        Fit the evaluator to the training data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.model.fit(x, y)
        return self

    def estimate(self, x_eval: Any) -> Dict[str, Any]:
        """
        Abstract method to estimate the model's performance on the given data.

        Parameters
        ----------
        x_eval : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        scores : dict
            A dictionary containing the evaluation scores.
        """
        print("Please implement the 'estimate' method to estimate " + str(x_eval))
        return {}

    def _get_scorer_names(self) -> List[str]:
        """
        Returns the names of the scorers.

        This helper method gets the names of the scoring functions to be used
        as keys in the results dictionary.

        Returns
        -------
        list
            A list containing the names of the scorers. If the scorer is a
            single callable, it returns `['score']`.
        """
        if isinstance(self.scorer, dict):
            return list(self.scorer.keys())
        if callable(self.scorer):
            return ["score"]
        return []
