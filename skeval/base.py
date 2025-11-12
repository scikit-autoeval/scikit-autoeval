from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

class BaseEvaluator(BaseEstimator):
    """
    Base abstract class for all evaluators in scikit-autoeval.

    All evaluators should inherit from this class and implement the `estimate` method.
    This class also inherits from `sklearn.base.BaseEstimator` to ensure compatibility
    with scikit-learn utilities like `get_params` and `set_params`.
    """
    
    def __init__(self, model, scorer=accuracy_score, verbose=False):
        self.model = model
        self.scorer = scorer
        self.verbose = verbose
    
    def fit(self, X, y):
        """
        Fit the evaluator to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def estimate(self, X_eval):
        """
        Abstract method to estimate the model's performance on the given data.

        Parameters
        ----------
        X_eval : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        scores : dict
            A dictionary containing the evaluation scores.
        """
        pass
    
    def _get_scorer_names(self):
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
        elif callable(self.scorer):
            return ['score']
        else:
            return []
