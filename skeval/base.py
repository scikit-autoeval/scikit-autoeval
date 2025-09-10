from sklearn.base import BaseEstimator

class BaseEvaluator(BaseEstimator):
    """
    Base abstract class for all evaluators in scikit-autoeval.

    All evaluators should inherit from this class and implement the `estimate` method.
    This class also inherits from `sklearn.base.BaseEstimator` to ensure compatibility
    with scikit-learn utilities like `get_params` and `set_params`.
    """

    def estimate(self, X):
        """
        Abstract method to estimate the model's performance on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        scores : dict
            A dictionary containing the evaluation scores.
        """
        pass
    
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
        pass


