from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from skeval.base import BaseEvaluator
import numpy as np

class RegressionBasedEvaluator(BaseEvaluator, RegressorMixin):
    """
    Regression-based evaluator for classification models.

    This evaluator estimates the accuracy of a classification model without
    requiring labeled test data. It trains a meta-regressor (e.g., Random Forest)
    that maps meta-features extracted from predicted probability distributions
    and classifier behavior to the model's true accuracy. Once trained, the
    regressor can be used to predict the accuracy of new classifiers on
    unlabeled datasets.

    Parameters
    ----------
    meta_regressor : object, default=None
        A regression model implementing `fit` and `predict`.
        If None, a RandomForestRegressor with 100 trees is used.
    verbose : bool, default=False
        If True, prints additional information during training and estimation.

    Attributes
    ----------
    meta_regressor : object
        The regression model trained to predict classifier accuracy.
    verbose : bool
        The verbosity level.
    mean_conf : float
        Mean confidence score, computed as the average maximum predicted probability per sample.
    std_conf : float
        Standard deviation of the confidence scores across all samples.
    mean_entropy : float
        Mean entropy of the predicted probability distributions, reflecting overall uncertainty.
    std_entropy : float
        Standard deviation of the entropy values, reflecting variability in uncertainty across samples.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skeval.regression_based import RegressionBasedEvaluator
    >>> 
    >>> # Load dataset
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    >>> 
    >>> # Base classifier
    >>> clf = LogisticRegression(max_iter=200)
    >>> 
    >>> # Create evaluator
    >>> evaluator = RegressionBasedEvaluator()
    >>> 
    >>> # Fit evaluator using labeled validation data
    >>> evaluator.fit(
    ...     estimators=[clf],
    ...     X_train_list=[X_train],
    ...     y_train_list=[y_train],
    ...     X_val_list=[X_val],
    ...     y_val_list=[y_val]
    ... )
    RegressionBasedEvaluator(...)
    >>> 
    >>> # Estimate accuracy on unlabeled data
    >>> acc_est = evaluator.estimate(clf, X_val)
    >>> print(f"Estimated accuracy: {acc_est:.2f}")
    Estimated accuracy: 0.92
    """

    def __init__(self, meta_regressor=None, verbose=False):
        self.meta_regressor = meta_regressor or RandomForestRegressor(n_estimators=100, random_state=42)
        self.verbose = verbose

    def _extract_metafeatures(self, estimator, X):
        """
        Extracts meta-features from input data based on predicted probability
        distributions.

        Examples of meta-features include:
        - mean entropy of predicted probability distributions
        - variance of predicted class probabilities
        - mean maximum probability (average confidence)

        Parameters
        ----------
        estimator : classifier with predict_proba method
            The base classifier used for probability estimation.
        X : array-like of shape (n_samples, n_features)
            Input data to extract meta-features from.

        Returns
        -------
        ndarray of shape (1, n_features)
            Extracted meta-features for the given input data.

        Raises
        ------
        ValueError
            If the classifier does not implement `predict_proba`.
        """
        if hasattr(estimator, "predict_proba"):
            probas = estimator.predict_proba(X)
        else:
            raise ValueError("The classifier must implement predict_proba.")

        max_probs = np.max(probas, axis=1)

        eps = 1e-12
        entropy = -np.sum(probas * np.log(probas + eps), axis=1)

        features = {
            "mean_conf": np.mean(max_probs),
            "std_conf": np.std(max_probs),
            "mean_entropy": np.mean(entropy),
            "std_entropy": np.std(entropy)
        }

        return np.array(list(features.values())).reshape(1, -1)

    def fit(self, estimators, X_train_list, y_train_list, X_val_list, y_val_list):
        """
        Trains the regression meta-model based on multiple classifiers
        and datasets, where the true accuracies are known.

        Parameters
        ----------
        estimators : list
            List of trained classifiers.
        X_train_list, y_train_list : list
            Lists of training data and labels.
        X_val_list, y_val_list : list
            Lists of validation data (with labels) to compute true accuracy.

        Returns
        -------
        self : object
            Fitted RegressionBasedEvaluator instance.
        """
        meta_X = []
        meta_y = []

        for est, X_train, y_train, X_val, y_val in zip(estimators, X_train_list, y_train_list, X_val_list, y_val_list):
            est.fit(X_train, y_train)

            feats = self._extract_metafeatures(est, X_val)

            acc = est.score(X_val, y_val)

            meta_X.append(feats.flatten())
            meta_y.append(acc)

            if self.verbose:
                print(f"[INFO] True accuracy: {acc:.4f}, Metafeatures: {feats.flatten()}")

        meta_X = np.array(meta_X)
        meta_y = np.array(meta_y)

        self.meta_regressor.fit(meta_X, meta_y)
        return self

    def estimate(self, estimator, X_unlabeled):
        """
        Estimates the accuracy of a classifier on an unlabeled dataset.

        Parameters
        ----------
        estimator : fitted classifier
            The pre-trained classifier to evaluate.
        X_unlabeled : array-like
            Unlabeled input dataset.

        Returns
        -------
        float
            Estimated accuracy of the classifier.
        """
        feats = self._extract_metafeatures(estimator, X_unlabeled)
        acc_estimate = self.meta_regressor.predict(feats)[0]

        if self.verbose:
            print(f"[INFO] Estimated accuracy: {acc_estimate:.4f}")

        return acc_estimate
