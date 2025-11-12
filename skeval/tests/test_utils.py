# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import NotFittedError

# Assuming the functions are in a utils.py file in this path
from skeval.utils import check_is_fitted, get_CV_and_real_scores

# A mock model that is "fitted" but lacks predict_proba or decision_function
class MockModelNoProba:
    def __init__(self):
        self.fitted_ = True # To pass sklearn_check_is_fitted
    
    def fit(self, X, y):
        self.fitted_ = True
        return self
    
    def predict(self, X):
        return np.zeros(len(X))

class TestCheckIsFitted(unittest.TestCase):

    def test_fitted_model_with_proba_passes(self):
        """
        Test that a fitted model with 'predict_proba' passes the check.
        """
        # CORRECTION: Added a second class (1) to the fit data
        X_fit = np.array([[0,0], [1,1]])
        y_fit = np.array([0, 1])
        model = LogisticRegression().fit(X_fit, y_fit)
        
        try:
            check_is_fitted(model)
        except (NotFittedError, ValueError):
            self.fail("check_is_fitted raised an exception unexpectedly.")

    def test_fitted_model_with_decision_function_passes(self):
        """
        Test that a fitted model with 'decision_function' passes the check.
        """
        # CORRECTION: Added a second class (1) to the fit data
        X_fit = np.array([[0,0], [1,1]])
        y_fit = np.array([0, 1])
        model = SVC().fit(X_fit, y_fit)
        
        try:
            check_is_fitted(model)
        except (NotFittedError, ValueError):
            self.fail("check_is_fitted raised an exception unexpectedly.")

    def test_unfitted_model_raises_not_fitted_error(self):
        """
        Test that an unfitted model raises NotFittedError.
        """
        model = LogisticRegression()
        # This should fail the internal sklearn_check_is_fitted
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)

    def test_model_without_proba_raises_value_error(self):
        """
        Test that a model without 'predict_proba' or 'decision_function'
        raises ValueError.
        """
        model = MockModelNoProba().fit(None, None)
        # This should fail the custom check
        with self.assertRaises(ValueError):
            check_is_fitted(model)


class TestGetCVAndRealScores(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        X, y = load_breast_cancer(return_X_y=True)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        cls.scorers = {
            "accuracy": accuracy_score,
            "f1": f1_score
        }
        cls.model_base = LogisticRegression(max_iter=1000)

    def test_output_structure_and_keys(self):
        """
        Test the output structure, keys, and value types.
        """
        results = get_CV_and_real_scores(
            self.model_base,
            self.scorers,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test
        )

        # Check top-level structure
        self.assertIsInstance(results, dict)
        self.assertIn("cv_scores", results)
        self.assertIn("real_scores", results)

        # Check sub-dictionary keys
        self.assertIsInstance(results["cv_scores"], dict)
        self.assertIsInstance(results["real_scores"], dict)
        self.assertIn("accuracy", results["cv_scores"])
        self.assertIn("f1", results["cv_scores"])
        self.assertIn("accuracy", results["real_scores"])
        self.assertIn("f1", results["real_scores"])

        # Check value types
        for score_name, score_value in results["cv_scores"].items():
            self.assertIsInstance(score_value, float)
            self.assertGreaterEqual(score_value, 0.0, f"CV score {score_name} was not >= 0")

        for score_name, score_value in results["real_scores"].items():
            self.assertIsInstance(score_value, float)
            self.assertGreaterEqual(score_value, 0.0, f"Real score {score_name} was not >= 0")


if __name__ == "__main__":
    unittest.main()