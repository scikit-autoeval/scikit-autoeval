# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import numpy as np
import io
import contextlib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import NotFittedError

from skeval.utils import check_is_fitted, get_cv_and_real_scores, print_comparison


# A mock model that is "fitted" but lacks predict_proba or decision_function
class MockModelNoProba:
    def __init__(self):
        self.fitted_ = True

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.zeros(len(X))


class TestCheckIsFitted(unittest.TestCase):

    def test_fitted_model_with_proba_passes(self):
        """Test that a fitted model with predict_proba passes the check."""
        X_fit = np.array([[0, 0], [1, 1]])
        y_fit = np.array([0, 1])
        model = LogisticRegression().fit(X_fit, y_fit)

        try:
            check_is_fitted(model)
        except (NotFittedError, ValueError):
            self.fail("check_is_fitted raised an exception unexpectedly.")

    def test_fitted_model_with_decision_function_passes(self):
        """Test that a fitted model with decision_function passes the check."""
        X_fit = np.array([[0, 0], [1, 1]])
        y_fit = np.array([0, 1])
        model = SVC().fit(X_fit, y_fit)

        try:
            check_is_fitted(model)
        except (NotFittedError, ValueError):
            self.fail("check_is_fitted raised an exception unexpectedly.")

    def test_unfitted_model_raises_not_fitted_error(self):
        """Test that an unfitted model raises NotFittedError."""
        model = LogisticRegression()
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)

    def test_model_without_proba_raises_value_error(self):
        """Test that models lacking proba/decision_function raise ValueError."""
        model = MockModelNoProba().fit(None, None)
        with self.assertRaises(ValueError):
            check_is_fitted(model)


class TestGetCVAndRealScores(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        X, y = load_breast_cancer(return_X_y=True)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        cls.scorers = {"accuracy": accuracy_score, "f1": f1_score}
        cls.model_base = LogisticRegression(max_iter=2000)

    def test_output_structure_and_keys(self):
        """Test the output structure, keys, and value types."""
        train_data = self.X_train, self.y_train
        test_data = self.X_test, self.y_test

        results = get_cv_and_real_scores(
            self.model_base,
            self.scorers,
            train_data,
            test_data,
        )

        # Structure
        self.assertIsInstance(results, dict)
        self.assertIn("cv_scores", results)
        self.assertIn("real_scores", results)

        # Keys inside dictionaries
        for section in ["cv_scores", "real_scores"]:
            self.assertIsInstance(results[section], dict)
            self.assertIn("accuracy", results[section])
            self.assertIn("f1", results[section])

        # Values must be floats >= 0
        for score_dict in [results["cv_scores"], results["real_scores"]]:
            for score_value in score_dict.values():
                self.assertIsInstance(score_value, float)
                self.assertGreaterEqual(score_value, 0.0)

    def test_print_comparison_runs_without_errors(self):
        """Ensure print_comparison prints output without raising exceptions."""
        scorers = {"acc": None, "f1": None}
        cv_scores = {"acc": 0.9, "f1": 0.8}
        estimated_scores = {"acc": 0.88, "f1": 0.79}
        real_scores = {"acc": 0.91, "f1": 0.82}

        # Capture printed output
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_comparison(
                scorers=scorers,
                cv_scores=cv_scores,
                estimated_scores=estimated_scores,
                real_scores=real_scores,
            )

        output = buffer.getvalue()
        self.assertIn("CV vs. Estimated vs. Real", output)


if __name__ == "__main__":
    unittest.main()
