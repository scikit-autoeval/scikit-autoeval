# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
import io
import sys

from ..regression import RegressionBasedEvaluator

class TestRegressionBasedEvaluator(unittest.TestCase):

    def setUp(self):
        """Sets up a dataset and a list of classifiers for the tests."""
        self.X, self.y = make_classification(
            n_samples=200, n_features=10, n_informative=5, n_classes=3, random_state=42
        )
        
        self.classifiers = [
            LogisticRegression(solver="liblinear", C=0.1, random_state=42),
            DecisionTreeClassifier(max_depth=5, random_state=42),
            DecisionTreeClassifier(max_depth=10, random_state=42),
        ]

    def test_initialization(self):
        """Test that the evaluator is initialized correctly."""
        # Test default case
        evaluator_default = RegressionBasedEvaluator()
        self.assertIsInstance(evaluator_default.model, RandomForestRegressor)
        self.assertFalse(evaluator_default.verbose)

        # Test with custom regressor and verbose=True
        custom_regressor = LinearRegression()
        evaluator_custom = RegressionBasedEvaluator(model=custom_regressor, verbose=True)
        self.assertIs(evaluator_custom.model, custom_regressor)
        self.assertTrue(evaluator_custom.verbose)

    def test_extract_metafeatures_calculation(self):
        """Verifies the calculation and shape of the extracted meta-features."""
        evaluator = RegressionBasedEvaluator()

        # Mock a classifier with predictable probability outputs
        class MockEstimator:
            def predict_proba(self, X):
                return np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])

        metafeatures = evaluator._extract_metafeatures(MockEstimator(), np.zeros((3, 2)))

        self.assertEqual(metafeatures.shape, (1, 4))

        probas = MockEstimator().predict_proba(None)
        max_probs = np.max(probas, axis=1)
        eps = 1e-12
        entropy = -np.sum(probas * np.log(probas + eps), axis=1)
        
        expected_features = np.array([
            np.mean(max_probs),
            np.std(max_probs),
            np.mean(entropy),
            np.std(entropy)
        ])
        
        np.testing.assert_allclose(metafeatures.flatten(), expected_features)

    def test_extract_metafeatures_raises_error(self):
        """Ensures an error is raised if the classifier lacks `predict_proba`."""
        evaluator = RegressionBasedEvaluator()
        class NoProbaEstimator:
            def predict(self, X): return np.zeros(X.shape[0])

        with self.assertRaisesRegex(ValueError, "The classifier must implement predict_proba."):
            evaluator._extract_metafeatures(NoProbaEstimator(), np.zeros((5, 2)))

    def test_fit_and_estimate_workflow(self):
        """Tests the full workflow of training the evaluator and estimating an accuracy."""
        evaluator = RegressionBasedEvaluator()
        
        # Create multiple train/validation splits to train the meta-regressor
        X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []
        for i in range(len(self.classifiers)):
            X_train, X_val, y_train, y_val = train_test_split(
                self.X, self.y, test_size=0.4, random_state=42 + i
            )
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_val_list.append(X_val)
            y_val_list.append(y_val)

        # Fit the evaluator
        evaluator.fit(self.classifiers, X_train_list, y_train_list, X_val_list, y_val_list)

        # Check that the internal meta-regressor has been fitted
        self.assertTrue(hasattr(evaluator.model, "feature_importances_"))

        # Prepare a new classifier for the estimation test
        X_train_new, X_unlabeled, y_train_new, y_unlabeled_true = train_test_split(
            self.X, self.y, test_size=0.5, random_state=100
        )
        estimator_to_test = LogisticRegression(solver="liblinear", random_state=42)
        estimator_to_test.fit(X_train_new, y_train_new)

        # Estimate the accuracy
        estimated_acc = evaluator.estimate(estimator_to_test, X_unlabeled)

        # Check the estimation result
        self.assertIsInstance(estimated_acc, float)
        self.assertTrue(0.0 <= estimated_acc <= 1.0, "Estimated accuracy must be between 0 and 1")

        # Sanity check: the estimation should be reasonably close to the true accuracy
        true_acc = estimator_to_test.score(X_unlabeled, y_unlabeled_true)
        self.assertLess(abs(true_acc - estimated_acc), 0.25, "The estimation should be close to the real accuracy")

    def test_verbose_mode(self):
        """Checks if verbose mode prints the expected information."""
        evaluator = RegressionBasedEvaluator(verbose=True)
        clf = LogisticRegression(solver="liblinear")
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.5, random_state=42)

        # Capture standard output (stdout)
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Test fit method output
        evaluator.fit(
            estimators=[clf],
            X_train_list=[X_train], y_train_list=[y_train],
            X_val_list=[X_val], y_val_list=[y_val],
        )
        fit_output = captured_output.getvalue()
        self.assertIn("[INFO] True accuracy:", fit_output)
        
        # Test estimate method output
        captured_output.truncate(0)
        captured_output.seek(0)
        
        clf.fit(X_train, y_train)
        evaluator.estimate(clf, X_val)
        estimate_output = captured_output.getvalue()
        self.assertIn("[INFO] Estimated accuracy:", estimate_output)

        # Restore standard output
        sys.stdout = sys.__stdout__

# Allows running the tests directly from the script
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)