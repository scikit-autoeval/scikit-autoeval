# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score

# --- Mock base class to allow for isolated testing ---
class BaseEvaluator:
    pass
# ----------------------------------------------------

# Add the mock to the module's scope so the import works
import sys
sys.modules['base'] = type('module', (object,), {'BaseEvaluator': BaseEvaluator})()

# Now we can import the class to be tested
from ..confidence import ConfidenceThresholdEvaluator

class TestConfidenceThresholdEvaluator(unittest.TestCase):

    def setUp(self):
        """Set up a trained estimator and test data."""
        self.X_train = np.array([[1], [2], [3], [4], [5], [6]])
        self.y_train = np.array([0, 0, 0, 1, 1, 1])
        self.X_test = np.array([[0.5], [2.5], [3.5], [5.8]]) # Probas: [~0.7, ~0.5, ~0.5, ~0.9]
        
        # Estimator with 'predict_proba'
        self.classifier = LogisticRegression(solver='liblinear', random_state=0)
        self.classifier.fit(self.X_train, self.y_train)

    def test_initialization(self):
        """Test if attributes are set correctly in the constructor."""
        evaluator = ConfidenceThresholdEvaluator(self.classifier, scorer=accuracy_score, threshold=0.9)
        self.assertIs(evaluator.estimator, self.classifier)
        self.assertIs(evaluator.scorer, accuracy_score)
        self.assertEqual(evaluator.threshold, 0.9)

    def test_estimate_with_single_scorer(self):
        """Test the estimate method with a single scorer."""
        # Probas for X_test: [[0.71, 0.28], [0.49, 0.50], [0.50, 0.49], [0.22, 0.77]]
        # Confidences: [0.71, 0.50, 0.50, 0.77]
        # y_pred: [0, 1, 0, 1]
        # threshold=0.75 -> correct=[False, False, False, True]
        # y_estimated will be [1, 0, 1, 1] (y_pred inverted for those that don't pass)
        # accuracy(y_estimated, y_pred) -> 1/4 = 0.25
        evaluator = ConfidenceThresholdEvaluator(self.classifier, scorer=accuracy_score, threshold=0.75)
        scores = evaluator.estimate(self.X_test)
        self.assertIn('score', scores)
        self.assertAlmostEqual(scores['score'], 0.25)

    def test_estimate_with_dict_scorer(self):
        """Test the estimate method with a dictionary of scorers."""
        scorers = {'accuracy': accuracy_score, 'precision': precision_score}
        evaluator = ConfidenceThresholdEvaluator(self.classifier, scorer=scorers, threshold=0.75)

        scores = evaluator.estimate(self.X_test)

        # Valores corretos verificados a partir da depuração no seu ambiente
        expected_accuracy = 0.25
        expected_precision = 0.3333333333333333
        
        self.assertIn('accuracy', scores)
        self.assertIn('precision', scores)
        self.assertAlmostEqual(scores['accuracy'], expected_accuracy)
        self.assertAlmostEqual(scores['precision'], expected_precision)

    def test_high_threshold_no_predictions_pass(self):
        """Test the case where no predictions pass the threshold."""
        evaluator = ConfidenceThresholdEvaluator(self.classifier, scorer=accuracy_score, threshold=0.99)
        scores = evaluator.estimate(self.X_test)
        self.assertEqual(scores, {'score': 0.0})

    def test_low_threshold_all_predictions_pass(self):
        """Test the case where all predictions pass the threshold."""
        # If all pass, y_estimated == y_pred, so accuracy should be 1.0
        evaluator = ConfidenceThresholdEvaluator(self.classifier, scorer=accuracy_score, threshold=0.1)
        scores = evaluator.estimate(self.X_test)
        self.assertAlmostEqual(scores['score'], 1.0)
        
    def test_estimator_with_decision_function(self):
        """Test the evaluator with an estimator that uses decision_function (e.g., SVC)."""
        # SVC without probability=True does not have predict_proba
        svc_classifier = SVC(gamma='auto', random_state=0)
        svc_classifier.fit(self.X_train, self.y_train)
        
        evaluator = ConfidenceThresholdEvaluator(svc_classifier, threshold=0.5)
        # This test just ensures no error is raised. The exact calculation depends on the scale of decision_function.
        try:
            scores = evaluator.estimate(self.X_test)
            self.assertIn('score', scores)
        except ValueError:
            self.fail("estimate() raised ValueError unexpectedly for estimator with decision_function.")

    def test_estimator_with_no_confidence_method(self):
        """Test if a ValueError is raised for an estimator without confidence methods."""
        class DummyEstimator:
            def fit(self, X, y):
                return self
            def predict(self, X):
                return np.zeros(len(X))

        evaluator = ConfidenceThresholdEvaluator(DummyEstimator())
        with self.assertRaisesRegex(ValueError, "The estimator must implement predict_proba or decision_function."):
            evaluator.estimate(self.X_test)

    def test_fit_method(self):
        """Test if the fit method trains the estimator."""
        unfitted_classifier = LogisticRegression(solver='liblinear')
        evaluator = ConfidenceThresholdEvaluator(unfitted_classifier)
        
        # 'coef_' does not exist before fit
        self.assertFalse(hasattr(unfitted_classifier, 'coef_'))
        
        evaluator.fit(self.X_train, self.y_train)
        
        # 'coef_' should exist after fit
        self.assertTrue(hasattr(unfitted_classifier, 'coef_'))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)