# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
import unittest
import numpy as np
import io
import contextlib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
from sklearn.exceptions import NotFittedError

from skeval.evaluators.confidence import ConfidenceThresholdEvaluator


class TestConfidenceThresholdEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up a trained model and test data."""
        self.X_train = np.array([[1], [2], [3], [4], [5], [6]])
        self.y_train = np.array([0, 0, 0, 1, 1, 1])
        self.X_test = np.array(
            [[0.5], [2.5], [3.5], [5.8]]
        )  # Probas: [~0.7, ~0.5, ~0.5, ~0.9]

        # Estimator with 'predict_proba'
        self.classifier = LogisticRegression(solver="liblinear", random_state=0)
        self.classifier.fit(self.X_train, self.y_train)

        # Data for multiclass test
        self.X_multi = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
        self.y_multi = np.array([0, 0, 1, 1, 2, 2, 0, 1])

    def test_initialization(self):
        """Test if attributes are set correctly in the constructor."""
        evaluator = ConfidenceThresholdEvaluator(self.classifier, scorer=accuracy_score)
        self.assertIs(evaluator.model, self.classifier)
        self.assertIs(evaluator.scorer, accuracy_score)

    def test_initialization_defaults(self):
        """Test if default parameters are set."""
        evaluator = ConfidenceThresholdEvaluator(self.classifier)
        self.assertFalse(evaluator.verbose)
        self.assertIs(evaluator.scorer, accuracy_score)

    def test_estimate_with_single_scorer(self):
        """Test the estimate method with a single scorer."""
        evaluator = ConfidenceThresholdEvaluator(self.classifier, scorer=accuracy_score)
        scores = evaluator.estimate(self.X_test, threshold=0.75)
        self.assertIn("score", scores)
        self.assertAlmostEqual(scores["score"], 0.25)

    def test_estimate_with_dict_scorer(self):
        """Test the estimate method with a dictionary of scorers."""
        scorers = {
            "accuracy": accuracy_score,
            "precision": lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
        }
        evaluator = ConfidenceThresholdEvaluator(self.classifier, scorer=scorers)

        scores = evaluator.estimate(self.X_test, threshold=0.75)

        expected_accuracy = 0.25
        expected_precision = 0.3333333333333333

        self.assertIn("accuracy", scores)
        self.assertIn("precision", scores)
        self.assertAlmostEqual(scores["accuracy"], expected_accuracy)
        self.assertAlmostEqual(scores["precision"], expected_precision)

    def test_high_threshold_no_predictions_pass(self):
        """Test the case where no predictions pass the threshold."""
        evaluator = ConfidenceThresholdEvaluator(self.classifier, scorer=accuracy_score)
        scores = evaluator.estimate(self.X_test, threshold=0.99)
        # Deve retornar 0.0 para todos os scorers
        self.assertEqual(scores, {"score": 0.0})

    def test_low_threshold_all_predictions_pass(self):
        """Test the case where all predictions pass the threshold."""
        # If all pass, y_estimated == y_pred, so accuracy should be 1.0
        evaluator = ConfidenceThresholdEvaluator(self.classifier, scorer=accuracy_score)
        scores = evaluator.estimate(self.X_test, threshold=0.1)
        self.assertAlmostEqual(scores["score"], 1.0)

    def test_estimator_with_decision_function(self):
        """Test the evaluator with an model that uses decision_function (e.g., SVC)."""
        # SVC without probability=True does not have predict_proba
        svc_classifier = SVC(gamma="auto", random_state=0)
        svc_classifier.fit(self.X_train, self.y_train)

        # O decision_function do SVC binário é 1D, testando o branch np.abs(decision)
        evaluator = ConfidenceThresholdEvaluator(svc_classifier)

        try:
            scores = evaluator.estimate(self.X_test, threshold=0.5)
            self.assertIn("score", scores)
        except ValueError:
            self.fail(
                "estimate() raised ValueError unexpectedly for model with decision_function."
            )

    def test_estimator_with_no_confidence_method(self):
        """Test if a ValueError is raised for an model without confidence methods."""

        class DummyEstimator:
            def fit(self, X, y):
                self.fitted_ = True  # Para passar no check_is_fitted
                return self

            def predict(self, X):
                return np.zeros(len(X))

        dummy = DummyEstimator()
        dummy.fit(self.X_train, self.y_train)  # Precisa estar "fitado"

        evaluator = ConfidenceThresholdEvaluator(dummy)

        # O erro agora é levantado em __get_confidences_and_correct
        with self.assertRaisesRegex(
            ValueError, "The model must implement predict_proba or decision_function."
        ):
            evaluator.estimate(self.X_test)

    def test_fit_method(self):
        """Test if the fit method trains the model."""
        unfitted_classifier = LogisticRegression(solver="liblinear")
        evaluator = ConfidenceThresholdEvaluator(unfitted_classifier)

        # 'coef_' does not exist before fit
        self.assertFalse(hasattr(unfitted_classifier, "coef_"))

        # Usando NotFittedError para verificar que não está fitado
        with self.assertRaises(NotFittedError):
            unfitted_classifier.predict(self.X_test)

        evaluator.fit(self.X_train, self.y_train)

        # 'coef_' should exist after fit
        self.assertTrue(hasattr(unfitted_classifier, "coef_"))
        # Deve ser capaz de prever agora
        try:
            unfitted_classifier.predict(self.X_test)
        except NotFittedError:
            self.fail("Model was not fitted by evaluator.fit()")

    # --- NOVOS TESTES PARA COBERTURA ---

    def test_verbose_logging_fit_and_estimate_single_scorer(self):
        """Test all verbose branches for fit and estimate (single scorer)."""
        evaluator = ConfidenceThresholdEvaluator(self.classifier, verbose=True)

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            evaluator.fit(self.X_train, self.y_train)
            evaluator.estimate(self.X_test, threshold=0.75)

        output = f.getvalue()

        # Testando saídas do 'fit'
        self.assertIn("[INFO] Model has been trained.", output)

        # Testando saídas do 'estimate'
        self.assertIn("[INFO] Confidences:", output)
        self.assertIn("[INFO] Passed threshold:", output)
        self.assertIn("[INFO] y_pred:", output)
        self.assertIn("[INFO] y_estimated:", output)
        # Test verbose para 'callable(self.scorer)'
        self.assertIn("[INFO] Estimated score:", output)

    def test_verbose_logging_dict_scorer(self):
        """Test verbose branch for dict scorer in estimate."""
        scorers = {"accuracy": accuracy_score}
        evaluator = ConfidenceThresholdEvaluator(
            self.classifier, scorer=scorers, verbose=True
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            evaluator.estimate(self.X_test, threshold=0.75)

        output = f.getvalue()
        # Test verbose para 'isinstance(self.scorer, dict)'
        self.assertIn("[INFO] Estimated scores:", output)

    def test_verbose_logging_no_pass_threshold(self):
        """Test verbose branch for 'no predictions passed'."""
        evaluator = ConfidenceThresholdEvaluator(self.classifier, verbose=True)

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            evaluator.estimate(self.X_test, threshold=0.99)

        output = f.getvalue()
        # Test verbose para 'if not np.any(correct):'
        self.assertIn("[INFO] No predictions passed the threshold.", output)

    def test_invalid_scorer_type_raises_error(self):
        """Test the 'else' branch in estimate for invalid scorer type."""
        evaluator = ConfidenceThresholdEvaluator(
            self.classifier, scorer="not_a_callable"
        )
        with self.assertRaisesRegex(
            ValueError, "'scorer' must be a callable or a dict of callables."
        ):
            evaluator.estimate(self.X_test)

        evaluator_none = ConfidenceThresholdEvaluator(self.classifier, scorer=None)
        with self.assertRaisesRegex(
            ValueError, "'scorer' must be a callable or a dict of callables."
        ):
            evaluator_none.estimate(self.X_test)

    def test_decision_function_multiclass(self):
        """Test decision_function branch for multiclass (decision.ndim > 1)."""
        svc_multi = SVC(decision_function_shape="ovr", random_state=0)
        svc_multi.fit(self.X_multi, self.y_multi)

        evaluator = ConfidenceThresholdEvaluator(svc_multi)

        try:
            scores = evaluator.estimate(self.X_multi, threshold=0.1)
            self.assertIn("score", scores)
            self.assertGreater(scores["score"], 0)  # Deve ser > 0 com threshold baixo
        except Exception as e:
            self.fail(f"estimate() failed on multiclass SVC: {e}")

    def test_limit_to_top_class_false_raises_error(self):
        """Test the 'else probas' branch for limit_to_top_class=False."""
        evaluator = ConfidenceThresholdEvaluator(self.classifier)

        with self.assertRaises(ValueError):
            evaluator.estimate(self.X_test, limit_to_top_class=False)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
