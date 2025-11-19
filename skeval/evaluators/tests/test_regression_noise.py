# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

import unittest
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import NotFittedError

# Assuming the RegressionNoiseEvaluator is in this path,
# which inherits from RegressionEvaluator
from skeval.evaluators.regression_noise import RegressionNoiseEvaluator


class TestRegressionNoiseEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        iris = load_iris()
        cancer = load_breast_cancer()

        # The fit method of RegressionNoiseEvaluator uses pandas operations
        # (.columns, .copy(), pd.concat), so we must provide DataFrames.
        cls.X_iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        cls.y_iris = iris.target

        cls.X_cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        cls.y_cancer = cancer.target

        cls.X_list = [cls.X_iris_df, cls.X_cancer_df]
        cls.y_list = [cls.y_iris, cls.y_cancer]

    def test_fit_and_estimate_with_single_scorer(self):
        """
        Test fitting the meta-regressor with noise and estimating performance.
        """
        model = LogisticRegression(max_iter=2000)

        # Use n_splits=2 and a small noise range for a fast test
        evaluator = RegressionNoiseEvaluator(
            model=model, scorer=accuracy_score, n_splits=2, verbose=False
        )

        # Fit the meta-regressors
        evaluator.fit(
            self.X_list, self.y_list, start_noise=10, end_noise=30, step_noise=10
        )

        # Per the example workflow, train a final model manually
        # Here we train on cancer data
        final_model = LogisticRegression(max_iter=2000).fit(
            self.X_cancer_df, self.y_cancer
        )
        evaluator.model = final_model

        # Estimate performance on the same dataset
        estimated_scores = evaluator.estimate(self.X_cancer_df)

        self.assertIn("score", estimated_scores)
        self.assertIsInstance(estimated_scores["score"], float)
        self.assertGreaterEqual(estimated_scores["score"], 0.0)

    def test_fit_and_estimate_with_multiple_scorers(self):
        """
        Test fit/estimate with multiple scorers.
        """
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        scorers = {
            "accuracy": accuracy_score,
            "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
        }

        evaluator = RegressionNoiseEvaluator(
            model=model, scorer=scorers, n_splits=2, verbose=False
        )

        # Fit the meta-regressors
        evaluator.fit(
            self.X_list, self.y_list, start_noise=10, end_noise=20, step_noise=10
        )

        # The .fit() method automatically fits self.model on X_list[0] (Iris)
        # We can use that fitted model to estimate on the Iris data.
        estimated_scores = evaluator.estimate(self.X_iris_df)

        self.assertIn("accuracy", estimated_scores)
        self.assertIn("f1_macro", estimated_scores)
        for v in estimated_scores.values():
            self.assertIsInstance(v, float)

    def test_estimate_without_fit_raises_error(self):
        """
        Test that estimate() raises NotFittedError if the main model (self.model)
        has not been fitted, even if the meta-regressors haven't been fitted.
        """
        model = LogisticRegression()
        evaluator = RegressionNoiseEvaluator(model=model, scorer=accuracy_score)

        # This should fail because self.model is not fitted
        with self.assertRaises(NotFittedError):
            evaluator.estimate(self.X_iris_df)

    def test_extract_metafeatures_shape(self):
        """
        Test the inherited _extract_metafeatures method shape.
        """
        model = LogisticRegression(max_iter=500)
        model.fit(self.X_iris_df, self.y_iris)

        evaluator = RegressionNoiseEvaluator(model=model)

        # Note: _extract_metafeatures handles both DataFrame and NumPy array inputs
        feats = evaluator._extract_metafeatures(model, self.X_iris_df)
        self.assertEqual(
            feats.shape, (1, 4)
        )  # mean_conf, std_conf, mean_entropy, std_entropy

    def test_fit_with_invalid_noise_params_raises_error(self):
        """
        Test that fit() raises ValueError for invalid noise parameters.
        """
        model = LogisticRegression()
        evaluator = RegressionNoiseEvaluator(model=model)

        # Test: start_noise < 0
        with self.assertRaises(ValueError):
            evaluator.fit(self.X_list, self.y_list, start_noise=-10)

        # Test: end_noise > 100
        with self.assertRaises(ValueError):
            evaluator.fit(self.X_list, self.y_list, end_noise=110)

        # Test: step_noise <= 0
        with self.assertRaises(ValueError):
            evaluator.fit(self.X_list, self.y_list, step_noise=0)


if __name__ == "__main__":
    unittest.main()
