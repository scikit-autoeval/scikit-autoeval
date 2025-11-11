# test_regression_evaluator.py
# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from sklearn.datasets import load_iris, load_breast_cancer, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import NotFittedError

from skeval.evaluators.regression import RegressionEvaluator

class TestRegressionBasedEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        iris = load_iris()
        cancer = load_breast_cancer()
        cls.X_list = [iris.data, cancer.data]
        cls.y_list = [iris.target, cancer.target]

    def test_fit_and_estimate_with_single_scorer(self):
        model = LogisticRegression(max_iter=1000)
        evaluator = RegressionEvaluator(model=model, scorer=accuracy_score, n_splits=3, verbose=False)
        evaluator.fit(self.X_list, self.y_list)

        final_model = LogisticRegression(max_iter=1000).fit(self.X_list[1], self.y_list[1])
        evaluator.model = final_model

        estimated_scores = evaluator.estimate(self.X_list[1])
        self.assertIn("score", estimated_scores)
        self.assertIsInstance(estimated_scores["score"], float)

    def test_fit_and_estimate_with_multiple_scorers(self):
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scorers = {
            "accuracy": accuracy_score,
            "f1_macro": lambda y, p: f1_score(y, p, average="macro")
        }
        evaluator = RegressionEvaluator(model=model, scorer=scorers, n_splits=2, verbose=False)
        evaluator.fit(self.X_list, self.y_list)

        final_model = RandomForestClassifier(n_estimators=50, random_state=42).fit(self.X_list[0], self.y_list[0])
        evaluator.model = final_model

        estimated_scores = evaluator.estimate(self.X_list[0])
        self.assertIn("accuracy", estimated_scores)
        self.assertIn("f1_macro", estimated_scores)
        for v in estimated_scores.values():
            self.assertIsInstance(v, float)

    def test_estimate_without_fit_raises_error(self):
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = LogisticRegression(max_iter=200)
        evaluator = RegressionEvaluator(model=model, scorer=accuracy_score)

        with self.assertRaises(NotFittedError):
            evaluator.estimate(X)

    def test_extract_metafeatures_shape(self):
        model = LogisticRegression(max_iter=500)
        model.fit(self.X_list[0], self.y_list[0])

        evaluator = RegressionEvaluator(model=model)
        feats = evaluator._extract_metafeatures(model, self.X_list[0])
        self.assertEqual(feats.shape, (1, 4))  # mean_conf, std_conf, mean_entropy, std_entropy


if __name__ == "__main__":
    unittest.main()
