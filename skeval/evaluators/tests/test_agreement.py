# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import NotFittedError

# Assuming the AgreementEvaluator is in this path
from skeval.evaluators.agreement import AgreementEvaluator

class TestAgreementEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load datasets for fitting and estimating
        cls.X_iris, cls.y_iris = load_iris(return_X_y=True)
        cls.X_cancer, cls.y_cancer = load_breast_cancer(return_X_y=True)

    def test_fit_and_estimate_with_single_scorer(self):
        """
        Test fitting and estimating with a single scorer (default accuracy).
        Uses default GaussianNB as the secondary model.
        """
        model = LogisticRegression(max_iter=1000)
        
        # Initialize evaluator, sec_model defaults to GaussianNB
        evaluator = AgreementEvaluator(model=model, scorer=accuracy_score, verbose=False)
        
        # Fit on one dataset (Cancer)
        evaluator.fit(self.X_cancer, self.y_cancer)

        # Estimate on the SAME dataset (Cancer)
        # The error was here: using X_cancer (30 features) after fitting on X_iris (4 features)
        estimated_score = evaluator.estimate(self.X_cancer)
        
        self.assertIsInstance(estimated_score, float)
        self.assertGreaterEqual(estimated_score, 0.0)
        self.assertLessEqual(estimated_score, 1.0)

    def test_fit_and_estimate_with_multiple_scorers(self):
        """
        Test fitting and estimating with a dictionary of multiple scorers
        and a custom secondary model.
        """
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        sec_model = DecisionTreeClassifier(random_state=42)
        
        scorers = {
            "accuracy": accuracy_score,
            "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro", zero_division=0)
        }
        
        evaluator = AgreementEvaluator(
            model=model, 
            scorer=scorers, 
            sec_model=sec_model, 
            verbose=False
        )
        
        # Fit on iris data
        evaluator.fit(self.X_iris, self.y_iris)

        # Estimate on the SAME dataset (Iris)
        # The error was here: using X_cancer (30 features) after fitting on X_iris (4 features)
        estimated_scores = evaluator.estimate(self.X_iris)
        
        # Check that scores are a dictionary with the correct keys
        self.assertIsInstance(estimated_scores, dict)
        self.assertIn("accuracy", estimated_scores)
        self.assertIn("f1_macro", estimated_scores)
        
        # Check that all scores are floats
        for score_name, score_value in estimated_scores.items():
            self.assertIsInstance(score_value, float)
            self.assertGreaterEqual(score_value, 0.0, f"{score_name} was not >= 0")
            self.assertLessEqual(score_value, 1.0, f"{score_name} was not <= 1")

    def test_estimate_without_fit_raises_error(self):
        """
        Test that calling estimate() before fit() raises a NotFittedError.
        """
        model = LogisticRegression()
        sec_model = GaussianNB()
        
        evaluator = AgreementEvaluator(model=model, sec_model=sec_model)

        # Expect a NotFittedError because neither model has been fitted
        with self.assertRaises(NotFittedError):
            evaluator.estimate(self.X_iris)
            
    def test_partial_fit_raises_error(self):
        """
        Test that NotFittedError is raised if only one model is fitted.
        This tests the internal check_is_fitted calls.
        """
        model = LogisticRegression().fit(self.X_iris, self.y_iris) # Fit main model
        sec_model = GaussianNB() # Do not fit secondary model
        
        evaluator = AgreementEvaluator(model=model, sec_model=sec_model)
        
        # We manually assigned a fitted model, but sec_model is not fitted
        # The evaluator's fit() method was not called, but estimate()
        # checks both models internally.
        with self.assertRaises(NotFittedError):
            evaluator.estimate(self.X_cancer)

    def test_default_secondary_model_instance(self):
        """
        Test that the secondary model defaults to GaussianNB if not provided.
        """
        model = LogisticRegression()
        evaluator = AgreementEvaluator(model=model)
        
        self.assertIsInstance(evaluator.sec_model, GaussianNB)


if __name__ == "__main__":
    unittest.main()