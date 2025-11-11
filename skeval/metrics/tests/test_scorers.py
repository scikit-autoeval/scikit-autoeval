# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
import unittest
import numpy as np
from sklearn.metrics import f1_score, precision_score

from skeval.metrics.scorers import make_scorer

class TestMakeScorer(unittest.TestCase):

    def setUp(self):
        """Set up the test data."""
        self.y_true = np.array([0, 1, 2, 0, 1, 2])
        self.y_pred = np.array([0, 2, 1, 0, 0, 1])

    def test_f1_macro_scorer(self):
        """Test creating a scorer for F1-score with 'macro' averaging."""
        macro_f1_scorer = make_scorer(f1_score, average='macro')
        
        # Calculate the score using the created scorer
        score = macro_f1_scorer(self.y_true, self.y_pred)
        
        # Calculate the score directly for comparison
        direct_call_score = f1_score(self.y_true, self.y_pred, average='macro')
        
        self.assertAlmostEqual(score, direct_call_score)
        self.assertAlmostEqual(score, 0.26666666666666666)

    def test_precision_weighted_scorer(self):
        """Test creating a scorer for precision_score with 'weighted' averaging."""
        weighted_precision_scorer = make_scorer(precision_score, average='weighted', zero_division=0)
        
        # Calculate the score using the created scorer
        score = weighted_precision_scorer(self.y_true, self.y_pred)
        
        # Calculate the score directly for comparison
        direct_call_score = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        
        # Assert both calculated scores are equal to the correct, manually verified value
        self.assertAlmostEqual(score, direct_call_score)
        self.assertAlmostEqual(score, 0.2222222222222222) # This line is corrected

    def test_scorer_with_custom_function(self):
        """Test if make_scorer works with a user-defined function."""
        def custom_metric(y_true, y_pred, penalty=0):
            return np.sum(y_true == y_pred) / len(y_true) - penalty
            
        # Create a scorer with a fixed penalty argument
        penalized_scorer = make_scorer(custom_metric, penalty=0.1)
        
        score = penalized_scorer(self.y_true, self.y_pred)
        
        # Manual calculation: (2 hits / 6 total) - 0.1 penalty
        expected_score = (2 / 6) - 0.1
        
        self.assertAlmostEqual(score, expected_score)

    def test_scorer_with_no_kwargs(self):
        """Test creating a scorer with no extra keyword arguments."""
        from sklearn.metrics import accuracy_score
        
        accuracy_scorer = make_scorer(accuracy_score)
        
        score = accuracy_scorer(self.y_true, self.y_pred)
        direct_call_score = accuracy_score(self.y_true, self.y_pred)
        
        self.assertAlmostEqual(score, direct_call_score)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)