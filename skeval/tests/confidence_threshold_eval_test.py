from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from evaluators.confidence import ConfidenceThresholdEvaluator
from metrics.scorers import make_scorer

# TESTE 1
X, y = make_classification(n_samples=200, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
model = LogisticRegression()
model2 = RandomForestClassifier()
threshold = 0.7
evaluator = ConfidenceThresholdEvaluator(
    estimator=model,
    scorer={'acc': accuracy_score, 'f1': f1_score},
    threshold=threshold
)

evaluator2 = ConfidenceThresholdEvaluator(
    estimator=model2,
    scorer={'acc': accuracy_score, 'f1': f1_score},
    threshold=threshold
)

evaluator.fit(X_train, y_train)
result = evaluator.estimate(X_test)
print("modelo 1", result)

evaluator2.fit(X_train, y_train)
result2 = evaluator2.estimate(X_test)
print("modelo 2", result2)

proba = model.predict_proba(X_test)
confidence = proba.max(axis=1)
mask = confidence >= threshold

# TESTE 2
X, y = make_classification(n_samples=150, n_features=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
f1_macro_scorer = make_scorer(f1_score, average='macro')
threshold = 0.6

evaluator = ConfidenceThresholdEvaluator(
    estimator=model,
    scorer=f1_macro_scorer,
    threshold=threshold
)

evaluator.fit(X_train, y_train)
result = evaluator.estimate(X_test)
print(result)

proba = model.predict_proba(X_test)
confidence = proba.max(axis=1)
mask = confidence >= threshold