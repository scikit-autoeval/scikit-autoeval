import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from evaluators.confidence import ConfidenceThresholdEvaluator
from metrics.comparison import score_error

def load_dataset(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Alzheimer"])
    y = df["Alzheimer"]
    return X, y

X_geri, y_geri = load_dataset("datasets/geriatria-controle-alzheimerLabel.csv")
X_neuro, y_neuro = load_dataset("datasets/neurologia-controle-alzheimerLabel.csv")

model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000))
evaluator = ConfidenceThresholdEvaluator(
    estimator=model,
    scorer={"acc": accuracy_score, "mse": mean_squared_error},
    threshold=0.7)

def run_score_error_test(X_train, y_train, X_test, y_test, label):
    evaluator.fit(X_train, y_train)
    
    # Predição real
    y_pred = evaluator.estimator.predict(X_test)
    real_score = {
        "acc": accuracy_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred)
    }

    # Estimativa
    estimated_score = evaluator.estimate(X_test)

    print(f"=== {label} ===")
    score_error(
        real_scores=real_score,
        estimated_scores=estimated_score,
        comparator={"acc": mean_absolute_error, "mse": mean_squared_error},
        verbose=True
    )

    print()

if __name__ == "__main__":
    run_score_error_test(X_geri, y_geri, X_neuro, y_neuro, "Geriatria → Neurologia")
    run_score_error_test(X_neuro, y_neuro, X_geri, y_geri, "Neurologia → Geriatria")
