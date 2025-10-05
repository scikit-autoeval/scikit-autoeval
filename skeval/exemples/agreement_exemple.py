import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from ..evaluators.agreement import AgreementEvaluator
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Carregar datasets
df_geriatria = pd.read_csv('./skeval/datasets/geriatria-controle-alzheimerLabel.csv')
df_neurologia = pd.read_csv('./skeval/datasets/neurologia-controle-alzheimerLabel.csv')

# Coluna alvo
target_col = 'Alzheimer'

def preprocess(df):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Converter categóricas para numéricas
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Preencher NaNs com a média da coluna
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    return X, y.values

# Preprocessar datasets
X_ger, y_ger = preprocess(df_geriatria)
X_neu, y_neu = preprocess(df_neurologia)

# Inicializar o AgreementEvaluator
main_model = GaussianNB()
evaluator = AgreementEvaluator(model=main_model, verbose=1)

# ===== FIT GERIATRIA =====
print("===== FIT GERIATRIA =====")
evaluator.fit(X_ger, y_ger)

# Métrica real: acurácia do modelo principal
main_model.fit(X_ger, y_ger)
y_pred_main = main_model.predict(X_ger)
accuracy_real_ger = accuracy_score(y_ger, y_pred_main)
print("Real accuracy (geriatria):", accuracy_real_ger)

# Estimativa pelo AgreementEvaluator
agreement_score_ger = evaluator.estimate(X_ger)
print("Estimated agreement score (geriatria):", agreement_score_ger)

# Comparação direta
print(f"Difference (real - estimated): {accuracy_real_ger - agreement_score_ger}")

# ===== FIT NEUROLOGIA =====
print("\n===== FIT NEUROLOGIA =====")
evaluator.fit(X_neu, y_neu)

# Métrica real
main_model.fit(X_neu, y_neu)
y_pred_main = main_model.predict(X_neu)
accuracy_real_neu = accuracy_score(y_neu, y_pred_main)
print("Real accuracy (neurologia):", accuracy_real_neu)

# Estimativa pelo AgreementEvaluator
agreement_score_neu = evaluator.estimate(X_neu)
print("Estimated agreement score (neurologia):", agreement_score_neu)

# Comparação direta
print(f"Difference (real - estimated): {accuracy_real_neu - agreement_score_neu}")
