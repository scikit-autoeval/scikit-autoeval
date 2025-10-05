import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from skeval.evaluators.confidence import ConfidenceThresholdEvaluator

# ======================
# 1. Carregar datasets
# ======================
df_geriatria = pd.read_csv('./skeval/datasets/geriatria-controle-alzheimerLabel.csv')
df_neurologia = pd.read_csv('./skeval/datasets/neurologia-controle-alzheimerLabel.csv')

# ======================
# 2. Separar features e target
# ======================
X1, y1 = df_geriatria.drop(columns=['Alzheimer']), df_geriatria['Alzheimer']
X2, y2 = df_neurologia.drop(columns=['Alzheimer']), df_neurologia['Alzheimer']

# ======================
# 3. Definir pipeline (KNNImputer + RandomForestClassifier)
# ======================
model = make_pipeline(
    KNNImputer(n_neighbors=5),
    RandomForestClassifier(n_estimators=300, random_state=42)
)

# ======================
# 4. Instanciar avaliador baseado em confiança
# ======================
scorers = {
    'accuracy': accuracy_score,
    'f1_macro': lambda y, p: f1_score(y, p, average='macro')
}

evaluator = ConfidenceThresholdEvaluator(
    model=model,
    scorer=scorers,
    threshold=0.6,
    verbose=False
)

# ======================
# 5. Divisão treino/teste no dataset de geriatria
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.5, random_state=42, stratify=y1
)

# ======================
# 6. Treinar modelo e avaliador
# ======================
evaluator.fit(X_train, y_train)

# ======================
# 7. Estimar desempenho no conjunto de teste
# ======================
estimated_scores = evaluator.estimate(X_test)

# ======================
# 8. Calcular desempenho real
# ======================
y_pred = model.fit(X_train, y_train).predict(X_test)
real_scores = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_macro': f1_score(y_test, y_pred, average='macro')
}

# ======================
# 9. Comparação lado a lado
# ======================
print("\nComparação entre resultados reais e estimados:")
for metric in scorers.keys():
    print(f"{metric} -> Real: {real_scores[metric]:.4f} | Estimado: {estimated_scores[metric]:.4f}")
