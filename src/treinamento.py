import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Caminhos relativos
DATA_PATH = 'data/dataset_libras.csv'
MODEL_SAVE_PATH = 'models/modelo_rf_libras.pkl'



def load_and_prepare_data():
    if not os.path.exists(DATA_PATH):
        print(f"[ERRO] Dataset não encontrado em: {DATA_PATH}")
        return None, None, None, None

    print("[INFO] Carregando o dataset...")
    df = pd.read_csv(DATA_PATH)

    # Verifica quantas amostras de cada letra você coletou
    print("\n[INFO] Distribuição das amostras por classe:")
    print(df['label'].value_counts())

    # Separa Features (X) e Labels (y)
    # y é a coluna 'label', X são todas as outras colunas (x0, y0, z0, x1...)
    y = df['label']
    X = df.drop('label', axis=1)

    # Divide os dados: 80% para treinar o modelo, 20% para testar se ele aprendeu mesmo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n[INFO] Iniciando o treinamento do modelo Random Forest...")
    
    # Instancia o classificador. n_estimators=100 significa 100 árvores de decisão.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # O comando .fit() é feito o treino
    model.fit(X_train, y_train)

    print("[INFO] Treinamento concluído. Avaliando o modelo...")
    
    # Pede para o modelo prever as classes usando os dados de teste que ele nunca viu
    y_pred = model.predict(X_test)

    # Calcula a precisão (0.0 a 1.0)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULTADO] Acurácia Geral do Modelo: {acc * 100:.2f}%")
    
    # Mostra um relatório detalhado de acertos e erros por cada letra
    print("\n[RESULTADO] Relatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred))

    return model


def save_model(model):
    # Garante que a pasta models/ exista
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"\n[SUCESSO] Modelo treinado e salvo em: {MODEL_SAVE_PATH}")

def main():
    print("=== Módulo de Treinamento de Machine Learning ===")
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    if X_train is not None:
        model = train_and_evaluate(X_train, X_test, y_train, y_test)
        save_model(model)

if __name__ == '__main__':
    main()