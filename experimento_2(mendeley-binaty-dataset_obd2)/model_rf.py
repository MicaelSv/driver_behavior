import time
import json
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import json

def save_to_json(metrics, name):
    with open(name, 'w') as f:
        json.dump(metrics, f)


def evaluate_metrics(y_test, y_pred):
    return {
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'roc_auc': metrics.roc_auc_score(y_test, y_pred),
        'precision': metrics.precision_score(y_test, y_pred),
        'recall': metrics.recall_score(y_test, y_pred),
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist()
    }


def evaluate_model(data_path, FEATURES):
    # Carregando o dataset
    df = pd.read_csv(data_path)

    # Mapeando as classes conforme solicitado
    df['Class'] = df['Class'].map({
        'Aggressive': 1, 
        'Normal': 0
    })

    # Dividindo os dados entre features e rótulos
    X = df[FEATURES]
    y = df['Class']

    # Aplicando o StandardScaler
    scaler = StandardScaler().fit(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    # Normalizando os dados
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Treinando o modelo Random Forest / no multiclasses, o autor utiliza 20 em n_estimators
    clf = RandomForestClassifier(n_estimators=50, min_samples_split=6, min_samples_leaf=3, max_features='sqrt', max_depth=10, bootstrap=True)
    start_train = time.time()
    clf.fit(X_train, y_train)
    end_train = time.time()

    # Fazendo as previsões
    start_test = time.time()
    y_pred = clf.predict(X_test)
    end_test = time.time()
    
    # Avaliando as métricas
    results = evaluate_metrics(y_test, y_pred)
    results['train_time'] = end_train - start_train
    results['test_time'] = end_test - start_test
    return results


if __name__ == '__main__':
    # Definindo as features para cada tipo de dataset
    FEATURES_RAW = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
    FEATURES_INF = [
        'AccX_entropy', 'AccX_complexity', 'AccX_fisher',
        'AccY_entropy', 'AccY_complexity', 'AccY_fisher',
        'AccZ_entropy', 'AccZ_complexity', 'AccZ_fisher',
        'GyroX_entropy', 'GyroX_complexity', 'GyroX_fisher',
        'GyroY_entropy', 'GyroY_complexity', 'GyroY_fisher',
        'GyroZ_entropy', 'GyroZ_complexity', 'GyroZ_fisher'
    ]

    # Lista de datasets com diferentes janelas e variáveis de interesse
    datasets = [
        ('dataset_obd2.csv', 'default', FEATURES_RAW),
        ('inf_w60_dx4_dataset_obd2.csv', '60', FEATURES_INF),
        ('inf_w120_dx4_dataset_obd2.csv', '120', FEATURES_INF),
        ('inf_w180_dx4_dataset_obd2.csv', '180', FEATURES_INF),
        ('inf_w240_dx4_dataset_obd2.csv', '240', FEATURES_INF)
    ]
    
    # Dicionário para armazenar as métricas do Random Forest
    rf_metrics_dict = {data_name: [] for _, data_name, _ in datasets}

    # Avaliando os datasets um por um
    num_repetition = 5
    for train_path, data_name, FEATURES in datasets:
        train_path = 'dataset/' + train_path
        for i in range(num_repetition):
            print(f'Dataset: {data_name}, Repetição {i}/{num_repetition}')
            result = evaluate_model(train_path, FEATURES)
            rf_metrics_dict[data_name].append(result)

    # Salvando as métricas em arquivo JSON
    save_to_json(rf_metrics_dict, 'metrics_results_rf.json')
