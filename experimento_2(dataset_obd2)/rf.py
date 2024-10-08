import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

import json

def save_to_json(metrics, name):
    with open(name, 'w') as f:
        json.dump(metrics, f, indent=4)


def evaluate_model(data_path, FEATURES):
    # Carregando o dataset
    df = pd.read_csv(data_path)

    # Mapeando as classes conforme solicitado
    df['Class'] = df['Class'].map({
        'Aggressive': 0, 
        'Normal': 1
    })

    # Dividindo os dados entre features e rótulos
    X = df[FEATURES]
    y = df['Class']

    # Aplicando o StandardScaler
    scaler = StandardScaler().fit(X)
    #X_scaled = scaler.transform(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalizando os dados
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Treinando o modelo Random Forest / no multiclasses, o autor utiliza 20 em n_estimators
    clf = RandomForestClassifier(n_estimators=50, min_samples_split=6, min_samples_leaf=3, max_features='sqrt', max_depth=5, bootstrap=True)
    clf.fit(X_train, y_train)

    # Fazendo as previsões
    y_pred = clf.predict(X_test)
    
    # Avaliando as métricas
    return evaluate_metrics(y_test, y_pred)


def evaluate_metrics(y_test, y_pred):
    return {
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'roc_auc': metrics.roc_auc_score(y_test, y_pred),
        'precision': metrics.precision_score(y_test, y_pred),
        'recall': metrics.recall_score(y_test, y_pred)
    }


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
    ('inf_w60_dx5_dataset_obd2.csv', '60', FEATURES_INF),
    ('inf_w120_dx5_dataset_obd2.csv', '120', FEATURES_INF),
    ('inf_w180_dx6_dataset_obd2.csv', '180', FEATURES_INF),
    ('inf_w240_dx6_dataset_obd2.csv', '240', FEATURES_INF),
    ('inf_w300_dx6_dataset_obd2.csv', '300', FEATURES_INF),
    ('inf_w360_dx6_dataset_obd2.csv', '360', FEATURES_INF),
    ('inf_w420_dx6_dataset_obd2.csv', '420', FEATURES_INF),
    ('inf_w480_dx6_dataset_obd2.csv', '480', FEATURES_INF),
    ('inf_w540_dx6_dataset_obd2.csv', '540', FEATURES_INF),
    ('inf_w600_dx6_dataset_obd2.csv', '600', FEATURES_INF),
    ('inf_w660_dx6_dataset_obd2.csv', '660', FEATURES_INF),
    ('inf_w720_dx6_dataset_obd2.csv', '720', FEATURES_INF),
    ('inf_w780_dx7_dataset_obd2.csv', '780', FEATURES_INF)
]

    # Dicionário para armazenar as métricas do Random Forest
    rf_metrics_dict = {data_name: [] for _, data_name, _ in datasets}

    # Avaliando os datasets um por um
    for train_path, data_name, FEATURES in datasets:
        for _ in range(10):
            result = evaluate_model(train_path, FEATURES)
            rf_metrics_dict[data_name].append(result)

    # Salvando as métricas em arquivo JSON
    save_to_json(rf_metrics_dict, 'rf_definitive_metrics_results.json')