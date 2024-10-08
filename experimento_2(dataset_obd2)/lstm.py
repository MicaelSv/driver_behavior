import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
from keras.callbacks import EarlyStopping

# Função para salvar os resultados em um arquivo JSON
def save_to_json(metrics, name):
    with open(name, 'w') as f:
        json.dump(metrics, f, indent=4)

# Função para calcular as métricas de avaliação
def evaluate_metrics(y_test, y_pred):
    return {
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'roc_auc': metrics.roc_auc_score(y_test, y_pred), 
        'precision': metrics.precision_score(y_test, y_pred),
        'recall': metrics.recall_score(y_test, y_pred),
    }

# Função principal para avaliar o modelo LSTM
def evaluate_lstm(data_path, FEATURES):
    # Carregando o dataset
    df = pd.read_csv(data_path)

    # Mapear as classes para valores binários
    df['Class'] = df['Class'].map({
        'Aggressive': 0, 
        'Normal': 1       
    })

    # Separando features e rótulos
    X = df[FEATURES]
    y = df['Class']

    # Dividindo os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalizando os dados
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Remodelando os dados para o formato esperado pelo LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Construindo o modelo LSTM
    model = Sequential()

    # Camada LSTM inicial com 6 unidades e dropout
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

    model.add(LSTM(6, dropout=0.2, return_sequences=True))

    model.add(LSTM(64, return_sequences=True, dropout=0.2))
    model.add(LSTM(128, return_sequences=True, dropout=0.3))
    model.add(LSTM(256, return_sequences=False, dropout=0.4))

    # Camada densa intermediária com 256 unidades
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))

    # Camada densa final para classificação binária
    model.add(Dense(1, activation='sigmoid'))  # Para classificação binária

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Função de perda para classificação binária
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Definindo EarlyStopping para evitar overfitting
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=20, batch_size=1, validation_split=0.2, verbose=0)

    # Fazendo previsões
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convertendo para classes binárias (0 ou 1)

    # Avaliando as métricas
    return evaluate_metrics(y_test, y_pred)


# Executando o código principal
if __name__ == '__main__':
    FEATURES_RAW = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
    FEATURES_INF = [
        'AccX_entropy', 'AccX_complexity', 'AccX_fisher',
        'AccY_entropy', 'AccY_complexity', 'AccY_fisher',
        'AccZ_entropy', 'AccZ_complexity', 'AccZ_fisher',
        'GyroX_entropy', 'GyroX_complexity', 'GyroX_fisher',
        'GyroY_entropy', 'GyroY_complexity', 'GyroY_fisher',
        'GyroZ_entropy', 'GyroZ_complexity', 'GyroZ_fisher'
    ]

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

    lstm_metrics_dict = {data_name: [] for _, data_name, _ in datasets}

    # Avaliando os datasets um por um
    for train_path, data_name, FEATURES in datasets:
        for _ in range(5):
            result = evaluate_lstm(train_path, FEATURES)
            lstm_metrics_dict[data_name].append(result)

    # Salvando os resultados em um arquivo JSON
    save_to_json(lstm_metrics_dict, 'lstm_metrics_results_binaria.json')
