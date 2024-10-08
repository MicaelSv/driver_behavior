import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
import json

def save_to_json(metrics, name):
    with open(name, 'w') as f:
        json.dump(metrics, f, indent=4)

def evaluate_metrics(y_test, y_pred):
    return {
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'precision': metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)
    }

def evaluate_lstm(data_path, FEATURES):
    df = pd.read_csv(data_path)

    df['Class'] = df['Class'].map({
        'Sudden-Acceleration': 0, 
        'Sudden-Right-Turn': 1, 
        'Sudden-Left-Turn': 2, 
        'Sudden-Break': 3
    })

    X = df[FEATURES]
    y = df['Class']

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

    # Adicionando a camada de entrada
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

    # Primeira camada LSTM com dropout
    model.add(LSTM(64, return_sequences=True, dropout=0.2))

    # Outras camadas LSTM com dropout
    model.add(LSTM(128, return_sequences=True, dropout=0.3))
    model.add(LSTM(256, return_sequences=False, dropout=0.4))

    # Camada densa final para a classificação
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4, activation='softmax'))  # 4 classes no total

    # Compilando o modelo com a configuração multiclasse
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    return evaluate_metrics(y_test, y_pred_classes)

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

    # Lista de datasets com diferentes janelas e variáveis de interesse
    datasets = [
        ('obd_data.csv', 'default', FEATURES_RAW),
        ('inf_w60_dx5_obd_data.csv', '60', FEATURES_INF),
        ('inf_w120_dx5_obd_data.csv', '120', FEATURES_INF),
        ('inf_w180_dx6_obd_data.csv', '180', FEATURES_INF),
        ('inf_w240_dx6_obd_data.csv', '240', FEATURES_INF),
        ('inf_w300_dx6_obd_data.csv', '300', FEATURES_INF),
        ('inf_w360_dx6_obd_data.csv', '360', FEATURES_INF),
        ('inf_w420_dx6_obd_data.csv', '420', FEATURES_INF),
        ('inf_w480_dx6_obd_data.csv', '480', FEATURES_INF),
        ('inf_w540_dx6_obd_data.csv', '540', FEATURES_INF),
        ('inf_w600_dx6_obd_data.csv', '600', FEATURES_INF),
        ('inf_w660_dx6_obd_data.csv', '660', FEATURES_INF),
        ('inf_w720_dx6_obd_data.csv', '720', FEATURES_INF),
        ('inf_w780_dx7_obd_data.csv', '780', FEATURES_INF)
    ]

    # Dicionário para armazenar as métricas do LSTM
    lstm_metrics_dict = {data_name: [] for _, data_name, _ in datasets}

    # Avaliando os datasets um por um
    for train_path, data_name, FEATURES in datasets:
        for _ in range(5):  
            result = evaluate_lstm(train_path, FEATURES)
            lstm_metrics_dict[data_name].append(result)

    # Salvando as métricas em arquivo JSON
    save_to_json(lstm_metrics_dict, 'lstm_metrics_results_definitive.json')