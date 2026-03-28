import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

RANDOM_STATE = 42

# Função para salvar os resultados em um arquivo JSON
def save_to_json(metrics, name):
    with open(name, 'w') as f:
        json.dump(metrics, f)


def get_inf_feature_time(data_path):
    time_file = f"{data_path}.time"
    if not os.path.exists(time_file):
        return 0.0

    time_df = pd.read_csv(time_file)
    return float(time_df['time_hc'].sum() + time_df['time_fs'].sum())


def evaluate_metrics(y_test, y_pred):
    return {
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'precision': metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist()
    }

def evaluate_lstm(data_path, FEATURES, sequence_length=50):
    """
    Avalia um modelo LSTM para classificação binária de comportamento (Agressivo/Normal)
    
    Args:
        data_path: Caminho para o arquivo CSV
        FEATURES: Lista de features a serem utilizadas
        sequence_length: Número de time steps para criar sequências
    """
    # Carregando o dataset
    df = pd.read_csv(data_path)
    
    # Mapear as classes para valores binários
    df['Class'] = df['Class'].map({
        'Sudden-Acceleration': 0, 
        'Sudden-Right-Turn': 1, 
        'Sudden-Left-Turn': 2, 
        'Sudden-Break': 3    
    })
    
    # Separando features e rótulos
    X = df[FEATURES].values
    y = df['Class'].values
    
    # Função para criar sequências temporais
    def create_sequences(X, y, seq_length):
        X_seq = []
        y_seq = []
        for i in range(len(X) - seq_length + 1):
            X_seq.append(X[i:i + seq_length])
            sequence_labels = y[i:i + seq_length]
            majority_label = np.bincount(sequence_labels).argmax()
            y_seq.append(majority_label)
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(X, y, sequence_length)

    # Divisão treino (70%) / validação (15%) / teste (15%)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_seq, y_seq, test_size=0.15, stratify=y_seq, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.176, stratify=y_tv, random_state=RANDOM_STATE
    )

    # Normalizando os dados (reshape para 2D, normalizar, reshape de volta para 3D)
    n_samples_train = X_train.shape[0] * X_train.shape[1]
    n_samples_val = X_val.shape[0] * X_val.shape[1]
    n_samples_test = X_test.shape[0] * X_test.shape[1]
    n_features = X_train.shape[2]

    X_train_2d = X_train.reshape(n_samples_train, n_features)
    X_val_2d = X_val.reshape(n_samples_val, n_features)
    X_test_2d = X_test.reshape(n_samples_test, n_features)

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_val_2d = scaler.transform(X_val_2d)
    X_test_2d = scaler.transform(X_test_2d)

    X_train = X_train_2d.reshape(X_train.shape[0], sequence_length, n_features)
    X_val = X_val_2d.reshape(X_val.shape[0], sequence_length, n_features)
    X_test = X_test_2d.reshape(X_test.shape[0], sequence_length, n_features)
    
    # Construindo o modelo LSTM com arquitetura otimizada
    model = Sequential([
        # Camada de entrada
        Input(shape=(sequence_length, n_features)),
        
        # # Primeira camada LSTM
        LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        
        # Segunda camada LSTM
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        
        # Camada de saída
        Dense(4, activation='softmax')
    ])
    
    # Compilar o modelo com otimizador e learning rate customizados
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        ]
    )
    
    # Callbacks para melhor treinamento
    callbacks = [
        # Early stopping com monitoramento da perda de validação
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
        
        # Reduzir learning rate quando o modelo estagnar
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        ),
        
        # Salvar o melhor modelo
        ModelCheckpoint(
            'best_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )
    ]
    
    start_train = time.time()
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
        shuffle=False
    )
    end_train = time.time()

    start_test = time.time()
    y_pred_prob = model.predict(X_test, verbose=0)
    y_val_prob = model.predict(X_val, verbose=0)
    end_test = time.time()
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_val_pred = np.argmax(y_val_prob, axis=1)

    feature_time = get_inf_feature_time(data_path)

    # Retornar métricas
    results = evaluate_metrics(y_test, y_pred)
    results['val_accuracy'] = metrics.accuracy_score(y_val, y_val_pred)
    results['train_time'] = end_train - start_train
    results['test_time'] = end_test - start_test
    results['feature_extraction_time'] = feature_time
    results['total_pipeline_time'] = feature_time + results['train_time'] + results['test_time']
    return results


# Executando o código principal
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
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
        ('obd_data.csv', 'default', FEATURES_RAW),
        ('inf_w60_dx4_obd_data.csv', '60', FEATURES_INF),
        ('inf_w120_dx4_obd_data.csv', '120', FEATURES_INF),
        ('inf_w180_dx4_obd_data.csv', '180', FEATURES_INF),
        ('inf_w240_dx4_obd_data.csv', '240', FEATURES_INF),
    ]

    lstm_metrics_dict = {data_name: [] for _, data_name, _ in datasets}

    # Avaliando os datasets um por um
    num_repetition = 5
    for train_path, data_name, FEATURES in datasets:
        train_path = os.path.join(base_dir, 'dataset', train_path)
        for i in range(num_repetition):
            print(f'Dataset: {data_name}, Repetição {i}/{num_repetition}')
            result = evaluate_lstm(train_path, FEATURES)
            lstm_metrics_dict[data_name].append(result)

    # Salvando os resultados em um arquivo JSON
    save_to_json(lstm_metrics_dict, 'metrics_results_lstm.json')
