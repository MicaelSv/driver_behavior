import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Função para salvar os resultados em um arquivo JSON
def save_to_json(metrics, name):
    with open(name, 'w') as f:
        json.dump(metrics, f)

# Função para calcular as métricas de avaliação
def evaluate_metrics(y_test, y_pred):
    return {
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'precision': metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'y_test': y_test.tolist(),
        'y_pred': y_pred.tolist()
    }

# Função principal para avaliar o modelo LSTM
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
        'AGGRESSIVE': 0,
        'NORMAL': 1,
        'SLOW': 2
    })
    
    # Separando features e rótulos
    X = df[FEATURES].values
    y = df['Class'].values
    
    # Função para criar sequências temporais
    def create_sequences(X, y, seq_length):
        """
        Cria sequências de dados para LSTM
        """
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - seq_length + 1):
            X_seq.append(X[i:i + seq_length])
            # Usar o label da maioria no intervalo da sequência
            sequence_labels = y[i:i + seq_length]
            majority_label = np.bincount(sequence_labels).argmax()
            y_seq.append(majority_label)
        return np.array(X_seq), np.array(y_seq)
    
    # Criar sequências temporais
    X_seq, y_seq = create_sequences(X, y, sequence_length)
    
    # Dividindo os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, stratify=y_seq
    )
    
    # Normalizando os dados (reshape para 2D, normalizar, reshape de volta para 3D)
    n_samples_train = X_train.shape[0] * X_train.shape[1]
    n_samples_test = X_test.shape[0] * X_test.shape[1]
    n_features = X_train.shape[2]
    
    # Reshape para normalização
    X_train_2d = X_train.reshape(n_samples_train, n_features)
    X_test_2d = X_test.reshape(n_samples_test, n_features)
    
    # Normalizar
    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train_2d)
    X_test_2d = scaler.transform(X_test_2d)
    
    # Reshape de volta para 3D
    X_train = X_train_2d.reshape(X_train.shape[0], sequence_length, n_features)
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
        Dense(3, activation='softmax')
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
            verbose=1
        ),
        
        # Reduzir learning rate quando o modelo estagnar
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Salvar o melhor modelo
        ModelCheckpoint(
            'best_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]
    
    # Treinar o modelo
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
        shuffle=False
    )
    
    # Fazendo previsões
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Retornar métricas
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
        ('test_motion_data.csv', 'default', FEATURES_RAW),
        ('inf_w60_dx4_test_motion_data.csv', '60', FEATURES_INF),
        ('inf_w120_dx4_test_motion_data.csv', '120', FEATURES_INF),
        ('inf_w180_dx4_test_motion_data.csv', '180', FEATURES_INF),
        ('inf_w240_dx4_test_motion_data.csv', '240', FEATURES_INF),
    ]

    lstm_metrics_dict = {data_name: [] for _, data_name, _ in datasets}

    # Avaliando os datasets um por um
    for train_path, data_name, FEATURES in datasets:
        train_path = 'dataset/' + train_path
        print('Dataset:', data_name)
        for i in range(5):
            print(f'\tRepetição #{i}')
            result = evaluate_lstm(train_path, FEATURES)
            lstm_metrics_dict[data_name].append(result)

    # Salvando os resultados em um arquivo JSON
    save_to_json(lstm_metrics_dict, 'metrics_results_lstm.json')
