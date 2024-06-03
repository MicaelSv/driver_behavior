import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from scipy.stats import sem
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import json

def save_to_json(metrics, name):
    with open(name, 'w') as f:
        json.dump(metrics, f, indent=4)

def build_scores_arrays(metrics_dict):
    accuracy_mean_list, accuracy_sem_list = [], []
    roc_auc_mean_list, roc_auc_sem_list = [], []
    precision_mean_list, precision_sem_list = [], []
    for data_name in metrics_dict.keys():
        accuracy_arr = [score['accuracy'] for score in metrics_dict[data_name]]
        roc_auc_arr = [score['roc_auc'] for score in metrics_dict[data_name]]
        precision_arr = [score['precision'] for score in metrics_dict[data_name]]
        accuracy_mean_list.append(np.mean(accuracy_arr))
        accuracy_sem_list.append(sem(accuracy_arr))
        roc_auc_mean_list.append(np.mean(roc_auc_arr))
        roc_auc_sem_list.append(sem(roc_auc_arr))
        precision_mean_list.append(np.mean(precision_arr))
        precision_sem_list.append(sem(precision_arr))
    return {
        'accuracy': {'mean': accuracy_mean_list, 'sem': accuracy_sem_list},
        'roc_auc': {'mean': roc_auc_mean_list, 'sem': roc_auc_sem_list},
        'precision': {'mean': precision_mean_list, 'sem': precision_sem_list},
    }

def evaluate_metrics(y_test, y_pred):
    return {
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'roc_auc': metrics.roc_auc_score(y_test, y_pred),
        'precision': metrics.precision_score(y_test, y_pred),
        'recall': metrics.recall_score(y_test, y_pred)
    }

def model_builder(input_shape):
    model = tf.keras.Sequential()

    lstm_units = [64, 128, 256]
    lstm_dropouts = [0.2, 0.3, 0.4]

    model.add(LSTM(units=6, dropout=0.2, return_sequences=True))

    for units, dropout in zip(lstm_units, lstm_dropouts):
        model.add(LSTM(units, dropout=dropout, return_sequences=True))

    model.add(LSTM(128))
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(Dense(1, activation='sigmoid'))

    learning_rate = 0.001

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model

def evaluate_lstm_helper(args):
    return evaluate_lstm(*args)

def evaluate_lstm(train_path, test_path, FEATURES):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train = df_train.drop(df_train[df_train['Class'] == 'SLOW'].index)
    df_test = df_test.drop(df_test[df_test['Class'] == 'SLOW'].index)

    df_train['Class'] = df_train['Class'].map({"AGGRESSIVE": 1, "NORMAL": 0, "SLOW": 0})
    df_test['Class'] = df_test['Class'].map({"AGGRESSIVE": 1, "NORMAL": 0, "SLOW": 0})

    X_train = df_train[FEATURES]
    X_test = df_test[FEATURES]

    y_train = df_train['Class']
    y_test = df_test['Class']

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    clf = model_builder(input_shape=(X_train.shape[1], X_train.shape[2]))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    clf.fit(X_train, y_train, epochs=50, batch_size=1, verbose=2, callbacks=[early_stopping], validation_split=0.2)

    y_pred = clf.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    return evaluate_metrics(y_test, y_pred)

def evaluate_random_forest_helper(args):
    return evaluate_random_forest(*args)

def evaluate_random_forest(train_path, test_path, FEATURES):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train = df_train.drop(df_train[df_train['Class'] == 'SLOW'].index)
    df_test = df_test.drop(df_test[df_test['Class'] == 'SLOW'].index)

    df_train['Class'] = df_train['Class'].map({"AGGRESSIVE": 0, "NORMAL": 1, "SLOW": 0})
    df_test['Class'] = df_test['Class'].map({"AGGRESSIVE": 0, "NORMAL": 1, "SLOW": 0})

    X_train = df_train[FEATURES]
    X_test = df_test[FEATURES]

    y_train = LabelEncoder().fit_transform(df_train['Class'])
    y_test = LabelEncoder().fit_transform(df_test['Class'])

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=50, min_samples_split=6, min_samples_leaf=3, max_features='sqrt', max_depth=10, bootstrap=True)
    clf.fit(X_train, y_train)
    

    y_pred = clf.predict(X_test)
    return evaluate_metrics(y_test, y_pred)

if __name__ == '__main__':
    FEATURES_RAW = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
    FEATURES_INF = ['AccX_entropy', 'AccX_complexity', 'AccY_entropy', 'AccY_complexity', 'AccZ_entropy', 'AccZ_complexity', 'GyroX_entropy', 'GyroX_complexity', 'GyroY_entropy', 'GyroY_complexity', 'GyroZ_entropy', 'GyroZ_complexity']

    datasets = [
        ('train_motion_data.csv', 'test_motion_data.csv', 'default', FEATURES_RAW),
        ('inf_w60_dx5_train_motion_data.csv', 'inf_w60_dx5_test_motion_data.csv', '60', FEATURES_INF),
        ('inf_w120_dx5_train_motion_data.csv', 'inf_w120_dx5_test_motion_data.csv', '120', FEATURES_INF),
        ('inf_w180_dx6_train_motion_data.csv', 'inf_w180_dx6_test_motion_data.csv', '180', FEATURES_INF),
        ('inf_w240_dx6_train_motion_data.csv', 'inf_w240_dx6_test_motion_data.csv', '240', FEATURES_INF),
        ('inf_w300_dx6_train_motion_data.csv', 'inf_w300_dx6_test_motion_data.csv', '300', FEATURES_INF),
        ('inf_w360_dx6_train_motion_data.csv', 'inf_w360_dx6_test_motion_data.csv', '360', FEATURES_INF),
        ('inf_w420_dx6_train_motion_data.csv', 'inf_w420_dx6_test_motion_data.csv', '420', FEATURES_INF),
        ('inf_w480_dx6_train_motion_data.csv', 'inf_w480_dx6_test_motion_data.csv', '480', FEATURES_INF),
        ('inf_w540_dx6_train_motion_data.csv', 'inf_w540_dx6_test_motion_data.csv', '540', FEATURES_INF),
        ('inf_w600_dx6_train_motion_data.csv', 'inf_w600_dx6_test_motion_data.csv', '600', FEATURES_INF),
        ('inf_w660_dx6_train_motion_data.csv', 'inf_w660_dx6_test_motion_data.csv', '660', FEATURES_INF),
        ('inf_w720_dx6_train_motion_data.csv', 'inf_w720_dx6_test_motion_data.csv', '720', FEATURES_INF),
        ('inf_w780_dx7_train_motion_data.csv', 'inf_w780_dx7_test_motion_data.csv', '780', FEATURES_INF)
    ]

    #criando dicionários para armazenar os resultados das avaliações dos modelos RF e LSTM
    rf_metrics_dict = {data_name: [] for _, _, data_name, _ in datasets}
    lstm_metrics_dict = {data_name: [] for _, _, data_name, _ in datasets}

    #aqui ficam os argumentos para as funções de avaliação, com múltiplas avaliações por conjunto de dados
    rf_args = [(train_path, test_path, FEATURES) for train_path, test_path, _, FEATURES in datasets for _ in range(28)]
    lstm_args = [(train_path, test_path, FEATURES) for train_path, test_path, _, FEATURES in datasets for _ in range(14)]

    #criando um pool de processos para permitir a execução paralela do treinamento dos modelos. 
    with Pool(cpu_count()) as p:    
        rf_results = p.map(evaluate_random_forest_helper, rf_args)
        lstm_results = p.map(evaluate_lstm_helper, lstm_args)

    #adicionando os resultados das métricas aos dicionários
    for (train_path, test_path, data_name, FEATURES), result in zip(datasets, rf_results):
        rf_metrics_dict[data_name].append(result)

    for (train_path, test_path, data_name, FEATURES), result in zip(datasets, lstm_results):
        lstm_metrics_dict[data_name].append(result)

    save_to_json(lstm_metrics_dict, 'lstm_metrics_results.json')
    save_to_json(rf_metrics_dict, 'rf_metrics_results.json')
