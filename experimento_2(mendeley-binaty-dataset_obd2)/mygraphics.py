import matplotlib.pyplot as plt
from scipy.stats import sem
import numpy as np
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

FONT_SIZE = 32

FIGDIRECTORY = "figures"
os.makedirs(FIGDIRECTORY, exist_ok=True)


def savefig(filename):
    path = os.path.join(FIGDIRECTORY, filename + '.svg')
    plt.savefig(path, format='svg')
    plt.close()


def get_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def build_scores_arrays(metrics_dict, model_name):
    accuracy_mean_list, accuracy_sem_list = [], []
    precision_mean_list, precision_sem_list = [], []
    recall_mean_list, recall_sem_list = [] , []
    
    accuracy_arr, precision_arr, recall_arr = [], [], []
    # Inicializa a matriz de confusão acumulada
    # Descobrimos o número de classes para criar a matriz
    all_classes = set()
    for data_name in metrics_dict.keys():
        for score in metrics_dict[data_name]:
            all_classes.update(score['y_test'])
    classes = sorted(all_classes)
    num_classes = len(classes)
    cm_sum = np.zeros((num_classes, num_classes), dtype=float)

    for data_name in metrics_dict.keys():
        for score in metrics_dict[data_name]:
            accuracy_arr.append(score['accuracy'])
            precision_arr.append(score['precision'])
            recall_arr.append(score['recall'])

            # Calcula a matriz de confusão para este score
            cm = confusion_matrix(score['y_test'], score['y_pred'], labels=classes, normalize='true')
            cm_sum = cm
            plot_confusion_matrix(cm, model_name, data_name)

        accuracy_mean_list.append(np.mean(accuracy_arr))
        accuracy_sem_list.append(sem(accuracy_arr))
        precision_mean_list.append(np.mean(precision_arr))
        precision_sem_list.append(sem(precision_arr))
        recall_mean_list.append(np.mean(recall_arr))
        recall_sem_list.append(sem(recall_arr))
    
    return {
        'accuracy': {'mean': accuracy_mean_list, 'sem': accuracy_sem_list},
        'precision': {'mean': precision_mean_list, 'sem': precision_sem_list},
        'recall': {'mean': recall_mean_list, 'sem': recall_sem_list},
        'cm_mean': cm_sum
    }

def plot_metrics(metrics_dict, model_name, loc=None):
    scores_dict = build_scores_arrays(metrics_dict, model_name)
    plt.figure(figsize=(18,10))
    X_names = metrics_dict.keys()
    X_axis = np.arange(len(X_names))
    
    plt.bar(X_axis - 0.25, scores_dict['accuracy']['mean'], 0.2, 
            yerr=scores_dict['accuracy']['sem'], label='Accuracy', edgecolor="black")
    plt.bar(X_axis, scores_dict['precision']['mean'], 0.2, 
            yerr=scores_dict['precision']['sem'], label='Precision', edgecolor="black")
    plt.bar(X_axis + 0.25, scores_dict['recall']['mean'], 0.2, 
            yerr=scores_dict['recall']['sem'], label='Recall', edgecolor="black") 
    
    plt.xticks(X_axis, X_names)
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.ylim((0.4, 1))
    plt.xlabel('Method', fontsize=FONT_SIZE)
    plt.ylabel('Score', fontsize=FONT_SIZE)
    if loc is None:
        plt.legend(fontsize=FONT_SIZE - 4, framealpha=1)
    else:
        plt.legend(fontsize=FONT_SIZE - 4, framealpha=1, loc=loc)
    savefig(f'graphic_{model_name}_mendeley_binary')
    return scores_dict


def plot_confusion_matrix(cm, model_name, dataname='', normalize=False, cmap='Blues'):
    # Cria o objeto de exibição
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plota a matriz
    disp.plot(cmap=cmap, values_format=".2f")

    # Personaliza o gráfico
    plt.tick_params(axis='both', which='major')
    plt.xlabel("Prediction")
    plt.ylabel("Real")
    plt.grid(False)
    plt.tight_layout()
    savefig(f'confusion_matrix_{model_name}_{dataname}_mendeley_binary')



# ===============================
# Avaliação do modelo Random Forest
# ===============================

print('Random Forest')

# Carrega as métricas salvas anteriormente a partir de um arquivo JSON
metrics_rf = get_from_json('metrics_results_rf.json')

# Plota as métricas do modelo Random Forest
scores_rf = plot_metrics(metrics_rf, 'rf', loc='lower right')

# ===============================
# Avaliação do modelo LSTM
# ===============================

print('LSTM')

# Carrega as métricas salvas anteriormente a partir de um arquivo JSON
metrics_lstm = get_from_json('metrics_results_lstm.json')

# Plota as métricas do modelo LSTM
scores_lstm = plot_metrics(metrics_lstm, 'lstm', loc='lower right')


