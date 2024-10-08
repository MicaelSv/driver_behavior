import matplotlib.pyplot as plt
from scipy.stats import sem
import numpy as np
import json

def get_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def build_scores_arrays(metrics_dict):
    accuracy_mean_list, accuracy_sem_list = [], []
    precision_mean_list, precision_sem_list = [], []
    recall_mean_list, recall_sem_list = [] , []
    
    for data_name in metrics_dict.keys():
        accuracy_arr = [score['accuracy'] for score in metrics_dict[data_name]]
        precision_arr = [score['precision'] for score in metrics_dict[data_name]]
        recall_arr = [score['recall'] for score in metrics_dict[data_name]]
        
        accuracy_mean_list.append(np.mean(accuracy_arr))
        accuracy_sem_list.append(sem(accuracy_arr))
        precision_mean_list.append(np.mean(precision_arr))
        precision_sem_list.append(sem(precision_arr))
        recall_mean_list.append(np.mean(recall_arr))
        recall_sem_list.append(sem(recall_arr))
    
    return {
        'accuracy': {'mean': accuracy_mean_list, 'sem': accuracy_sem_list},
        'precision': {'mean': precision_mean_list, 'sem': precision_sem_list},
        'recall': {'mean': recall_mean_list, 'sem': recall_sem_list}
    }

def plot_metrics(metrics_dict, model_name):
    scores_dict = build_scores_arrays(metrics_dict)
    font_size = 24
    plt.figure(figsize=(18,10))
    X_names = metrics_dict.keys()
    X_axis = np.arange(len(X_names))
    
    plt.bar(X_axis - 0.25, scores_dict['accuracy']['mean'], 0.2, 
            yerr=scores_dict['accuracy']['sem'], label='Acurácia', edgecolor="black")
    plt.bar(X_axis, scores_dict['precision']['mean'], 0.2, 
            yerr=scores_dict['precision']['sem'], label='Precisão', edgecolor="black")
    plt.bar(X_axis + 0.25, scores_dict['recall']['mean'], 0.2, 
            yerr=scores_dict['recall']['sem'], label='Recall', edgecolor="black")
    
    plt.xticks(X_axis, X_names)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.ylim((0.5, 1))
    plt.xlabel('Método', fontsize=font_size)
    plt.ylabel('Pontuação', fontsize=font_size)
    plt.legend(fontsize=font_size - 4)
    plt.savefig(f'graphic2_{model_name}.png')

# Carregar as métricas e gerar o gráfico
# Para rodar o grafico do LSTM, basta descomentar ele aqui em baixo

metrics_rf = get_from_json('rf_metrics_results.json')
#metrics_lstm = get_from_json('lstm_metrics_results_definitive.json')
plot_metrics(metrics_rf, 'random_forest')
#plot_metrics(metrics_lstm, 'lstm_definitive')
