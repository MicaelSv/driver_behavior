import multiprocessing
import numpy as np
import ordpy
import os.path
import pandas as pd
import time
import warnings

# Suprimir o aviso específico
warnings.filterwarnings("ignore", message="Be mindful the correct calculation of Fisher information depends on all possible permutations")


class ExtractInformation:
    def __init__(self, df, path_out, window_length, feature_label_name, embedding_dimension=6, number_of_threads=1):
        self.__df = df
        self.__path_out = path_out
        self.__window_length = window_length
        self.__label = feature_label_name
        self.__d = embedding_dimension
        self.__threads_number = number_of_threads

    def get_parameters(self):
        return self.__d
    
    def run(self):
        threads_list = []
        sublist = self.__getsublist(self.__df, self.__window_length)

        # Separa o Dataframe "sublist" em "self.__threads_number" sublistas de mesmo tamanho para que cada thread trabalhe em uma parte da lista
        sublist_len = len(sublist)
        x = sublist_len // self.__threads_number
        sublist_split = [sublist[i:i+x] for i in range(0, sublist_len, x)]
        
        for i, thread_df in enumerate(sublist_split):
            filename = f'{self.__path_out}.part{i}.csv'
            p = multiprocessing.Process(target=self.__run,
                                        args=(thread_df, filename, self.__d, self.__label))
            print('  Opening thread for', filename, '...')
            p.start()
            threads_list.append(p)
        return threads_list

    @staticmethod
    def __run(df_list, fileout, dx, label):
        new_df = None
        new_df_sz = 0
        time_probs = []
        time_hc = []
        time_fs = []
        for window_df in df_list:
            row = {}
            for feature in window_df.drop(columns=[label]).columns:
                window = window_df[feature]
                if len(window) < dx:
                    h = c = f = s = 'NaN'
                else:
                    t0 = time.time()
                    data_probs = ordpy.ordinal_distribution(window, dx=dx, return_missing=True)[1]
                    tf = time.time()
                    time_probs.append(tf - t0)
                    #
                    t0 = time.time()
                    h, c = ordpy.complexity_entropy(data=data_probs, dx=dx, probs=True)
                    tf = time.time()
                    time_hc.append(tf - t0)
                    #
                    t0 = time.time()
                    f, s = ordpy.fisher_shannon(data=data_probs, dx=dx, probs=True)
                    tf = time.time()
                    time_fs.append(tf - t0)
                row[feature] = window_df[feature].values[-1]
                row[f'{feature}_entropy'] = h
                row[f'{feature}_complexity'] = c
                row[f'{feature}_fisher'] = f
                lst = window_df[label].values.tolist()
                row[label] = max(lst,key=lst.count)  # Get value most frequently
            if new_df is None:
                new_df = pd.DataFrame([row])
            else:
                new_df.loc[new_df_sz] = row
            new_df_sz += 1
        new_df.to_csv(fileout, index=False)
        time_dict = {
            'time_probs': time_probs,
            'time_hc': time_hc,
            'time_fs': time_fs
        }
        time_df = pd.DataFrame.from_dict(time_dict)
        time_df.to_csv(fileout + '.time', index=False)

    @staticmethod
    def __getsublist(original_list, delta):
        pivot = 0
        sublist = []
        list_len = len(original_list)
        shift = 1
        while pivot + delta <= list_len:
            sublist.append(original_list[pivot:pivot + delta])
            pivot += shift
        return sublist


def choose_embedded_dimension(series_length):
    if series_length <= 120:
        return 5
    if series_length <= 720:
        return 6
    if series_length <= 5040:
        return 7
    return 8


def dataset__dataset_obd2():
    print('[dataset] dataset_obd2')
    path_in = './datasets/dataset_obd2.csv'
    df = pd.read_csv(path_in)
    thread_list = []
    for series_length in range(60, 901, 60):
        d = choose_embedded_dimension(series_length)
        directory_out = f'./datasets/dataset_obd2/window{series_length}_dx{d}'
        path_out = f'{directory_out}/dataset_obd2_inf.csv'

        print(path_out)
        if not os.path.exists(directory_out):
            print('\t[create path]', directory_out)
            os.makedirs(directory_out, exist_ok=True)

        thread_list += ExtractInformation(
            df=df, 
            path_out=path_out, 
            window_length=series_length, 
            feature_label_name='Class', 
            embedding_dimension=d, 
            number_of_threads=4
        ).run()
    print('Total parcial', len(thread_list), 'threads')
    for i, thread in enumerate(thread_list, start=1):
        print(f'{i}/{len(thread_list)}')
        thread.join()
    


def dataset__obd_data():
    print('[dataset] obd_data')
    path_in = './datasets/obd_data.csv'
    df = pd.read_csv(path_in)
    thread_list = []
    for series_length in range(60, 901, 60):
        d = choose_embedded_dimension(series_length)
        directory_out = f'./datasets/obd_data/window{series_length}_dx{d}'
        path_out = f'{directory_out}/dataset_obd2_inf.csv'

        print(path_out)
        if not os.path.exists(directory_out):
            print('\t[create path]', directory_out)
            os.makedirs(directory_out, exist_ok=True)

        thread_list += ExtractInformation(
            df=df, 
            path_out=path_out, 
            window_length=series_length, 
            feature_label_name='Class', 
            embedding_dimension=d, 
            number_of_threads=4
        ).run()
    print('Total parcial', len(thread_list), 'threads')
    for i, thread in enumerate(thread_list, start=1):
        print(f'{i}/{len(thread_list)}')
        thread.join()


if __name__ == '__main__':
    dataset__dataset_obd2()
    dataset__obd_data()