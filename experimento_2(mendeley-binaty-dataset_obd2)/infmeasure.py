import multiprocessing
import numpy as np
import ordpy
import pandas as pd
import time
import os
from pathlib import Path


def get_sub_lists(original_list, delta):
    pivot = 0
    sub_lists = []
    len_list = len(original_list)
    shift = 1
    while pivot + delta <= len_list:
        sub_lists.append(original_list[pivot:pivot + delta])
        pivot += shift
    return sub_lists


def run(df, fileout, window_size, dx, label):
    sliding_window_df = get_sub_lists(df, window_size)
    new_df = None
    new_df_sz = 0
    time_hc = []
    time_fs = []
    feature_times = {}
    for window_df in sliding_window_df:
        row = {}
        for feature in window_df.drop([label], axis=1).columns:
            window_without_duplicate = window_df[feature].loc[window_df[feature].shift() != window_df[feature]]
            if len(window_without_duplicate) < dx:
                h = c = f = s = 'NaN'
            else:
                t0 = time.time()
                h, c = ordpy.complexity_entropy(window_without_duplicate, dx=dx)
                tf = time.time()
                elapsed_hc = tf - t0
                time_hc.append(elapsed_hc)
                #
                t0 = time.time()
                s, f = ordpy.fisher_shannon(window_without_duplicate, dx=dx)
                tf = time.time()
                elapsed_fs = tf - t0
                time_fs.append(elapsed_fs)

                if feature not in feature_times:
                    feature_times[feature] = {'time_hc': [], 'time_fs': []}
                feature_times[feature]['time_hc'].append(elapsed_hc)
                feature_times[feature]['time_fs'].append(elapsed_fs)
            row[f'{feature}_entropy'] = h
            row[f'{feature}_complexity'] = c
            row[f'{feature}_fisher'] = f
            row[f'{feature}_shannon'] = s
            lst = window_df[label].values.tolist()
            row[label] = max(lst,key=lst.count)
        if new_df is None:
            new_df = pd.DataFrame([row])
        else:
            new_df.loc[new_df_sz] = row
        new_df_sz += 1
    new_df.to_csv(fileout, index=False)
    time_dict = {
        'time_hc': time_hc,
        'time_fs': time_fs
    }
    time_df = pd.DataFrame.from_dict(time_dict)
    time_df.to_csv(f'{fileout}.time', index=False)

    if feature_times:
        feature_rows = []
        for feature, values in feature_times.items():
            sum_time_hc = float(np.sum(values['time_hc']))
            sum_time_fs = float(np.sum(values['time_fs']))
            mean_time_hc = float(np.mean(values['time_hc']))
            mean_time_fs = float(np.mean(values['time_fs']))
            feature_rows.append({
                'feature': feature,
                'calc_count': len(values['time_hc']),
                'sum_time_hc': sum_time_hc,
                'sum_time_fs': sum_time_fs,
                'mean_time_hc': mean_time_hc,
                'mean_time_fs': mean_time_fs,
                'sum_time_total': sum_time_hc + sum_time_fs,
                'mean_time_total': mean_time_hc + mean_time_fs,
            })
        feature_time_df = pd.DataFrame(feature_rows)
        feature_time_df.to_csv(f'{fileout}.time_by_feature.csv', index=False)

    print("[done]", fileout)

class InformationHandleFile:
    def __init__(self, path,  window, shift=1, dx=6):
        self.__window = window
        self.__shift = shift
        self.__dx = dx
        base_dir = Path(__file__).resolve().parent
        self.__path = (base_dir / path).resolve()
        self.__time_hc = []
        self.__time_fs = []
        self.__class = 'Class'

    def get_parameters(self):
        return self.__dx

    def __process_file(self, file):
        filein = self.__path / file
        fileout = self.__path / f'inf_w{self.__window}_dx{self.__dx}_{Path(file).name}'
        df = pd.read_csv(filein)
        p = multiprocessing.Process(target=run,
                                    args=(df, fileout,
                                        self.__window,
                                        self.__dx,
                                        'Class')
                                    )
        p.start()
        return [p]


    def create_inf_measures_dataset(self):
        thread_pool = self.__process_file('dataset_obd2.csv')
        # len_pool = len(thread_pool)
        # print(f'{len_pool} threads. Window={self.__window}. dx={self.__dx}')
        # for i, thread in enumerate(thread_pool, start=1):
        #     thread.join()
        #     print(f'{i}/{len_pool}')
        return thread_pool


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    ihf_configs = [
        (60, 4), (120, 4), (180, 4), (240, 4), 
#        (300, 6),
#        (360, 6), (420, 6), (480, 6), (540, 6), (600, 6),
#        (660, 6), (720, 6), (780, 7)
    ]

    thread_pool = []

    for window, dx in ihf_configs:
        ihf = InformationHandleFile(path='dataset', window=window, dx=dx)
        thread_pool.extend(ihf.create_inf_measures_dataset())

    len_pool = len(thread_pool)
    print("Total treads:", len_pool)
    for i, thread in enumerate(thread_pool, start=1):
        thread.join()
        print(f'{i}/{len_pool}')
