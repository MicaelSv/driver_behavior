import multiprocessing
import numpy as np
import ordpy
import pandas as pd
import time
import os


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
                time_hc.append(tf - t0)
                #
                t0 = time.time()
                s, f = ordpy.fisher_shannon(window_without_duplicate, dx=dx)
                tf = time.time()
                time_fs.append(tf - t0)
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
    time_df.to_csv(fileout+'.time', index=False)
    print("[done]", fileout)

class InformationHandleFile:
    def __init__(self, path,  window, shift=1, dx=6):
        self.__window = window
        self.__shift = shift
        self.__dx = dx
        self.__path = path
        self.__time_hc = []
        self.__time_fs = []
        self.__class = 'Class'

    def get_parameters(self):
        return self.__dx

    def __process_file(self, file):
        filein = file
        fileout = f'{self.__path}/inf_w{self.__window}_dx{self.__dx}_{file.split("/")[-1]}'
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
        ihf = InformationHandleFile(path='.', window=window, dx=dx)
        thread_pool.extend(ihf.create_inf_measures_dataset())

    len_pool = len(thread_pool)
    print("Total treads:", len_pool)
    for i, thread in enumerate(thread_pool, start=1):
        thread.join()
        print(f'{i}/{len_pool}')
