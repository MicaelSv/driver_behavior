import multiprocessing
import numpy as np
import ordpy
import pandas as pd
import time


class InformationHandleFile:
    def __init__(self, path,  window, shift=1, dx=6):
        self.__window = window
        self.__shift = shift
        self.__dx = dx
        self.__path = path
        self.__time_hc = []
        self.__time_fs = []
        self.__class = 'Class'

    @staticmethod
    def get_sub_lists(original_list, delta):
        pivot = 0
        sub_lists = []
        len_list = len(original_list)
        shift = 1
        while pivot + delta <= len_list:
            sub_lists.append(original_list[pivot:pivot + delta])
            pivot += shift
        return sub_lists

    def get_parameters(self):
        return self.__dx

    @staticmethod
    def __run(df, fileout, window_size, dx, label):
        new_df = None
        new_df_sz = 0
        time_hc = []
        time_fs = []
        df_train_normal = df.loc[df['Class'] == 'NORMAL']
        df_train_slow = df.loc[df['Class'] == 'SLOW']
        df_train_aggressive = df.loc[df['Class'] == 'AGGRESSIVE']
        for _df in (df_train_normal, df_train_slow, df_train_aggressive):
            sliding_window_df = InformationHandleFile.get_sub_lists(_df, window_size)
            for window_df in sliding_window_df:
                row = {}
                for feature in window_df.drop([label, 'Timestamp'], axis=1).columns:
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
                    # row[feature] = window_df[feature].values[-1]
                    row[f'{feature}_entropy'] = h
                    row[f'{feature}_complexity'] = c
                    row[f'{feature}_fisher'] = f
                    row[f'{feature}_shannon'] = s
                    row[label] = window_df[label].values[-1]
                    # # Get value most frequently
                    # lst = window_df[label].values.tolist()
                    # row[label] = max(lst,key=lst.count)
                    # row[label] = np.bincount(lst).argmax()
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

    def __process_file(self, file):
        filein = f'{self.__path}/{file}'
        fileout = f'{self.__path}/inf_w{self.__window}_dx{self.__dx}_{file}'
        df = pd.read_csv(filein)
        p = multiprocessing.Process(target=self.__run,
                                    args=(df, fileout,
                                        self.__window,
                                        self.__dx,
                                        'Class')
                                    )
        p.start()
        return [p]

    def create_inf_measures_dataset(self):
        thread_pool = self.__process_file('test_motion_data.csv')
        thread_pool += self.__process_file('train_motion_data.csv')
        return thread_pool
    
    @staticmethod
    def join_threads(threads):
        len_pool = len(thread_pool)
        print(f'{len_pool} threads.')
        for i, thread in enumerate(thread_pool, start=1):
            thread.join()
            print(f'{i}/{len_pool}')


def choose_embedded_dimension(series_length):
    if series_length <= 6:
        return 3
    if series_length <= 24:
        return 4
    if series_length <= 120:
        return 5
    if series_length <= 720:
        return 6
    if series_length <= 5040:
        return 7
    return 8


if __name__ == '__main__':
    path = './dataset'
    thread_pool = []
    for series_length in range(60, 781, 60):  # 60 to 780
        d = choose_embedded_dimension(series_length)
        thread_pool += InformationHandleFile(
                            path=path,
                            window=series_length,
                            dx=d
                        ).create_inf_measures_dataset()
    InformationHandleFile.join_threads(thread_pool)
