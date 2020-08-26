import os
import sys
import gc
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm, trange
from sklearn.cluster import KMeans, DBSCAN

sys.path.append('../src')
import FPMC
import STDBSCAN
import preprocessing, params_tuning, visualization, calc_metrics

# numba
import warnings
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

target_column = 'packageName'

def parse_args():
    """
    Настройка argparse
    :return: parser arguments as input for def main
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='the path to csv.', type=str)
    parser.add_argument('--train', help='history period for train (minutes)', type=int, default=5)
    parser.add_argument('--pred', help='time period for prediction (minutes)', type=int, default=5)
    parser.add_argument('-e', '--n_epoch', help='# of epoch', type=int, default=20)
    parser.add_argument('--n_neg', help='# of neg samples', type=int, default=5)
    parser.add_argument('-n', '--n_factor', help='dimension of factorization', type=int, default=64)
    parser.add_argument('-l', '--learn_rate', help='learning rate', type=float, default=0.01)
    parser.add_argument('-r', '--regular', help='regularization', type=float, default=0.01)

    return parser.parse_args()


def main():
    args = parse_args()

    path = args.path
    train_period = args.train
    test_period = args.pred

    df, app_list, idx_to_app, app_to_idx = preprocessing.get_df(path)
    loc_cols = ['longitude', 'latitude']
    time_cols = [x for x in df.columns if (('timeDay' in x) or ('timeWeek' in x))]

    df_train = df[df.index.date < (df.iloc[-1].name.date() - pd.Timedelta(7, unit='days'))]
    df_test = df[df.index.date >= (df.iloc[-1].name.date() - pd.Timedelta(7, unit='days'))]

    df_train, gps_wifi, gps = preprocessing.fill_missing_locations(df_train, mode='train')


if __name__ == '__main__':
    main()
