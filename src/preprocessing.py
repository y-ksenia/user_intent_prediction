import os
from itertools import chain
from itertools import combinations, product

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier


def get_df(path):
    dataframe = pd.read_csv(path)

    #     convert to DateTime, correct to China time
    dataframe['timestamp'] = dataframe['timestamp'].apply(
        lambda d: pd.to_datetime(d, unit='ms').tz_localize('UTC').tz_convert('Asia/Hong_Kong'))
    dataframe['timestamp'] = dataframe['timestamp'].dt.round('1s')

    #     calc time of the day in minutes (as ratio)
    dataframe['timeDay'] = (dataframe.timestamp.dt.hour * 60 + dataframe.timestamp.dt.minute) * (2 * np.pi) / (24 * 60)
    dataframe['timeDay_sin'] = dataframe['timeDay'].apply(lambda x: np.sin(x))
    dataframe['timeDay_cos'] = dataframe['timeDay'].apply(lambda x: np.cos(x))
    #     calc time of the week in minutes (as ratio)
    dataframe['timeWeek'] = ((dataframe.timestamp.dt.dayofweek * 24 + dataframe.timestamp.dt.hour) * 60 +
                             dataframe.timestamp.dt.minute) * (2 * np.pi) / (7 * 24 * 60)
    dataframe['timeWeek_sin'] = dataframe['timeWeek'].apply(lambda x: np.sin(x))
    dataframe['timeWeek_cos'] = dataframe['timeWeek'].apply(lambda x: np.cos(x))

    #     remove duplicated observations
    dataframe = dataframe.drop_duplicates()
    dataframe = dataframe.drop(['rowid', 'activityName', 'city', 'timeDay', 'timeWeek'], 1)

    dataframe = dataframe.dropna()
    dataframe = dataframe.sort_values('timestamp')
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.set_index('timestamp')

    # get apps of interest
    application_list = set(dataframe.packageName.unique())
    idx_to_app_dict = {i: app for i, app in enumerate(application_list)}
    app_to_idx_dict = {app: i for i, app in enumerate(application_list)}

    return dataframe, application_list, idx_to_app_dict, app_to_idx_dict


def fill_missing_locations(dataframe, mode='train', gps_wifi=None, gps=None,
                           cols=None):
    if cols is None:
        cols = ['longitude', 'latitude', 'MCC', 'MNC', 'LAC', 'cellID', 'BSSID']
    df = dataframe.copy()

    # first step: convert -1 to np.nan
    if all([x in df.columns for x in cols[:2]]):
        df['latitude'] = df['latitude'].apply(lambda x: x if x != -1. else np.nan)
        df['longitude'] = df['longitude'].apply(lambda x: x if x != -1. else np.nan)
    else:
        raise RuntimeError('No location information in dataframe!')

    # based on GPS and Wifi info fill missing locations
    if all([x in df.columns for x in cols[2:]]):
        if mode == 'train':
            gps_wifi = df.groupby(['MCC', 'MNC', 'LAC', 'cellID', 'BSSID']).agg(
                {'longitude': 'mean', 'latitude': 'mean'})
            gps_wifi = gps_wifi[gps_wifi.index.get_level_values(0) != 0]
            df[['longitude', 'latitude']] = df.groupby(['MCC', 'MNC', 'LAC', 'cellID', 'BSSID']) \
                [['longitude', 'latitude']].transform(lambda x: x.fillna(x.mean()))
        elif mode == 'test':
            for i in gps_wifi.itertuples():
                df[df[['MCC', 'MNC', 'LAC', 'cellID', 'BSSID']] == i[0]]['longitude'].fillna(i[1], inplace=True)
                df[df[['MCC', 'MNC', 'LAC', 'cellID', 'BSSID']] == i[0]]['latitude'].fillna(i[2], inplace=True)

    # for the rest missing locations - fill with mean based on GPS info only (when Wifi is not available)
    if all([x in df.columns for x in cols[2:-1]]):
        if mode == 'train':
            gps = df.groupby(['MCC', 'MNC', 'LAC', 'cellID']).agg({'longitude': 'mean', 'latitude': 'mean'})
            gps = gps[gps.index.get_level_values(0) != 0]
            df[['longitude', 'latitude']] = df.groupby(['MCC', 'MNC', 'LAC', 'cellID']) \
                [['longitude', 'latitude']].transform(lambda x: x.fillna(x.mean()))
        elif mode == 'test':
            for i in gps.itertuples():
                df[df[['MCC', 'MNC', 'LAC', 'cellID']] == i[0]]['longitude'].fillna(i[1], inplace=True)
                df[df[['MCC', 'MNC', 'LAC', 'cellID']] == i[0]]['latitude'].fillna(i[2], inplace=True)

    # for the rest of missing columns fill them back to -1
    df = df.fillna(-1)
    df = df.drop(['cellID', 'MCC', 'MNC', 'LAC', 'BSSID'], axis=1)
    if mode == 'train':
        return df, gps_wifi, gps
    else:
        return df


def clusterization(dataframe, cols, eps, min_samples, dim='time', mode='train', classifier=None):
    df = dataframe.copy()
    metric = None
    coords = None

    if dim == 'time':
        metric = 'euclidean'
        coords = df[cols].values
    elif dim == 'loc':
        metric = 'haversine'
        coords = np.radians(df[cols].values)
        kms_per_radian = 6371.0088
        eps = eps / kms_per_radian

    if mode == 'train':
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm='brute').fit(coords)
        df[f'{dim}_cluster'] = clusterer.labels_

        classifier = KNeighborsClassifier(n_neighbors=min_samples)
        classifier = classifier.set_params(**{k: v for k, v in clusterer.get_params().items()
                                              if k in classifier.get_params()})
        classifier = classifier.set_params(**{'n_neighbors': 1})
        classifier = classifier.fit(coords, df[f'{dim}_cluster'])
        return df, clusterer, classifier
    elif mode == 'test':
        df[f'{dim}_cluster'] = classifier.predict(coords)
        return df


def pairs(*lists):
    for t in combinations(lists, 2):
        for pair in product(*t):
            yield pair


def final_clusterization(dataframe, time_cols, loc_cols, other_cols=None, mode='train', clucter_dict=None):
    if other_cols is None:
        other_cols = []
    df = dataframe.copy()

    time_clusters = df['time_cluster'].unique()
    loc_clusters = df['loc_cluster'].unique()
    if mode == 'train':
        clucter_dict = {}
        for i, pair in enumerate(pairs(time_clusters, loc_clusters)):
            clucter_dict[pair] = i
    df['cluster'] = list(zip(df.time_cluster, df.loc_cluster))
    df['cluster'] = df['cluster'].apply(lambda x: clucter_dict[x])
    to_drop = list(chain(['time_cluster', 'loc_cluster'], time_cols, loc_cols, other_cols))
    df = df.drop(to_drop, axis=1)
    if mode == 'train':
        return df, clucter_dict
    elif mode == 'test':
        return df


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def generate_previous_events(dataframe, minutes, app_to_idx_dict, mode='train', from_=None):
    df = dataframe.copy()
    df = df.reset_index().drop_duplicates().set_index('timestamp')
    if from_ is None:
        from_ = dataframe.index[0]

    # first supportive dataframe
    df1 = df['packageName'].reset_index().copy()
    df1['ones'] = np.ones(len(df1))

    # second supportive dataframe
    df2 = df.reset_index().reset_index()[['index', 'timestamp']].copy()
    df2['start_time'] = df2.timestamp - pd.Timedelta(minutes, unit='minutes')
    df2['end_time'] = df2['timestamp']
    df2 = df2.drop('timestamp', 1)
    df2['ones'] = np.ones(len(df2))

    # cross-join of two supportive dataframes
    merged = df1.copy().merge(df2.copy(), on='ones', how='inner', copy=True)
    # filter only interesting data periods
    filtered = merged[
        (merged['timestamp'] < merged['end_time'])
        &
        (merged['timestamp'] >= merged['start_time'])
        ].copy()

    prev_apps = filtered.groupby('index').agg({'packageName': list, 'end_time': 'last'})
    prev_apps['timestamp'] = prev_apps['end_time']
    prev_apps['prev_apps'] = prev_apps['packageName']
    prev_apps = prev_apps.drop(['end_time', 'packageName'], axis=1)

    df = df.copy().reset_index().merge(prev_apps.copy(), on='timestamp', how='outer') \
        [['timestamp', 'cluster', 'packageName', 'prev_apps']]
    df = df.set_index('timestamp')
    df['cur_app_idx'] = df['packageName'].apply(lambda x: app_to_idx_dict[x])
    df['prev_apps_idx'] = df['prev_apps'].apply \
        (lambda x: [app_to_idx_dict[i] for i in f7(x)] if isinstance(x, list) else np.nan)
    # df = df.ffill()
    df = df.drop(['packageName', 'prev_apps'], axis=1)
    if mode == 'train':
        df = df.dropna()
    elif mode == 'test':
        nan_idx = df['prev_apps_idx'].isna().index
        df['prev_apps_idx'] = df['prev_apps_idx'].ffill()
        df['prev_apps_idx'].loc[nan_idx] = df['prev_apps_idx'].loc[nan_idx].apply(lambda x: [x[-1]] \
            if isinstance(x, list) else np.nan)
    df = df[df.index >= from_]
    return df


def generate_next_events(dataframe, minutes, app_to_idx_dict, mode='train', from_=None):
    df = dataframe.copy()
    df = df.reset_index().drop_duplicates().set_index('timestamp')
    if from_ is None:
        from_ = dataframe.index[0]

    # first supportive dataframe
    df1 = df['packageName'].reset_index().copy()
    df1['ones'] = np.ones(len(df1))

    # second supportive dataframe
    df2 = df.reset_index().reset_index()[['index', 'timestamp']].copy()
    df2['end_time'] = df2.timestamp + pd.Timedelta(minutes, unit='minutes')
    df2['start_time'] = df2['timestamp']
    df2 = df2.drop('timestamp', 1)
    df2['ones'] = np.ones(len(df2))

    # cross-join of two supportive dataframes
    merged = df1.copy().merge(df2.copy(), on='ones', how='inner', copy=True)
    # filter only interesting data periods
    filtered = merged[
        (merged['timestamp'] < merged['end_time'])
        &
        (merged['timestamp'] >= merged['start_time'])
        ].copy()

    next_apps = filtered.groupby('index').agg({'packageName': list, 'start_time': 'last'})
    next_apps['timestamp'] = next_apps['start_time']
    next_apps['next_apps'] = next_apps['packageName']
    next_apps = next_apps.drop(['start_time', 'packageName'], axis=1)

    df = df.copy().reset_index().merge(next_apps.copy(), on='timestamp', how='outer') \
        [['timestamp', 'cluster', 'packageName', 'next_apps']]
    df = df.set_index('timestamp')
    df['cur_app_idx'] = df['packageName'].apply(lambda x: app_to_idx_dict[x])
    df['next_apps_idx'] = df['next_apps'].apply \
        (lambda x: [app_to_idx_dict[i] for i in f7(x)] if isinstance(x, list) else np.nan)

    df = df.drop(['packageName', 'next_apps'], axis=1)

    # nan_idx = df['next_apps_idx'].isna().index
    # df['next_apps_idx'] = df['next_apps_idx'].ffill()
    # df['next_apps_idx'].loc[nan_idx] = df['next_apps_idx'].loc[nan_idx].apply(lambda x: [x[-1]] \
    #                                                                         if isinstance(x, list) else np.nan)
    # df = df[df.index >= from_]
    return df


def process_dataframe(path):
    df, app_list, idx_to_app, app_to_idx = get_df(path)
    loc_cols = ['longitude', 'latitude']
    time_cols = [x for x in df.columns if (('timeDay' in x) or ('timeWeek' in x))]

    df = fill_missing_locations(df)
    df = clusterization(df, time_cols, eps=0.1, min_samples=10, dim='time')
    df = clusterization(df, loc_cols, eps=0.5, min_samples=10, dim='loc')
    df = final_clusterization(df, time_cols, loc_cols)
    df = df.sort_index()
    df = generate_previous_events(df, 5, app_to_idx)
    return df, app_list, idx_to_app, app_to_idx


def main():
    path = '../data/BetaUser Data V1.1'
    case_list = sorted([x for x in os.listdir(path) if x.endswith('.csv')])
    case = case_list[0]

    df, app_list, idx_to_app, app_to_idx = process_dataframe(os.path.join(path, case))
    print(df)


if __name__ == '__main__':
    main()
