from datetime import timedelta
from geopy.distance import great_circle
from sklearn.cluster import DBSCAN
from tqdm import tqdm

"""
INPUTS:
    df={o1,o2,...,on} Set of objects
    spatial_threshold = Maximum geographical coordinate (spatial) distance value
    temporal_threshold = Maximum non-spatial distance value
    min_neighbors = Minimun number of points within Eps1 and Eps2 distance
OUTPUT:
    C = {c1,c2,...,ck} Set of clusters
"""


def ST_DBSCAN(df, spatial_threshold, temporal_threshold, min_neighbors,  time_cols=None):
    if time_cols is None:
        time_cols = ['timestamp']
    cluster_label = 0
    NOISE = -1
    UNMARKED = 777777
    stack = []

    # initialize each point with unmarked
    df['cluster'] = UNMARKED

    if len(time_cols) > 1:
        clusterer = DBSCAN(eps=temporal_threshold, min_samples=min_neighbors, metric='euclidean').fit(df[time_cols].values)
        df['time_cluster'] = clusterer.labels_

    # for each point in database
    for index, point in tqdm(df.iterrows()):
        if df.loc[index]['cluster'] == UNMARKED:

            neighborhood = retrieve_neighbors(index, df, spatial_threshold, temporal_threshold, time_cols=time_cols)

            if len(neighborhood) < min_neighbors:
                df.loc[index, 'cluster'] = NOISE

            else:  # found a core point
                cluster_label = cluster_label + 1
                df.loc[index, 'cluster'] = cluster_label  # assign a label to core point

                for neig_index in neighborhood:  # assign core's label to its neighborhood
                    df.loc[neig_index, 'cluster'] = cluster_label
                    stack.append(neig_index)  # append neighborhood to stack

                while len(stack) > 0:  # find new neighbors from core point neighborhood
                    current_point_index = stack.pop()
                    new_neighborhood = retrieve_neighbors(current_point_index, df, spatial_threshold,
                                                          temporal_threshold, time_cols=time_cols)

                    if len(new_neighborhood) >= min_neighbors:  # current_point is a new core
                        for neig_index in new_neighborhood:
                            neig_cluster = df.loc[neig_index]['cluster']
                            if (neig_cluster != NOISE) & (neig_cluster == UNMARKED):
                                # TODO: verify cluster average before add new point
                                df.loc[neig_index, 'cluster'] = cluster_label
                                stack.append(neig_index)
    return df


def retrieve_neighbors(index_center, df, spatial_threshold, temporal_threshold, time_cols):
    neigborhood = []

    center_point = df.loc[index_center].copy()
    # print(center_point)

    # filter by time
    if len(time_cols) == 1:
        min_time = center_point[time_cols] - timedelta(minutes=temporal_threshold)
        max_time = center_point[time_cols] + timedelta(minutes=temporal_threshold)
        df = df[(df[time_cols] >= min_time) & (df[time_cols] <= max_time)]
    else:
        df = df[df['time_cluster'] == center_point['time_cluster']]

    # filter by distance
    for index, point in df.iterrows():
        if index != index_center:
            distance = great_circle((center_point['latitude'], center_point['longitude']),
                                    (point['latitude'], point['longitude'])).meters
            if distance <= spatial_threshold:
                neigborhood.append(index)

    return neigborhood
