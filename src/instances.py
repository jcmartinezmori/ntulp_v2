import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pickle
from src.config import *
import src.helper as helper


def community_area():

    g, stops_df, lines_df, trips_df = helper.preprocess_load()

    samples_df = []

    N = [i for i in range(trips_df['Pickup Community Area'].nunique())]
    J = [j for j in range(lines_df.shape[0])]
    K = [k for k in range(1)]
    A = [lines_df['length'].to_list()]
    B = [[0] for _ in N]
    V = [np.array([0 for _ in J], dtype=float) for _ in N]

    for idx, trip in trips_df.iterrows():
        print(idx)
        if int(trip.o_node) in AIRPORT_NODES or int(trip.d_node) in AIRPORT_NODES:
            continue
        if int(trip.d_node) in nx.single_source_dijkstra_path_length(
                g, trip.o_node, cutoff=2*LINES_DIST_TRGT, weight='length'
        ):
            continue
        V_t = []
        for _, line in lines_df.iterrows():
            if trip.o_node in line.dist.keys() and trip.d_node in line.dist.keys():
                o_dist = line.dist[trip.o_node]
                if o_dist <= LINES_DIST_TRGT:
                    o_term = 1
                elif o_dist >= LINES_DIST_CTFF:
                    o_term = 0
                else:
                    o_term = (LINES_DIST_CTFF - o_dist) / (LINES_DIST_CTFF - LINES_DIST_TRGT)
                d_dist = line.dist[trip.d_node]
                if d_dist <= LINES_DIST_TRGT:
                    d_term = 1
                elif d_dist >= LINES_DIST_CTFF:
                    d_term = 0
                else:
                    d_term = (LINES_DIST_CTFF - d_dist) / (LINES_DIST_CTFF - LINES_DIST_TRGT)
                V_tj = min(o_term, d_term)
            else:
                V_tj = 0
            V_t.append(V_tj)
        V_t = np.array(V_t, dtype=float)
        B[int(trip['Pickup Community Area'] - 1)][0] += 1
        V[int(trip['Pickup Community Area'] - 1)] += V_t
        B[int(trip['Dropoff Community Area'] - 1)][0] += 1
        V[int(trip['Dropoff Community Area'] - 1)] += V_t

        samples_df.append((int(trip['Pickup Community Area'] - 1), trip['o_x'], trip['o_y']))
        samples_df.append((int(trip['Dropoff Community Area'] - 1), trip['d_x'], trip['d_y']))

    samples_df = pd.DataFrame(samples_df, columns=['community_area', 'x', 'y'])
    samples_df = samples_df.groupby('community_area')[['x', 'y']].mean().reset_index()
    samples_df['node'] = samples_df.apply(lambda sample: ox.nearest_nodes(g, sample.x, sample.y), axis=1)
    samples_df['ct'] = samples_df['community_area'].map({i: B[i][0] for i in N})

    for i, V_i in enumerate(V):
        V[i] = V_i / np.linalg.norm(V_i)

    samples_df.to_csv('{0}/results/instances/samples_df_{1}_{2}.csv'.format(RELPATH, FILENAME, 'community_area'), index=False)

    instance = N, J, K, A, B, V
    with open('{0}/results/instances/instance_{1}_{2}.pkl'.format(RELPATH, FILENAME, 'community_area'), 'wb') as file:
        pickle.dump(instance, file)


if __name__ == '__main__':
    community_area()
