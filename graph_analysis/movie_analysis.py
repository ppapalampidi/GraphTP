import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
import os
from matplotlib import pyplot as plt
import numpy as np

import networkx as nx
import pandas as pd


def construct_movie_graph(markers, graph_connections, i):
    colors = ['mediumaquamarine', 'seagreen', 'goldenrod', 'red', 'lightcoral']
    new_graph_con = []
    Edges = []
    num_neighbors = []
    for j in range(len(graph_connections[i])):
        scene_neighbors = graph_connections[i][j]
        num_neighbors.append(0)
        for n, k in enumerate(scene_neighbors):
            if k != 0:
                num_neighbors[-1] += 1
                Edges.append((j, n))
                scene_neighbors[n] = 1
        new_graph_con.append(scene_neighbors)

    for x in range(len(new_graph_con)):
        if new_graph_con[x][x] == 1:
            new_graph_con[x][x] = 0

    adj_matrix = np.asarray(new_graph_con)
    labels = range(len(new_graph_con))

    DF_adj = pd.DataFrame(adj_matrix, index=labels, columns=labels)
    G_helper = nx.Graph()
    G_helper.add_nodes_from(labels)

    group = []
    for node in G_helper:
        if node in markers[0]:
            group.append(colors[0])
        elif node in markers[1]:
            group.append(colors[1])
        elif node in markers[2]:
            group.append(colors[2])
        elif node in markers[3]:
            group.append(colors[3])
        elif node in markers[4]:
            group.append(colors[4])
        else:
            group.append('lightblue')

    labels = range(len(new_graph_con))
    G_helper = nx.Graph()
    G_helper.add_nodes_from(labels)

    markers_temp = [x for sublist in markers for x in sublist for sublist in
                    markers]

    keep_useful = []

    for w in range(DF_adj.shape[0]):
        col_label = DF_adj.columns[w]
        for m in range(DF_adj.shape[1]):
            row_label = DF_adj.index[m]
            node = DF_adj.iloc[w, m]
            if node == 1 and (
                row_label in markers_temp):
                G_helper.add_edge(col_label, row_label)
                if col_label not in keep_useful:
                    keep_useful.append(col_label)
                if row_label not in keep_useful:
                    keep_useful.append(row_label)

    for_removal = []

    for node in G_helper:
        if node not in keep_useful:
            for_removal.append(node)

    for node in for_removal:
        G_helper.remove_node(node)

    return (G_helper)


if __name__ == '__main__':

    output_folder = '../outputs/'

    parser = argparse.ArgumentParser(description='TP identification evaluation.')

    parser.add_argument('--folder',
                        help='model folder for evaluation')

    args = parser.parse_args()

    model_folder = args.folder
    model_tested = args.folder

    movie_names = []
    graph_connections = []
    labels = []
    tps = []

    category_1 = ['Juno (film)', 'The Back-up Plan', 'The Breakfast Club',
                  '500 Days of Summer',
                  'Crazy, Stupid, Love', 'Easy A', 'Marley & Me (film)',
                  'No Strings Attached (film)']

    category_2 = ['Arbitrage (film)', 'Panic Room', 'The Shining (film)',
                  'One Eight Seven', 'Black Swan (film)',
                  'Gothika', 'Heat (1995 film)', 'House of 1000 Corpses',
                  'Sleepy Hollow (film)', 'The Talented Mr',
                  'The Thing (1982 film)']

    category_3 = ['Die Hard', 'Soldier (1998 American film)',
                  'The Crying Game',
                  'Total Recall (1990 film)', '2012 (film)',
                  'From Russia with Love (film)',
                  'American Gangster (film)', 'Collateral Damage (film)',
                  'Oblivion (2013 film)']

    category_4 = ['Moon (film)', 'Slumdog Millionaire',
                  'The Last Temptation of Christ (film)', 'Unforgiven',
                  'American Beauty (film)', 'Jane Eyre (2011 film)',
                  'The Majestic (film)', 'A Walk to Remember']

    count_per_TP = {
        'Comedy/Romance': {'tp1': [], 'tp2': [], 'tp3': [], 'tp4': [],
                           'tp5': []},
        'Thriller/Mystery': {'tp1': [], 'tp2': [], 'tp3': [], 'tp4': [],
                             'tp5': []},
        'Action': {'tp1': [], 'tp2': [], 'tp3': [], 'tp4': [], 'tp5': []},
        'Drama/Other': {'tp1': [], 'tp2': [], 'tp3': [], 'tp4': [], 'tp5': []}}

    for i in range(5):
        folder_now = os.path.join(output_folder, model_folder + '_' + str(i + 1))

        with open(folder_now + '/movie_names_' + model_tested + '.pkl', 'rb') \
                as handle:
            movie_names += pickle.load(handle)
        with open(folder_now + '/graph_neighbors_' + model_tested + '.pkl', 'rb') \
                as handle:
            graph_connections += pickle.load(handle)
        with open(folder_now + '/labels_' + model_tested + '.pkl', 'rb') \
                as handle:
            labels += pickle.load(handle)
        with open(folder_now + '/turning_points_' + model_tested + '.pkl', 'rb') \
                as handle:
            tps += pickle.load(handle)

    """
    Construct graphs per movie
    """

    G_all = []
    markers_all = []

    for k, movie_name in enumerate(movie_names):
        markers = []
        for x, tp in enumerate(tps[k]):
            markers.append(tp)
        G_now = construct_movie_graph(markers, graph_connections, k)
        G_all.append(G_now)
        markers_all.append(markers)

    """
    Compute and plot connectivity per TP based on genre
    """
    for q, G_now in enumerate(G_all):
        if movie_names[q] in category_1:
            cat = 'Comedy/Romance'
        elif movie_names[q] in category_2:
            cat = 'Thriller/Mystery'
        elif movie_names[q] in category_3:
            cat = 'Action'
        else:
            cat = 'Drama/Other'

        nbunch = []
        for i in markers_all[q]:
            for j in i:
                nbunch.append(j)

        connectivity = nx.all_pairs_node_connectivity(G_now)
        tp1 = []
        tp2 = []
        tp3 = []
        tp4 = []
        tp5 = []
        for key, value in connectivity.items():
            keep_now = []
            for key1, value1 in value.items():
                keep_now.append(value1)
            degree = np.asarray(keep_now).mean()
            node = key

            if node in markers_all[q][0]:
                tp1.append(degree)
            elif node in markers_all[q][1]:
                tp2.append(degree)
            elif node in markers_all[q][2]:
                tp3.append(degree)
            elif node in markers_all[q][3]:
                tp4.append(degree)
            elif node in markers_all[q][4]:
                tp5.append(degree)

        tp1 = np.asarray(tp1).mean()
        tp2 = np.asarray(tp2).mean()
        tp3 = np.asarray(tp3).mean()
        tp4 = np.asarray(tp4).mean()
        tp5 = np.asarray(tp5).mean()

        if not np.isnan(tp1):
            count_per_TP[cat]['tp1'].append(tp1)
        if not np.isnan(tp2):
            count_per_TP[cat]['tp2'].append(tp2)
        if not np.isnan(tp3):
            count_per_TP[cat]['tp3'].append(tp3)
        if not np.isnan(tp4):
            count_per_TP[cat]['tp4'].append(tp4)
        if not np.isnan(tp5):
            count_per_TP[cat]['tp5'].append(tp5)

    all_avgs = []
    cats = []
    linestyles = ['-', '--', '-.', ':']
    colors = ['teal', 'black', 'olive', 'crimson']
    cnt = 0
    for key, value in count_per_TP.items():
        avgs = []
        stds = []
        for key1, value1 in count_per_TP[key].items():
            avg_now = np.asarray(value1).mean()
            std_now = np.asarray(value1).std()
            avgs.append(avg_now)
            stds.append(std_now)
        all_avgs.append(avgs)
        cats.append(key)
        plt.plot(avgs, 'D-', label=key, linestyle=linestyles[cnt],
                 color=colors[cnt], linewidth=5)
        cnt += 1

    plt.xticks([0, 1, 2, 3, 4], ['TP1', 'TP2', 'TP3', 'TP4', 'TP5'])
    plt.tight_layout(pad=0)
    plt.legend(loc='best')
    plt.rcParams['figure.figsize'] = [20, 11]

    destination_folder = os.path.join(output_folder, 'movie_graphs')

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    plt.savefig(os.path.join(destination_folder,
                             "connectivity_per_TP_genre.pdf"))