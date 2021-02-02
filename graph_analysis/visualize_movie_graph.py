import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
import os
from matplotlib import pyplot as plt
import networkx as nx

from movie_analysis import construct_movie_graph


if __name__ == '__main__':

    output_folder = '../outputs/'

    parser = argparse.ArgumentParser(description='TP identification evaluation.')

    parser.add_argument('--folder',
                        help='model folder of the TP identification results')

    parser.add_argument('--movie', choices={'Juno (film)', 'The Back-up Plan',
                   'The Breakfast Club', '500 Days of Summer',
                  'Crazy, Stupid, Love', 'Easy A', 'Marley & Me (film)',
                  'No Strings Attached (film)', 'Arbitrage (film)', 'Panic Room',
                   'The Shining (film)', 'One Eight Seven', 'Black Swan (film)',
                  'Gothika', 'Heat (1995 film)', 'House of 1000 Corpses',
                  'Sleepy Hollow (film)', 'The Talented Mr',
                  'The Thing (1982 film)', 'Die Hard',
                  'Soldier (1998 American film)', 'The Crying Game',
                  'Total Recall (1990 film)', '2012 (film)',
                  'From Russia with Love (film)',
                  'American Gangster (film)', 'Collateral Damage (film)',
                  'Oblivion (2013 film)','Moon (film)', 'Slumdog Millionaire',
                  'The Last Temptation of Christ (film)', 'Unforgiven',
                  'American Beauty (film)', 'Jane Eyre (2011 film)',
                  'The Majestic (film)', 'A Walk to Remember'},
                        help='Movie name for graph visualization')

    args = parser.parse_args()

    model_folder = args.folder
    model_tested = args.folder
    movie = args.movie

    movie_names = []
    graph_connections = []
    labels = []
    tps = []

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

    id = movie_names.index(movie)

    markers = []
    for x, tp in enumerate(tps[id]):
        markers.append(tp)
    G_helper = construct_movie_graph(markers, graph_connections, id)

    colors = ['mediumaquamarine', 'seagreen', 'goldenrod', 'red', 'lightcoral']

    color_map = []
    for node in G_helper:
        if node in markers[0]:
            color_map.append(colors[0])
        elif node in markers[1]:
            color_map.append(colors[1])
        elif node in markers[2]:
            color_map.append(colors[2])
        elif node in markers[3]:
            color_map.append(colors[3])
        elif node in markers[4]:
            color_map.append(colors[4])
        else:
            color_map.append('lightgrey')

    nx.draw(G_helper, node_color=color_map, edge_color='grey', alpha=1,
            with_labels=True, font_color='grey')

    destination_folder = os.path.join(output_folder, 'movie_graphs')

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    plt.savefig(os.path.join(destination_folder, movie + ".pdf"))
