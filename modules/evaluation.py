import os
import pickle
import numpy as np
import argparse


def find_minimum_distance(list1, list2):
    minimum_distance = 10000
    for item1 in list1:
        for item2 in list2:
            if abs(item1 - item2) < minimum_distance:
                minimum_distance = abs(item1 - item2)
    return minimum_distance


def evaluate_tps(posteriors, labels, neighborhood):

    avg_distance = []
    exact = 0
    total = 0
    agreement = 0

    for i, movie in enumerate(posteriors):
        distance_movie = []
        for k, tp in enumerate(movie):
            markers = labels[i][k]

            selected_markers = neighborhood[i][k]

            for w, item in enumerate(markers):
                total += 1
                if item in selected_markers:
                    agreement += 1
            minimum_distance = find_minimum_distance(markers, selected_markers)
            if minimum_distance == 0:
                exact += 1
            avg_distance.append(minimum_distance / len(tp))
            distance_movie.append(minimum_distance / len(tp))

    return(np.asarray(avg_distance).mean(), (exact) / (len(posteriors) * 5),
           (agreement) / total)


if __name__ == '__main__':

    output_folder = '../outputs/'

    parser = argparse.ArgumentParser(description='TP identification evaluation.')

    parser.add_argument('--folder',
                        help='model folder for evaluation')

    args = parser.parse_args()

    model_folder = args.folder
    model_tested = args.folder

    posteriors = []
    labels = []
    tps = []

    for i in range(5):

        folder_now = os.path.join(output_folder, model_folder + '_' + str(i+1))

        with open(folder_now + '/posteriors_' + model_tested + '.pkl', 'rb') \
                as handle:
            posteriors += pickle.load(handle)
        with open(folder_now + '/labels_' + model_tested + '.pkl', 'rb') \
                as handle:
            labels += pickle.load(handle)
        with open(folder_now + '/turning_points_' + model_tested + '.pkl', 'rb')\
                as handle:
            tps += pickle.load(handle)

    distance, pa, ta = evaluate_tps(posteriors, labels, tps)

    print("Results:\n")
    print("Total Agreement: %.4f \n" % ta)
    print("Partial Agreement: %.4f \n" % pa)
    print("Distance: %.4f \n" % distance)


