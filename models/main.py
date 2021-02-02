import os
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import pickle

import csv
import sys
from sklearn.model_selection import KFold

from tabulate import tabulate

sys.path.append('../')
from models.baseline_trainer import Trainer_TP_identification, Trainer_simpler
from modules.data.collates import MovieCollate_multimodal
from modules.data.datasets import VideoMovies
from modules.graph_models import GraphTP, TAM
from sys_config import MODEL_CNF_DIR
from utils.generic import number_h
from utils.opts import train_options

from torch.utils.tensorboard import SummaryWriter


def find_minimum_distance(list1, list2):
    minimum_distance = 10000
    for item1 in list1:
        for item2 in list2:
            if abs(item1 - item2) < minimum_distance:
                minimum_distance = abs(item1 - item2)
    return minimum_distance


def evaluate_tps(posteriors, labels, neighborhood):

    # # #
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


default_config = os.path.join(MODEL_CNF_DIR, "GraphTP.yaml")
opts, config = train_options(default_config)

final_scores = {'pa': [], 'ta': [], 'distance': []}

model_name_now = config["model_name"]

data_folder = config["data"]["folder"]

with open(config["data"]["train_labels"], "rb") as f:
    train_labels = pickle.load(f)
with open(config["data"]["test_labels"], "rb") as f:
    test_labels = pickle.load(f)

data_split = {}
with open('../dataset/splits.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data_split.update({row[0]: row[1]})

all_test_ids = list(range(38))

kfold = KFold(5, True, 1)

_set = 1

for indices_dev, indices_test in kfold.split(all_test_ids):

    final_scores['ta'].append(0)
    final_scores['pa'].append(0)
    final_scores['distance'].append(0)

    output_folder = config["data"]["output_folder"] + model_name_now + \
                    '_' + str(_set) + '/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tensorboard_output = config["data"]["tensorboard_outputs"] + \
                         model_name_now + '_' + str(_set)

    if not os.path.exists(tensorboard_output):
        os.makedirs(tensorboard_output)

    writer = SummaryWriter(log_dir=tensorboard_output)

    print("Cross-validation: Fold number: %d" % _set)

    print("Building training dataset...")
    train_set = VideoMovies(data_folder, data_split, 'train',
                          max_scene_length=config["data"]["max_scene_length"],
                          noisy_distrs=train_labels)

    print("Building validation dataset...")
    dev_set = VideoMovies(data_folder, data_split, 'test',
                        max_scene_length=config["data"]["max_scene_length"],
                        labels=test_labels, indices=indices_dev)

    print("Building test dataset...")
    test_set = VideoMovies(data_folder, data_split, 'test',
                           max_scene_length=config["data"]["max_scene_length"],
                           labels=test_labels, indices=indices_test)

    train_loader = DataLoader(train_set, shuffle=True,
                              collate_fn=MovieCollate_multimodal())
    dev_loader = DataLoader(dev_set, shuffle=False,
                            collate_fn=MovieCollate_multimodal())
    val_loader = DataLoader(test_set, shuffle=False,
                            collate_fn=MovieCollate_multimodal())

    ####################################################################
    # Model
    ####################################################################

    if config["model"]["name"] == 'GraphTP':
        model = GraphTP(config["model"])
    else:
        model = TAM(config["model"], window_length=0.2)
    model.to(opts.device)

    print(model)

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.Adam(parameters)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters()
                                 if p.requires_grad)

    print("Total Params:", number_h(total_params))
    print("Total Trainable Params:", number_h(total_trainable_params))

    ####################################################################
    # Training Pipeline
    ####################################################################

    # Trainer: responsible for managing the training process
    if config["model"]["name"] == 'GraphTP':
        trainer = Trainer_TP_identification(val_loader, model, train_loader,
                                        dev_loader, [optimizer], config,
                                        opts.device)
    else:
        trainer = Trainer_simpler(val_loader, model, train_loader,
                                        dev_loader, [optimizer], config,
                                        opts.device)

    ####################################################################
    # Training Loop
    ####################################################################
    best_loss = None
    best_score = 0
    final_posteriors = []
    final_neighborhood = []
    groundtruth = []
    graph_neighborhood = []

    print("Movies for evaluation:")
    print(test_set.data)

    for epoch in range(config["epochs"]):

        train_loss = trainer.train_epoch(epoch)
        val_loss, posteriors, top_neighborhood, labels = trainer.eval_epoch()
        test_loss, posteriors_test, top_neighborhood_test, labels_test, \
        graph_neighborhoods = trainer.eval_epoch_test()

        ground_truth = labels
        ground_truth_test = labels_test

        distance_val, pa_val, score = evaluate_tps(posteriors, labels,
                                                   top_neighborhood)
        distance_test, pa_test, score_test = evaluate_tps(posteriors_test,
                                                          labels_test,
                                                          top_neighborhood_test)
        print()
        if score >= best_score and epoch > 5:

            best_score = score
            final_posteriors = posteriors_test
            groundtruth = ground_truth_test
            final_neighborhood = top_neighborhood_test
            graph_neighborhood = graph_neighborhoods

            final_scores['ta'][-1] = score_test
            final_scores['pa'][-1] = pa_test
            final_scores['distance'][-1] = distance_test

            train_neighborhoods, train_tps = trainer.keep_train_graphs()

            trainer.checkpoint()

        table = [["Dev loss", val_loss], ["TA_dev", score],
                 ["PA_dev", pa_val], ["TA_test", score_test],
                 ["PA_test", pa_test], ["Train loss", train_loss]]
        print(tabulate(table))

        print("\n" * 2)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('TA/val', score, epoch)
        writer.add_scalar('TA/test', score_test, epoch)
        writer.add_scalar('PA/val', pa_val, epoch)
        writer.add_scalar('PA/test', pa_test, epoch)
        writer.add_scalar('Distance/val', distance_val, epoch)
        writer.add_scalar('Distance/test', distance_test, epoch)

    ## save to pickle files

    with open((output_folder + 'movie_names_' + model_name_now + '.pkl'),
              'wb') as f:
        pickle.dump(test_set.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open((output_folder + 'posteriors_' + model_name_now + '.pkl'),
              'wb') as f:
        pickle.dump(final_posteriors, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open((output_folder + 'labels_' + model_name_now + '.pkl'),
              'wb') as f:
        pickle.dump(groundtruth, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open((output_folder + 'turning_points_' + model_name_now + '.pkl'),
              'wb') as f:
        pickle.dump(final_neighborhood, f, protocol=pickle.HIGHEST_PROTOCOL)

    if config["model"]["name"] == 'GraphTP':
        with open((output_folder + 'graph_neighbors_' + model_name_now + '.pkl'),
              'wb') as f:
            pickle.dump(graph_neighborhood, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open((output_folder + 'graph_neighbors_training_set_' + model_name_now
                   + '.pkl'), 'wb') as f:
            pickle.dump(train_neighborhoods, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open((output_folder + 'turning_points_training_set_' + model_name_now
                   + '.pkl'), 'wb') as f:
            pickle.dump(train_neighborhoods, f, protocol=pickle.HIGHEST_PROTOCOL)

    _set += 1

print("Final cross-validation scores:")
print("Total Agreement: %.4f \n" % np.asarray(final_scores['ta']).mean())
print("Partial Agreement: %.4f \n" % np.asarray(final_scores['pa']).mean())
print("Distance: %.4f \n" % np.asarray(final_scores['distance']).mean())
