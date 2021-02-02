import numpy

import torch
import torch.nn as nn
from modules.trainer import Trainer
from torch.nn import functional as F
import scipy.stats as stats



class Trainer_TP_identification(Trainer):

    def __init__(self, test_loader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_loader = test_loader

    def _process_batch(self, script, scene_lengths, audio, vision, labels):
        sa, neighborhood_distribution, neighborhood_decision \
            = self.model(script,scene_lengths, audio, vision, self.device)

        criterion = nn.KLDivLoss()
        neighborhood_loss = 0
        all_losses = []
        sigma = 0.01

        for i in range(neighborhood_distribution.size(0)):
            current_distr = F.softmax(neighborhood_distribution[i], dim=0)
            dist = torch.from_numpy(
                stats.norm.pdf(list(range(current_distr.size(0))),
                               (i),
                               sigma * current_distr.size(0))). \
                float().cuda(sa.get_device())
            dist = F.softmax(dist/0.05, dim=0)
            loss_now = F.kl_div(torch.log(current_distr), dist)
            all_losses.append(loss_now)
            neighborhood_loss += loss_now

        neighborhood_loss = neighborhood_loss/neighborhood_distribution.size(0)

        loss = criterion(torch.log(sa), labels)

        total_loss = loss/5 + 10*neighborhood_loss

        return total_loss, sa, neighborhood_decision, labels


    def eval_epoch(self):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()
        losses = []
        outputs = []
        labels = []

        if isinstance(self.valid_loader, (tuple, list)):
            iterator = zip(*self.valid_loader)
        else:
            iterator = self.valid_loader

        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):
                if isinstance(self.valid_loader, (tuple, list)):
                    batch = list(map(lambda x:
                                     list(map(lambda y: y.to(self.device), x)),
                                     batch))
                else:
                    batch = list(map(lambda x: x.to(self.device), batch))

                batch_losses, batch_outputs, batch_neighborhoods, batch_labels \
                    = self._process_batch(*batch)

                loss, _losses = self._aggregate_losses(batch_losses)
                losses.append(_losses)
                outputs.append(batch_outputs.data.cpu().numpy())
                labels.append(batch_labels.data.cpu().numpy())

        final_posteriors = []
        final_neighborhood = []
        final_labels = []

        for k, movie in enumerate(outputs):
            final_posteriors.append([])
            final_neighborhood.append([])
            final_labels.append([])

            posteriors = movie.tolist()

            indices = list(range(len(posteriors[0])))

            for j, tp in enumerate(posteriors):

                final_posteriors[-1].append(tp)
                indices_of_w = [x for _, x in sorted(zip(tp,
                                                         list(range(len(tp)))),
                                                     reverse=True)]
                top_post_index = indices_of_w[0]
                if top_post_index != 0 and top_post_index != (len(tp) - 1):
                    top_neighborhood = indices[(top_post_index-1):(top_post_index+2)]
                elif top_post_index == 0:
                    top_neighborhood = indices[top_post_index:(top_post_index+3)]
                else:
                    top_neighborhood = indices[(top_post_index-2):(top_post_index+1)]
                final_neighborhood[-1].append(top_neighborhood)

                final_labels[-1].append(numpy.nonzero(numpy.asarray(labels[k][j]))[0])

        return numpy.array(losses).mean(axis=0), final_posteriors, final_neighborhood, final_labels

    def eval_epoch_test(self):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()
        losses = []
        outputs = []
        neighborhoods = []
        labels = []

        if isinstance(self.test_loader, (tuple, list)):
            iterator = zip(*self.test_loader)
        else:
            iterator = self.test_loader

        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):
                # move all tensors in batch to the selected device
                if isinstance(self.test_loader, (tuple, list)):
                    batch = list(map(lambda x:
                                     list(map(lambda y: y.to(self.device), x)),
                                     batch))
                else:
                    batch = list(map(lambda x: x.to(self.device), batch))

                batch_losses, batch_outputs, batch_neighborhoods, batch_labels \
                    = self._process_batch(*batch)
                loss, _losses = self._aggregate_losses(batch_losses)
                losses.append(_losses)
                outputs.append(batch_outputs.data.cpu().numpy())
                labels.append(batch_labels.data.cpu().numpy())
                batch_neighborhoods = batch_neighborhoods.data.cpu().numpy()
                neighborhoods.append(batch_neighborhoods)


        final_posteriors = []
        final_neighborhood = []
        final_labels = []
        graph_neighborhoods = []

        for k, movie in enumerate(outputs):
            final_posteriors.append([])
            final_neighborhood.append([])
            final_labels.append([])
            graph_neighborhoods.append(neighborhoods[k])

            posteriors = movie.tolist()

            indices = list(range(len(posteriors[0])))

            for j, tp in enumerate(posteriors):

                final_posteriors[-1].append(tp)
                indices_of_w = [x for _, x in sorted(zip(tp,
                                                         list(range(len(tp)))),
                                                     reverse=True)]
                top_post_index = indices_of_w[0]
                if top_post_index != 0 and top_post_index != (len(tp) - 1):
                    top_neighborhood = indices[(top_post_index-1):(top_post_index+2)]
                elif top_post_index == 0:
                    top_neighborhood = indices[top_post_index:(top_post_index+3)]
                else:
                    top_neighborhood = indices[(top_post_index-2):(top_post_index+1)]
                final_neighborhood[-1].append(top_neighborhood)

                final_labels[-1].append(numpy.nonzero(numpy.asarray(labels[k][j]))[0])

        return numpy.array(losses).mean(axis=0), final_posteriors, \
               final_neighborhood, final_labels, graph_neighborhoods


    def keep_train_graphs(self):
        self.model.eval()
        losses = []
        outputs = []
        neighborhoods = []

        if isinstance(self.train_loader, (tuple, list)):
            iterator = zip(*self.train_loader)
        else:
            iterator = self.train_loader

        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):
                # move all tensors in batch to the selected device
                if isinstance(self.test_loader, (tuple, list)):
                    batch = list(map(lambda x:
                                     list(map(lambda y: y.to(self.device),
                                              x)),
                                     batch))
                else:
                    batch = list(map(lambda x: x.to(self.device), batch))

                batch_losses, batch_outputs, batch_neighborhoods, batch_labels \
                    = self._process_batch(
                    *batch)
                loss, _losses = self._aggregate_losses(batch_losses)
                losses.append(_losses)
                outputs.append(batch_outputs)
                batch_neighborhoods = batch_neighborhoods.data.cpu().numpy()
                neighborhoods.append(batch_neighborhoods)


        final_neighborhood = []
        graph_neighborhoods = []

        for i, movie in enumerate(outputs):

            final_neighborhood.append([])
            graph_neighborhoods.append(neighborhoods[i])

            posteriors = movie.tolist()

            indices = list(range(len(posteriors[0])))

            for j, tp in enumerate(posteriors):

                indices_of_w = [x for _, x in
                                sorted(zip(tp, list(range(len(tp)))),
                                       reverse=True)]
                top_post_index = indices_of_w[0]
                if top_post_index != 0 and top_post_index != (len(tp) - 1):
                    top_neighborhood = indices[(top_post_index - 1):(
                    top_post_index + 2)]
                elif top_post_index == 0:
                    top_neighborhood = indices[
                                       top_post_index:(top_post_index + 3)]
                else:
                    top_neighborhood = indices[(top_post_index - 2):(
                    top_post_index + 1)]
                final_neighborhood[-1].append(top_neighborhood)

        return graph_neighborhoods, final_neighborhood


class Trainer_simpler(Trainer_TP_identification):

    def _process_batch(self, script, scene_lengths, audio, vision, labels):
        sa = self.model(script, scene_lengths, audio, vision, self.device)

        criterion = nn.KLDivLoss()

        loss = criterion(torch.log(sa), labels)

        total_loss = loss / 5

        return total_loss, sa, sa, labels


