import math
import time

import numpy
import torch
from torch.nn.utils import clip_grad_norm_
from utils._logging import epoch_progress
from utils.training import save_checkpoint
from torch.nn import functional as F

class BaseTrainer:

    def __init__(self, train_loader, valid_loader,
                 config, device,
                 batch_end_callbacks=None, loss_weights=None,
                 parallel=False,
                 **kwargs):

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.loss_weights = loss_weights

        self.config = config
        self.log_interval = self.config["log_interval"]
        self.batch_size = self.config["batch_size"]
        self.checkpoint_interval = self.config["checkpoint_interval"]
        self.clip = self.config["model"]["clip"]
        self.parallel = parallel

        if batch_end_callbacks is None:
            self.batch_end_callbacks = []
        else:
            self.batch_end_callbacks = [c for c in batch_end_callbacks
                                        if callable(c)]

        self.epoch = 0
        self.step = 0
        self.progress_log = None

        # init dataset
        if isinstance(self.train_loader, (tuple, list)):
            self.train_set_size = len(self.train_loader[0].dataset)
        else:
            self.train_set_size = len(self.train_loader.dataset)

        if isinstance(self.valid_loader, (tuple, list)):
            self.val_set_size = len(self.valid_loader[0].dataset)
        else:
            self.val_set_size = len(self.valid_loader.dataset)

        self.n_batches = math.ceil(
            float(self.train_set_size) / self.batch_size)
        self.total_steps = self.n_batches * self.batch_size

    def anneal_init(self, param, steps=None):
        if isinstance(param, list):
            if steps is None:
                steps = self.total_steps
            return numpy.geomspace(param[0], param[1], num=steps).tolist()
        else:
            return param

    def anneal_step(self, param):
        if isinstance(param, list):
            try:
                _val = param[self.step]
            except:
                _val = param[-1]
        else:
            _val = param

        return _val

    def _batch_to_device(self, batch):
        if isinstance(self.train_loader, (tuple, list, numpy.ndarray)):
            batch = list(map(lambda x:
                             list(map(lambda y: y.to(self.device), x)),
                             batch))
        else:
            batch = list(map(lambda x: x.to(self.device), batch))
        return batch

    def _aggregate_losses(self, batch_losses, loss_weights=None):
        """
        This function computes a weighted sum of the models losses
        Args:
            batch_losses(torch.Tensor, tuple):

        Returns:
            loss_sum (int): the aggregation of the constituent losses
            loss_list (list, int): the constituent losses

        """
        if isinstance(batch_losses, (tuple, list)):

            if loss_weights is None:
                loss_weights = self.loss_weights

            if loss_weights is None:
                loss_sum = sum(batch_losses)
                loss_list = [x.item() for x in batch_losses]
            else:
                loss_sum = sum(w * x for x, w in
                               zip(batch_losses, loss_weights))

                loss_list = [w * x.item() for x, w in
                             zip(batch_losses, loss_weights)]
        else:
            loss_sum = batch_losses
            loss_list = batch_losses.item()
        return loss_sum, loss_list


class Trainer(BaseTrainer):
    """
    An abstract class representing a Trainer.
    A Trainer object, is responsible for handling the training process and
    provides various helper methods.

    All other trainers should subclass it.
    All subclasses should override process_batch, which handles the way
    you feed the input data to the model and performs a forward pass.
    """

    def __init__(self, model, train_loader, valid_loader,
                 optimizers, config, device,
                 batch_end_callbacks=None, loss_weights=None, **kwargs):

        super().__init__(train_loader, valid_loader, config, device,
                         batch_end_callbacks, loss_weights, **kwargs)

        self.model = model
        self.optimizers = optimizers
        self.device = device

        if not isinstance(self.optimizers, (tuple, list)):
            self.optimizers = [self.optimizers]

    def _process_batch(self, *args):
        raise NotImplementedError

    def train_epoch(self, epoch):
        """
        Train the network for one epoch and return the average loss.
        * This will be a pessimistic approximation of the true loss
        of the network, as the loss of the first batches will be higher
        than the true.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.train()

        losses = []

        self.epoch += 1
        epoch_start = time.time()

        if isinstance(self.train_loader, (tuple, list, numpy.ndarray)):
            iterator = zip(*self.train_loader)
        else:
            iterator = self.train_loader

        for i_batch, batch in enumerate(iterator, 1):

            self.step += 1

            # zero gradients
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            try:
                batch = self._batch_to_device(batch)
            except:
                continue
            batch_losses, batch_outputs, batch_neighborhoods, labels = \
                self._process_batch(*batch)

            #################FOR SECOND LOSS###############################

            batch_outputs = F.log_softmax(batch_outputs, -1)

            # aggregate the losses into a single loss value
            loss_sum, loss_list = self._aggregate_losses(batch_losses)
            losses.append(loss_list)

            loss_sum.backward()

            if self.clip is not None:
                # clip_grad_norm_(self.model.parameters(), self.clip)
                for optimizer in self.optimizers:
                    clip_grad_norm_((p for group in optimizer.param_groups
                                     for p in group['params']), self.clip)

            # update weights
            for optimizer in self.optimizers:
                optimizer.step()

            if self.step % self.log_interval == 0:
                self.progress_log = epoch_progress(self.epoch, i_batch,
                                                   self.batch_size,
                                                   self.train_set_size,
                                                   epoch_start)

            for c in self.batch_end_callbacks:
                if callable(c):
                    c(batch, losses, loss_list, batch_outputs)

        return numpy.array(losses).mean(axis=0)

    def eval_epoch(self):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()
        losses = []

        if isinstance(self.valid_loader, (tuple, list)):
            iterator = zip(*self.valid_loader)
        else:
            iterator = self.valid_loader

        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):

                # move all tensors in batch to the selected device
                if isinstance(self.valid_loader, (tuple, list)):
                    batch = list(map(lambda x:
                                     list(map(lambda y: y.to(self.device), x)),
                                     batch))
                else:
                    batch = list(map(lambda x: x.to(self.device), batch))

                batch_losses, batch_outputs = self._process_batch(*batch)

                # aggregate the losses into a single loss value
                loss, _losses = self._aggregate_losses(batch_losses)
                losses.append(_losses)

        return numpy.array(losses).mean(axis=0)

    def get_state(self):

        state = {
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model,#.state_dict(),
            "model_class": self.model.__class__.__name__,
            "optimizers": [x.state_dict() for x in self.optimizers],
            # "vocab": _vocab,
        }

        return state

    def checkpoint(self, name=None, timestamp=False, tags=None, verbose=False):

        if name is None:
            name = self.config["name"]

        return save_checkpoint(self.get_state(),
                               name=name, tag=tags, timestamp=timestamp,
                               verbose=verbose, model=self.model)
