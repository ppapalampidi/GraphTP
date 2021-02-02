import _pickle as pickle
import os
import copy
from operator import itemgetter

import numpy
from torch.utils.data import Dataset

from sys_config import DATA_DIR

from utils.nlp import vectorize, vectorize_doc_with_vocab
import scipy.stats as stats


class VideoMovies(Dataset):

    def __init__(self, input_folder, splits, set_now,
                 max_scene_length=50, noisy_distrs=None, labels=None,
                 indices=None, **kwargs):

        """
        Dataset for multi-modal TP identification on movies from TRIPOD
        :param input_folder: folder with multimodal features per TRIPOD movie
        :param noisy_distrs: probability distributions per movie and TP
                            as computed by the teacher model
        :param labels: gold-standard labels for TPs
                       (for TRIPOD available only for the test set)
        """

        self.folder = input_folder

        self.set = set_now

        self.max_length = max_scene_length

        _, _, filenames = next(os.walk(input_folder))

        data = [x.split('.')[0] for x in filenames if splits[x.split('.')[0]] == set_now]

        if self.set == 'train':
            self.data = []
        else:
            self.data = [x for m, x in enumerate(data) if m in indices]

        self.teacher_logits = []

        if noisy_distrs != None:
            print("constructing teacher distributions...")
            self._assign_noisy_distributions(noisy_distrs)
        else:
            print("storing gold TP indices...")
            self._assign_gold_labels(labels)

        print(len(self.data))


    def _assign_gold_labels(self, labels):

        for movie_name in self.data:

            v = labels[movie_name]

            self.teacher_logits.append(v)


    def _assign_noisy_distributions(self, noisy_labels):

        for movie_name, value in noisy_labels.items():

            name = movie_name.split('_')[0]

            if name == 'Pirates of the Caribbean':
                name = 'Pirates of the Caribbean_ The Curse of the Black Pearl'

            self.data.append(name)
            self.teacher_logits.append(value)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        with open(os.path.join(self.folder, self.data[index] + '.pkl'), "rb") as f:
            sample = pickle.load(f)

        script = sample['text']

        script = [_x[:self.max_length] for _x in script if (not numpy.isnan(_x).any())]

        audio = sample['audio']
        vision = sample['vision']
        if self.set == 'train':
            labels = self.teacher_logits[index]
        else:
            labels = numpy.zeros((len(self.teacher_logits[index]), len(script)))
            for i, ids in enumerate(self.teacher_logits[index]):
                for id in ids:
                    labels[i][id] = 1

        return script, [len(x) for x in script], audio, vision, labels



