import torch
from torch.nn.utils.rnn import pad_sequence


class MovieCollate_multimodal:
    def __init__(self, *args):
        super().__init__(*args)

    def pad_samples(self, vecs):
        return pad_sequence([torch.FloatTensor(x) for x in vecs], True)

    def _collate(self, script, scene_lengths, audio, vision, labels):
        """
        Important: Assumes batch_size = 1
        """
        script = self.pad_samples(script)
        scene_lengths = torch.LongTensor(scene_lengths)
        audio = self.pad_samples(audio)
        audio = torch.FloatTensor(audio)
        vision = self.pad_samples(vision)
        vision = torch.FloatTensor(vision)
        labels = torch.FloatTensor(labels)
        return script, scene_lengths, audio, vision, labels

    def __call__(self, batch):
        assert len(batch) == 1, "Works only with batch_size = 1!"
        return self._collate(*batch[0])