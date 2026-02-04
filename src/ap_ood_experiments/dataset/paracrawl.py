import logging

import torch


logger = logging.getLogger(__name__)

class ParacrawlDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, src_lang, trg_lang, split='train'):
        self.file_path = file_path
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.split = split

        logger.info(f'Loading Paracrawl dataset from {file_path}')
        with open(self.file_path) as f:
            self.segments = self.__parse_file(f)
            # file = ''.join([next(f) for _ in range(lines_number)])

        self.segments = self._get_split(self.segments)

    def _get_split(self, dataset):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            train_split, test_split = torch.utils.data.random_split(dataset, [len(dataset) - 10_000, 10_000])

        if self.split not in ['train', 'validation']:
            raise ValueError('split must be either "train" or "validation"')
        return test_split if self.split == 'validation' else train_split

    def __parse_file(self, file):
        split_sentences_ = [line.split('\t') for line in file]
        split_sentences = [sentence for sentence in split_sentences_ if len(sentence) == 2]
        return split_sentences

    def __getitem__(self, i):
        src, target = self.segments[i]
        return {'translation': {self.src_lang: src, self.trg_lang: target}}

    def __len__(self):
        return len(self.segments)
