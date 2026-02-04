import torch

class OpusDataset(torch.utils.data.Dataset):
    def __init__(self, src_path, trg_path, src_lang, trg_lang):
        self.src = src_path
        self.trg = trg_path
        self.src_lang = src_lang
        self.trg_lang = trg_lang

        with open(self.src) as f:
            src_file = f.read()

        with open(self.trg) as f:
            trg_file = f.read()

        self.src_segments, self.trg_segments = self.__parse_files(src_file, trg_file)

    def __parse_files(self, src_file, trg_file):
        src_sentences = src_file.split('\n')
        trg_sentences = trg_file.split('\n')
        assert len(src_sentences) == len(trg_sentences)
        return src_sentences, trg_sentences

    def __getitem__(self, i):
        src_segment = self.src_segments[i]
        trg_segment = self.trg_segments[i]
        return {'translation': {self.src_lang: src_segment, self.trg_lang: trg_segment}}

    def __len__(self):
        return len(self.src_segments)
