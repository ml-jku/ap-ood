from bs4 import BeautifulSoup
import torch

class WMT15Dataset(torch.utils.data.Dataset):
    def __init__(self, src_path, trg_path, src_lang, trg_lang, document_tag='doc'):
        self.src = src_path
        self.trg = trg_path
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.document_tag = document_tag

        with open(self.src) as f:
            src_file = f.read()

        with open(self.trg) as f:
            trg_file = f.read()

        self.src_segments, self.trg_segments = self.__parse_files(src_file, trg_file)

    def __parse_files(self, src_file, trg_file):
        src = BeautifulSoup(src_file, 'xml')
        trg= BeautifulSoup(trg_file, 'xml')
        src_documents = src.find_all(self.document_tag)
        trg_documents = trg.find_all(self.document_tag)
        assert len(src_documents) == len(trg_documents)

        src_segments = []
        trg_segments = []
        for src_document, trg_document in zip(src_documents, trg_documents):
            doc_src_segments = [s.contents[0] for s in src_document.find_all('seg')]
            doc_trg_segments = [s.contents[0] for s in trg_document.find_all('seg')]
            assert len(doc_src_segments) == len(doc_trg_segments)
            src_segments.extend(doc_src_segments)
            trg_segments.extend(doc_trg_segments)
        assert len(src_segments) == len(trg_segments)
        return src_segments, trg_segments

    def __getitem__(self, i):
        src_segment = self.src_segments[i]
        trg_segment = self.trg_segments[i]
        return {'translation': {self.src_lang: src_segment, self.trg_lang: trg_segment}}

    def __len__(self):
        return len(self.src_segments)
