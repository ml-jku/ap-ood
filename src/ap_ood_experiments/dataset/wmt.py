import datasets
import torch


# There is a slight distribution shift between the train and test set of the WMT15 dataset on Hugging Face. 
# Therefore, we manually split the train set into a new train and test set to ensure the I.I.D. assumption is satisfied.
class WMTDataset(torch.utils.data.Dataset):
    def __init__(self, path='wmt/wmt15', name='fr-en', split='train', cache_dir=None):
        dataset = datasets.load_dataset(path, name=name, split='train', cache_dir=cache_dir)
        self.split = split
        assert split in ['train', 'test']
        self.dataset = self._get_split(dataset)

    def _get_split(self, dataset):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            train_split, test_split = torch.utils.data.random_split(dataset, [len(dataset) - 10_000, 10_000])
            return train_split if self.split == 'train' else test_split
    
    def __getitem__(self, i):
        return self.dataset[i]
    
    def __len__(self):
        return len(self.dataset)
