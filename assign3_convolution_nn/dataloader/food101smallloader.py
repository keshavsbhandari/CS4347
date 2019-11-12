from configs import (train_path101, test_path101)
from utils import get_data101 as get_data
from PIL import Image
from torch.utils.data import (Dataset, DataLoader)
from dataloader import (rand_transform101, train_args, test_args)
import random
import torch
import numpy as np

class HotDogDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, datalist, transform=None):
        """
        :param transform:TRANSFORMATION FOR DATA_AUGMENTATION
        """
        self.datalist = datalist
        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        path, label = self.datalist[idx]
        image = Image.open(path)
        while np.array(image).shape[-1] != 3:
            path, label = random.sample(self.datalist, 1)[0]
            image = Image.open(path)
        sample = {'image': image}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        sample.update({'label': torch.tensor(label)})
        return sample


class HotDogDataSetLoader(object):
    def __init__(self):
        self.train_list = get_data(train_path101)
        self.test_list = get_data(test_path101)
        self.transform = rand_transform101

        self.trainsets = HotDogDataset(self.train_list, transform=rand_transform101)
        self.testsets = HotDogDataset(self.test_list, transform=rand_transform101)

    def random(self):
        random.shuffle(self.train_list)
        random.shuffle(self.test_list)

        self.trainsets = HotDogDataset(self.train_list, transform=rand_transform101)
        self.testsets = HotDogDataset(self.test_list, transform=rand_transform101)

    def train_test(self):
        self.random()
        return DataLoader(self.trainsets, **train_args), DataLoader(self.testsets, **test_args)

    def train(self):
        self.random()
        return DataLoader(self.trainsets, **train_args)

    def test(self):
        self.random()
        return DataLoader(self.testsets, **test_args)