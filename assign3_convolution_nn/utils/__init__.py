from torchvision.utils import make_grid
from torchvision.transforms import (ToPILImage, ToTensor)
from PIL import Image
from pathlib import Path
import random
import torch
from configs import train_path

def get_data(path):
    """
    :param path: path to data directory
    :return: list of tuples containing datapath and label 0 for no hotdog 1 for hotdog
    """
    paths = Path(path).glob('*/**/*.jpg')
    paths = [*map(lambda x: (x.as_posix(), 1*(x.parent.name == 'hot_dog')), paths)]
    return paths

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count