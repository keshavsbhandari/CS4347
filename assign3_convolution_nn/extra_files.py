from dataloader.hotdogloader import HotDogDataSetLoader
from torchvision.utils import make_grid
from torchvision.transforms import (ToPILImage, ToTensor)
from utils import get_data
from configs import train_path
from PIL import Image
import torch
import random

def show_random_data(nrow=4, ncol=4):
    traindata = get_data(train_path)
    hotdog = [path for path,label in traindata if label == 1]
    not_hotdog = [path for path, label in traindata if label == 0]

    sample_hotdog = random.sample(hotdog, nrow * ncol)
    sample_not_hotdog = random.sample(not_hotdog, nrow * ncol)

    sample_hotdog = [*map(lambda x:Image.open(x), sample_hotdog)]
    sample_not_hotdog = [*map(lambda x: Image.open(x), sample_not_hotdog)]

    x = [*map(lambda x:x.resize((128,128)),sample_hotdog)]
    y = [*map(lambda x:x.resize((128,128)),sample_not_hotdog)]

    x = torch.cat([ToTensor()(x_).unsqueeze(0) for x_ in x],0)
    y = torch.cat([ToTensor()(y_).unsqueeze(0) for y_ in y],0)

    xgrid = ToPILImage()(make_grid(x,nrow=nrow))
    ygrid = ToPILImage()(make_grid(y,nrow=nrow))

    xgrid.show()
    ygrid.show()

def showrandom_batch():
    dl = HotDogDataSetLoader()
    data = next(iter(dl.train()))
    ToPILImage()(make_grid(data['image'], nrow=5)).show()