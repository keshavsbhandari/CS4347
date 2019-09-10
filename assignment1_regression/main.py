import sys
from utils import readjson
from models.SimpleRergression import Linear
from utils.DataLoader import DataLoader
from models.Model import Model


if __name__ == '__main__':
    config = readjson(sys.argv[1])
    linear = Linear(**config['linear'])
    dataloader = DataLoader(**config['dataloader'])
    modal = Model(linear, dataloader, **config['modal'])
    modal.fit()