import pandas as pd
import random
from itertools import islice
from utils.RandomDataGenerator import getrandxy
from utils.RandomDataGenerator import Data


def chunkify(it, size):
    it = iter(it)
    return [*map(lambda x: list(x), iter(lambda: tuple(islice(it, size)), ()))]

"""
Dataloader is an important part of machine learning pipeline
dataloader creates data input pipelines
here data is read from train_path and test_path

batch_size if given will process your data in batch

remember this dataset has implemented kfolds with simplest approach, 
we took only random validation set each time, however, during trianing this is implemented for k = 1 only
you can repeat n iterations to as many k times resulting k*n times iterations 
per every k there is one validation set 
"""

class DataLoader(object):
    def __init__(self, train_path = None,
                 test_path = None,
                 batch_size=None,
                 random_mode = False,
                 n = 100,
                 xdim = 1,
                 ydim = 1):
        self.train_path = train_path
        self.batch_size = batch_size
        self.test_path = test_path
        self.random_mode = random_mode

        if not random_mode:
            self.test = pd.read_csv(self.test_path)
            self.valsize = len(self.test)
            self.gettest()
        else:
            self.data = getrandxy(n, xdim, ydim)

    def __getTrainVal(self, df):
        valindex, trainindex = self.__getKFoldIndex(list(df.index))
        return df.iloc[trainindex], df.iloc[valindex]

    def __init_kfolds(self):
        df = pd.read_csv(self.train_path)
        self.train, self.val = self.__getTrainVal(df)
        self.train.reset_index(drop=True, inplace=True)

    def __getxy(self, df):
        label = df.pop('label').values.reshape(-1,1)
        return df.values, label

    def getval(self):
        self.valset = self.__getxy(self.val)
        return  self.valset

    def gettest(self):
        self.testset = self.__getxy(self.test)
        return self.testset

    def get_random(self):
        return self.data

    def __iter__(self):
        if self.random_mode:
            yield self.get_random()
        else:
            self.__init_kfolds()
            self.getval()
            if self.batch_size is None:
                data = Data()
                data.trainx, data.trainy = self.__getxy(self.train)
                data.valx, data.valy = self.valset
                data.testx, data.testy = self.testset
                yield data

            else:
                chunks = chunkify(list(self.train.index), self.batch_size)
                for i,chunk in enumerate(chunks):
                    data = Data()
                    data.trainx, data.trainy = self.__getxy(self.train.iloc[chunk])
                    data.valx, data.valy = self.valset
                    data.testx, data.testy = self.testset
                    data.i = i
                    data.batchnum = len(chunks)
                    yield data

    def __getKFoldIndex(self, index):
        index = index.copy()
        valin = random.sample(index, self.valsize)
        [*map(lambda t: index.remove(t), valin)]
        return valin, index