import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd

"""
Readers are encourage to understand the following DynamicPlotter class
Though this is not a nice way of doing, but it does our basic work in a simplest way
try to study and if you can re-write in smarter way
"""

class DynamicPlotter(object):
    def __init__(self, use_random):
        self.use_random = use_random

    def plot_linear(self,**kwargs):
        trainx, trainy, trainy_, trainy_e = [*map(lambda x:kwargs.get(x), ["trainx", "trainy", "trainy_","trainy_e"])]
        testx, testy, testy_, testy_e = [*map(lambda x: kwargs.get(x), ["testx", "testy", "testy_","testy_e"])]
        valx, valy, valy_,valy_e = [*map(lambda x: kwargs.get(x), ["valx", "valy", "valy_","valy_e"])]

        plt.scatter(trainx, trainy, label='real')
        plt.scatter(trainx, trainy_, label='predicted')
        plt.scatter(trainx, trainy_e, label='analytical')
        plt.title('ON TRAIN DATA')
        plt.legend()
        plt.show()

        plt.scatter(testx, testy, label='real')
        plt.scatter(testx, testy_, label='predicted')
        plt.scatter(testx, testy_e, label = 'analytical')
        plt.title('ON TEST DATA')
        plt.legend()
        plt.show()


        plt.scatter(valx, valy, label='real')
        plt.scatter(valx, valy_, label='predicted')
        plt.scatter(valx, valy_e, label='analytical')
        plt.title("ON VALIDATION DATA")
        plt.legend()
        plt.show()


    def plot_graph(self, rmse_train,r2_train,cost_train,rmse_val,r2_val,cost_val,rmse_test,r2_test,cost_test):
        plt.plot(rmse_train, label='train')
        plt.plot(rmse_test, label='test')
        plt.plot(rmse_val, label='val')
        plt.title('RMSE')
        plt.legend()
        plt.show()

        plt.plot(r2_train, label='train')
        plt.plot(r2_test, label='test')
        plt.plot(r2_val, label='val')
        plt.title('R2 Score')
        plt.legend()
        plt.show()

        plt.plot(cost_train, label='train')
        plt.plot(cost_test, label='test')
        plt.plot(cost_val, label='val')
        plt.title('COST')
        plt.legend()
        plt.show()


    def print_info(self, i):
        stats = [dict(iteration = i ,mode='train', **self.train),
                 dict(iteration = i, mode='test', **self.test),
                 dict(iteration = i, mode='val', **self.val)]

        statdf = pd.DataFrame(stats)

        statdf = statdf[['iteration','mode','cost','r2_score','rmse']]

        print(tabulate(statdf, headers = statdf.columns, showindex = False, tablefmt = 'grid'))

    def plot_stat(self,train, test, val, ith_batch):
        self.train = train
        self.val = val
        self.test = test
        self.print_info(ith_batch)

