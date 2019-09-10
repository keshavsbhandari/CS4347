import numpy as np
from utils.DynamicPlotter import DynamicPlotter
from utils.RandomDataGenerator import analytical


E = lambda x: np.insert(x, 0, 1, axis=1)

"""
LINEAR CLASS IMPLEMENTS Linear Regression 
xdim and ydim are no. of features in x and y,
alpha is a learning rate
lmbda is a regularization parameter
"""

class Linear(object):
    def __init__(self, iterations, xdim=1, ydim=1, alpha = 0.1, lmbda = 0, use_random = False):
        self.iterations = iterations
        self.theta = np.random.rand(xdim + 1, ydim)
        """
        QUESITON NO. 1
        
        DEFINE One line of PREDICT FUNCTION USING LAMBDA OPERATIONS
        self.predict = lambda x:____??
        """
        self.predict = 

        self.alpha = alpha
        self.lmbda = lmbda
        self.plotter = DynamicPlotter(use_random)



    def tapeit(self):
        self.records['train']['rmse'].append(self.train_stat['rmse'].item())
        self.records['train']['r2_score'].append(self.train_stat['r2_score'].item())
        self.records['train']['cost'].append(self.train_stat['cost'].item())
        self.records['test']['rmse'].append(self.test_stat['rmse'].item())
        self.records['test']['r2_score'].append(self.test_stat['r2_score'].item())
        self.records['test']['cost'].append(self.test_stat['cost'].item())
        self.records['val']['rmse'].append(self.val_stat['rmse'].item())
        self.records['val']['r2_score'].append(self.val_stat['r2_score'].item())
        self.records['val']['cost'].append(self.val_stat['cost'].item())

    """
    HERE x_ is training data and y is a true label
    atevery_step helps to print information [NOT IMPORTANT FOR NOW]
    """
    def gradient_descent(self, x_, y, verbose = False, atevery_step = 10, **kwargs):
        x = E(x_)
        self.records = {'train': {'rmse': [], 'r2_score': [], 'cost': []},
                        'test': {'rmse': [], 'r2_score': [], 'cost': []},
                        'val': {'rmse': [], 'r2_score': [], 'cost': []}}
        for i in range(self.iterations):
            """
            WRITE YOUR CODE HERE
            QUESITON NO. 2
            
            1. calcluate prediction using x and theta
            2. calculate error
            3. calculate cost function whish is sum of squares of errors divided by 2 times length of x
            4. update theta and add regularization parameter
            
            uncomment and add your code in following statement
            
            NOTE NOTE NOTE
            NOTE YOU CAN WRITE MULTIPLE LINES OF CODE EXCEPT UNLESS MENTIONED EXPLICITLY
            
            """

            # prediction =
            # error =
            # cost =
            # self.theta =



            print('steps : {} cost : {}'.format(i,cost.item())) if ((not i % atevery_step) and verbose) else None

            if verbose:
                self.train_stat = self.evaluate(x_, y)
                self.test_stat = self.evaluate(**kwargs.get('test'))
                self.val_stat = self.evaluate(**kwargs.get('val'))
                self.tapeit()
                self.plotter.plot_stat(self.train_stat, self.test_stat, self.val_stat, i)


        if verbose:
            self.plotter.plot_graph(rmse_train=self.records['train']['rmse'],
                                    r2_train=self.records['train']['r2_score'],
                                    cost_train=self.records['train']['cost'],
                                    rmse_val=self.records['val']['rmse'],
                                    r2_val=self.records['val']['r2_score'],
                                    cost_val=self.records['val']['cost'],
                                    rmse_test=self.records['test']['rmse'],
                                    r2_test=self.records['test']['r2_score'],
                                    cost_test=self.records['test']['cost'])

        if kwargs.get('unimodal'):
            testx,testy = kwargs.get('test')['x'], kwargs.get('test')['y']
            valx, valy = kwargs.get('val')['x'], kwargs.get('val')['y']

            self.plotter.plot_linear(trainx = x_, trainy = y, trainy_ = self.predict(x_),trainy_e = analytical(x_,y),
                                     testx = testx ,testy= testy, testy_ = self.predict(testx),testy_e = analytical(testx,testy),
                                     valx = valx, valy = valy, valy_ = self.predict(valx), valy_e = analytical(valx,valy))
        return self.theta, self.records

    def __rmse(self, y, y_pred):
        """
        QUESTION NO. 3
        :param y:true label
        :param y_pred: predicted label
        :return: return square root of( sum of square of error divide by length of y)

        uncomment and return rmse

        """
        #rmse =
        return rmse

    def __r2_score(self, y, y_pred):

        """
        QUESTION NO. 4
        :param y: true label
        :param y_pred: predicted label
        :return: should be r2_score
        How to calcluate r2 score
            1. calculate ss_tot(total sum of squares) which is sum of square of difference of real y and mean of real y
            2. calculate ss_res(total sum of residue) which is sum of square of difference of real  y and pred y
            3. r2 score is 1 - ratio of ss_res and ss_tot

            uncomment following lines and add your version of code
        """

        # mean_y =
        # ss_tot =
        # ss_res =
        # r2 =

        return r2

    def __get_cost(self,y,y_pred):
        """
        QUESTION NO. 5
        :param y:true label
        :param y_pred: predicted label
        :return: should return cost

        How to calculate cost:
            1. calculate error which is difference of y and y_pred
            2. calcluate sum  of square of error and divide by (2*length of y)
            uncomment and add your version of code below
        """
        # error
        # cost =
        return cost



    def evaluate(self,x,y):
        y_pred = self.predict(x)
        return {'rmse':self.__rmse(y, y_pred),
                'r2_score':self.__r2_score(y,y_pred),
                'cost':self.__get_cost(y,y_pred)
                }

