from utils.DynamicPlotter import DynamicPlotter

"""
MODEL CLASS WILL TAKE dataloader and model and implement fit function to train your data

unimodal refers to the case where x has only one feature
verbose is set true for printing convenient inforomation depending upon given data

"""

class Model(object):
    def __init__(self, model, dataloader, unimodal = False, verbose = True):
        self.model = model
        self.unimodal = unimodal
        self.dataloader = dataloader
        self.verbose = verbose

        self.records = {'train': {'rmse': [], 'r2_score': [], 'cost': []},
                        'test': {'rmse': [], 'r2_score': [], 'cost': []},
                        'val': {'rmse': [], 'r2_score': [], 'cost': []}}

        if not self.dataloader.random_mode:
            self.plotter = DynamicPlotter(verbose)

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

    def fit(self):
        for i, data in enumerate(self.dataloader):
            self.model.gradient_descent(x_ = data.trainx,
                                        y = data.trainy,
                                        verbose = self.verbose,
                                        atevery_step = 10,
                                        test = {'x':data.testx,'y':data.testy},
                                        val = {'x':data.valx, 'y':data.valy},
                                        unimodal = self.unimodal
                                        )

            if not(self.dataloader.random_mode or self.unimodal or self.verbose):
                self.train_stat = self.model.evaluate(data.trainx, data.trainy)
                self.test_stat = self.model.evaluate(data.testx, data.testy)
                self.val_stat = self.model.evaluate(data.valx, data.valy)
                self.tapeit()

                self.plotter.plot_stat(train = self.train_stat,
                                       test = self.test_stat,
                                       val = self.val_stat,
                                       ith_batch = i)

        if not(self.dataloader.random_mode or self.unimodal or self.verbose):
            self.plotter.plot_graph(rmse_train = self.records['train']['rmse'],
                                    r2_train = self.records['train']['r2_score'],
                                    cost_train = self.records['train']['cost'],
                                    rmse_val = self.records['val']['rmse'],
                                    r2_val = self.records['val']['r2_score'],
                                    cost_val = self.records['val']['cost'],
                                    rmse_test = self.records['test']['rmse'],
                                    r2_test = self.records['test']['r2_score'],
                                    cost_test = self.records['test']['cost'])