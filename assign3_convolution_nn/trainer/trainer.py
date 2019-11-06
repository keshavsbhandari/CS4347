from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import functional as F
from configs import epoch, model_path, lr, metrics_path
import torch
import os
from tqdm import tqdm
from utils import AverageMeter

from dataloader.hotdogloader import HotDogDataSetLoader
from models.simpleclassifier import NaiveDLClassifier


from os import environ as env
if torch.cuda.is_available():
    GPUS = env["CUDA_VISIBLE_DEVICES"]
    GPUS_LIST = list(range(GPUS).split(','))
else:
    GPUS_LIST = []

class HotDogTrainer(object):
    def __init__(self):
        super(HotDogTrainer, self).__init__()
        self.model = NaiveDLClassifier()
        self.epoch = epoch
        self.data = HotDogDataSetLoader()
        self.gpu_ids = GPUS_LIST
        self.load_model_path = model_path
        self.stat_cache = None
        self.global_step = 9
        self.writer = SummaryWriter()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, len(self.data.train()))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss = torch.nn.BCELoss()

    def initialize(self):
        if torch.cuda.is_available():
            self.model.to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
        if self.load_model_path:
            if os.path.exists(self.load_model_path):
                self.load_old_best()

    def savemodel(self, metrics):
        import json
        with open(metrics_path,'w') as f:
            json.dump(metrics, f)
        if torch.cuda.is_available():
            torch.save(self.model.module.state_dict(), self.load_model_path)
        else:
            torch.save(self.model.state_dict(), self.load_model_path)

    def train(self, nb_epoch):
        trainstream = tqdm(self.data.train())
        self.avg_loss = AverageMeter()
        self.avg_acc = AverageMeter()
        self.model.train()

        for i, data in enumerate(trainstream):
            self.global_step += 1
            trainstream.set_description("TRAINING")

            x = data['image'].to(self.device)
            y = data['label'].to(self.device).unsqueeze(-1)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                y_ = self.model.float()(x)
                loss = self.loss(y_, y.float())

                y_ = (y_ >= 0.5).int().flatten()
                acc = 100. * ((y_ == y.flatten()).sum()) / float(len(y_))
                self.avg_acc.update(acc)
                self.avg_loss.update(loss.item())

                loss.backward()

                self.optimizer.step()

                self.writer.add_scalar('Loss/Train', self.avg_loss.avg, self.global_step)
                self.writer.add_scalar('Accuracy/Train', self.avg_acc.avg, self.global_step)

                trainstream.set_postfix({'epoch':nb_epoch,
                                         'loss': self.avg_loss.avg,
                                         'accuracy':self.avg_acc.avg})
        self.scheduler.step()
        trainstream.close()
        self.test(nb_epoch)

    def test(self, nb_epoch):
        self.model.eval()
        teststream = tqdm(self.data.test())

        self.avg_loss = AverageMeter()
        self.avg_acc = AverageMeter()

        teststream.set_description('TESTING')
        with torch.no_grad():
            for i, data in enumerate(teststream):
                x = data['image']
                y = data['label'].unsqueeze(-1)
                y_ = self.model(x)
                loss = self.loss(y_, y.float())
                y_ = (y_ >= 0.5).int().flatten()
                acc = 100. * ((y_ == y.flatten()).sum()) / float(len(y_))
                self.avg_acc.update(acc)

                self.avg_acc.update(acc)
                self.avg_loss.update(loss.item())

                teststream.set_postfix({'epoch': nb_epoch,
                                        'loss': self.avg_loss.avg,
                                        'accuracy': self.avg_acc.avg})

        self.writer.add_scalar('Loss/Test', self.avg_loss.avg, nb_epoch)
        self.writer.add_scalar('Accuracy/Test', self.avg_acc.avg, nb_epoch)



        if not self.stat_cache:
            self.stat_cache = {'best':self.avg_acc.avg}
            print('SAVING MODEL')
            self.savemodel({'best':self.avg_acc.avg})
        else:
            if self.stat_cache['best'] < self.avg_acc.avg:
                print('LOADING BEST MODEL')
                self.load_old_best()

    def load_old_best(self):
        import json
        with open(metrics_path, 'r') as f:
            self.stat_cache = json.load(f)

        if torch.cuda.is_available():
            self.model.module.load_state_dict(torch.load(self.load_model_path))
        else:
            self.model.load_state_dict(torch.load(self.load_model_path))

    def run(self):
        self.initialize()
        for i in range(self.epoch):
            self.train(i)
        self.writer.close()