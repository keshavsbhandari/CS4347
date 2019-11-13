import torch.nn as nn


class NaiveDLClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(NaiveDLClassifier, self).__init__()
        self.pipe1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                             out_channels=8,
                                             kernel_size=5,
                                             stride=2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=5,
                                                stride=2))

        self.pipe2 = nn.Sequential(nn.Conv2d(in_channels=8,
                                             out_channels=16,
                                             kernel_size=3,
                                             stride=2),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3,
                                                stride=2))

        self.pipe3 = nn.Sequential(nn.Conv2d(in_channels=16,
                                             out_channels=32,
                                             kernel_size=2,
                                             stride=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2,
                                                stride=1))

        self.pipes = nn.Sequential(self.pipe1, self.pipe2, self.pipe3)

        self.predictor = nn.Sequential(nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.pipes(x)
        x = self.predictor(x.view(x.size(0), -1))
        return x




