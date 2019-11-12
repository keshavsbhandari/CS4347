import os
maindir = os.path.realpath('.')

"""
DATA PATH
"""
train_path = f'{maindir}/data/hotdog/train/'
test_path = f'{maindir}/data/hotdog/test/'
train_path101 = f'{maindir}/data/food101small/train/'
test_path101 = f'{maindir}/data/food101small/test/'

epoch = 100
model_path = f'{maindir}/archive/best.pth'
metrics_path= f'{maindir}/archive/metrics.txt'

model_path101 = f'{maindir}/archive/best101.pth'
metrics_path101 = f'{maindir}/archive/metrics101.txt'

lr = 0.01

# Note this can be done programmatically in more smarter way
label_101_dict = {'apple_pie': 0,
                  'baby_back_ribs': 1,
                  'baklava': 2,
                  'beef_carpaccio': 3,
                  'beef_tartare': 4,
                  'beet_salad': 5,
                  'beignets': 6,
                  'bibimbap': 7,
                  'bread_pudding': 8,
                  'breakfast_burrito': 9}

