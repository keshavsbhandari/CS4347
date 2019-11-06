from torchvision.transforms import *
size = (128, 128)
rand_transform = Compose([
    RandomApply([ColorJitter(),
                  RandomGrayscale(),
                  RandomAffine(degrees=(-90, +90)),
                  RandomCrop(size),
                  RandomHorizontalFlip(0.1),
                  RandomPerspective(p=0.1),
                  ]),
    Resize(size),
    ToTensor()])

train_args = dict(batch_size=10,
                  pin_memory=True,
                  num_workers=0,
                  )
test_args = dict(batch_size=1,
                  pin_memory=True,
                  num_workers=0,
                  )
