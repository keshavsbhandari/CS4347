from extra_files import (showrandom_batch,show_random_data)
from trainer.trainer101 import HotDogTrainer

if __name__ == '__main__':
    """
    Visualize random data
    """
    # show_random_data()

    """
    Visualize Random Dataloader 
    """
    # showrandom_batch()

    htd = HotDogTrainer()
    #
    htd.run()
