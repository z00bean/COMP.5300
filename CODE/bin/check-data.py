import sys
import os
import json
from easydict import EasyDict as edict

import torchvision
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from data.dataset import ChexpertDataset

import matplotlib.pyplot as plt
import numpy as np

def show(idx):
    img = images[idx] #* conf.pixel_std + conf.pixel_mean     # unnormalize
    #img = images[2] + conf.pixel_mean
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    configPath = os.path.dirname(os.path.abspath(__file__)) + '/../config/conf1.json'
    print("configPath = ", configPath)

    with open(configPath) as f:
        conf = edict(json.load(f))
    print("conf.long_side = ", conf.long_side)
        
    _chexpertDataset = ChexpertDataset(conf, conf.dev_csv, phase = 'train', u_approach = 0)
    dataloader_train = DataLoader(
            _chexpertDataset,
            batch_size=conf.train_batch_size, num_workers=conf.num_workers,
            drop_last=True, shuffle=True)

    dataiter = iter(dataloader_train)
    images, labels = dataiter.next()
    #imshow(torchvision.utils.make_grid(images), conf.pixel_std, conf.pixel_mean)

    print(' '.join('%d \t %s\n' % (j, labels[j]) for j in range(len(labels))))

    show(7)

