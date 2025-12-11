import torch
import numpy as np
import os
from .sampler import ClassAwareSampler

class Timesig_loader(dataloader.Dataset):
    def __init__(self, path, transform=None):
        data = np.load(path)
        self.x = data['X']
        self.y = data['y']
        self.num_classes = 4
        self.cls_num_list = [np.sum(np.array(self.y)==i) for i in range(self.num_classes)]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return index, self.x[index], self.y[index] # Return index as well to follow lnr convention

class Timesig(object):

    def __init__(self, distributed, root='./data/timesig', batch_size=128, num_works=40, config = None):      
        train_dataset = Timesig_loader(os.path.join(root, "train_data.npz"))
        val_dataset = Timesig_loader(os.path.join(root, "val_data.npz"))
        eval_dataset = Timesig_loader(os.path.join(root, "test_data.npz"))

        self.cls_num_list = train_dataset.cls_num_list

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)
        self.val =  val_dataset
        self.train_dataset = train_dataset
        print('train',self.cls_num_list)