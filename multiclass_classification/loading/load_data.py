from datasets.cifar10 import CIFAR10_LT
from datasets.cifar100 import CIFAR100_LT
from datasets.places import Places_LT
from datasets.imagenet import ImageNet_LT
from datasets.ina2018 import iNa2018
from datasets.timesig import Timesig

def load_data(config, selmix=False, original=True):
    # Data loading code
    if config.dataset == 'cifar10':
        dataset = CIFAR10_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                             batch_size=config.batch_size, num_works=config.workers, config=config)
    # The following are not modified to run with current changes and fixes (mainly val_dataset --> self.val_dataset)
    # elif config.dataset == 'cifar100':
    #     dataset = CIFAR100_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
    #                           batch_size=config.batch_size, num_works=config.workers)

    # elif config.dataset == 'places':
    #     dataset = Places_LT(config.distributed, root=config.data_path,
    #                         batch_size=config.batch_size, num_works=config.workers)

    # elif config.dataset == 'imagenet':
    #     dataset = ImageNet_LT(config.distributed, root=config.data_path,
    #                           batch_size=config.batch_size, num_works=config.workers)

    # elif config.dataset == 'ina2018':
    #     dataset = iNa2018(config.distributed, root=config.data_path,
    #                       batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'timesig':
        dataset = Timesig(config.distributed, root=config.data_path, batch_size=config.batch_size, num_works=config.workers, config=config)


    train_loader = dataset.train_balance
    train_loader_all = dataset.train_balance
    if config.valtestsplit:
        val_loader = dataset.val
    else:
        val_loader = dataset.eval
    cls_num_list = dataset.cls_num_list
    sel_val = dataset.val_dataset
    train_dataset = dataset.train_dataset

    if selmix:
        return train_loader, train_loader_all, val_loader, cls_num_list, sel_val, train_dataset
    else:
        return train_loader, train_loader_all, val_loader, cls_num_list