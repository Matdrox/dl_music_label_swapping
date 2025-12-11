import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import pprint
import math

import torch
import torch.optim
import torch.nn.functional as F

from datasets.cifar10 import CIFAR10_LT
from datasets.cifar100 import CIFAR100_LT
from datasets.places import Places_LT
from datasets.imagenet import ImageNet_LT
from datasets.ina2018 import iNa2018

from models import load_model

from utils import config, update_config, create_logger
from utils import AverageMeter, ProgressMeter
from utils import accuracy, calibration

from methods import mixup_data, mixup_criterion
from methods import LabelAwareSmoothing


def parse_args():
    parser = argparse.ArgumentParser(description='MiSLAS training (Stage-2)')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args


best_acc1 = 0
its_ece = 100


def main():

    args = parse_args()
    logger, model_dir = create_logger(config, args.cfg)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + str(config))

    if config.deterministic:
        seed = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')


    ngpus_per_node = torch.cuda.device_count()
    
    # Simply call main_worker function
    main_worker(config.gpu, ngpus_per_node, config, logger, model_dir)


def main_worker(gpu, ngpus_per_node, config, logger, model_dir):
    global best_acc1, its_ece
    config.gpu = gpu
#     start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    model, classifier, lws_model = load_model.load_model(gpu, config, logger, load_lws=False)

    # Data loading code
    if config.dataset == 'cifar10':
        dataset = CIFAR10_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                             batch_size=config.batch_size, num_works=config.workers, config=config)

    elif config.dataset == 'cifar100':
        dataset = CIFAR100_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                              batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'places':
        dataset = Places_LT(config.distributed, root=config.data_path,
                            batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'imagenet':
        dataset = ImageNet_LT(config.distributed, root=config.data_path,
                              batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'ina2018':
        dataset = iNa2018(config.distributed, root=config.data_path,
                          batch_size=config.batch_size, num_works=config.workers)

    train_loader = dataset.train_balance
    val_loader = dataset.eval
    cls_num_list = dataset.cls_num_list
    if config.distributed:
        train_sampler = dataset.dist_sampler

    # define loss function (criterion) and optimizer

    criterion = LabelAwareSmoothing(cls_num_list=cls_num_list, smooth_head=config.smooth_head,
                                    smooth_tail=config.smooth_tail).cuda(config.gpu)

    optimizer = torch.optim.SGD([{"params": classifier.parameters()},
                                {'params': lws_model.parameters()}], config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    for epoch in range(config.num_epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, config)

        if config.dataset != 'places':
            block = None
        # train for one epoch
        train(train_loader, model, classifier, lws_model, criterion, optimizer, epoch, config, logger, block)

        # evaluate on validation set
        acc1, ece = validate(val_loader, model, classifier, lws_model, criterion, config, logger, block)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            its_ece = ece
        logger.info('Best Prec@1: %.3f%% ECE: %.3f%%\n' % (best_acc1, its_ece))
        if not config.multiprocessing_distributed or (config.multiprocessing_distributed
                                                      and config.rank % ngpus_per_node == 0):
            if config.dataset == 'places':
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                    'state_dict_classifier': classifier.state_dict(),
                    'state_dict_block': block.state_dict(),
                    'state_dict_lws_model': lws_model.state_dict(),
                    'best_acc1': best_acc1,
                    'its_ece': its_ece,
                }, is_best, model_dir)
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_model': model.state_dict(),
                    'state_dict_classifier': classifier.state_dict(),
                    'state_dict_lws_model': lws_model.state_dict(),
                    'best_acc1': best_acc1,
                    'its_ece': its_ece,
                }, is_best, model_dir)


def train(train_loader, model, classifier, lws_model, criterion, optimizer, epoch, config, logger, block=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    training_data_num = len(train_loader.dataset)
    end_steps = int(np.ceil(float(training_data_num) / float(train_loader.batch_size)))
    progress = ProgressMeter(
        end_steps,
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode

    if config.dataset == 'places':
        model.eval()
        if config.shift_bn:
            block.train()
        else:
            block.eval()
    else:
        if config.shift_bn:
            model.train()
        else:
            model.eval()
    classifier.train()

    end = time.time()

    for i, (index, images, target) in enumerate(train_loader):
        if i > end_steps:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(config.gpu, non_blocking=True)
            target = target.cuda(config.gpu, non_blocking=True)

        if config.mixup is True:
            images, targets_a, targets_b, lam = mixup_data(images, target, alpha=config.alpha)
            with torch.no_grad():
                if config.dataset == 'places':
                    feat = block(model(images))
                else:
                    feat = model(images)
            output = classifier(feat.detach())
            output = lws_model(output)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        else:
            # compute output
            with torch.no_grad():
                if config.dataset == 'places':
                    feat = block(model(images))
                else:
                    feat = model(images)
            output = classifier(feat.detach())
            output = lws_model(output)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i, logger)


def validate(val_loader, model, classifier, lws_model, criterion, config, logger, block=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Eval: ')

    # switch to evaluate mode
    model.eval()
    if config.dataset == 'places':
        block.eval()
    classifier.eval()
    class_num = torch.zeros(config.num_classes).cuda()
    correct = torch.zeros(config.num_classes).cuda()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if config.gpu is not None:
                images = images.cuda(config.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(config.gpu, non_blocking=True)

            # compute output
            if config.dataset == 'places':
                feat = block(model(images))
            else:
                feat = model(images)
            output = classifier(feat)
            output = lws_model(output)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1)
            target_one_hot = F.one_hot(target, config.num_classes)
            predict_one_hot = F.one_hot(predicted, config.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i, logger)

        acc_classes = correct / class_num
        head_acc = acc_classes[config.head_class_idx[0]:config.head_class_idx[1]].mean() * 100
        med_acc = acc_classes[config.med_class_idx[0]:config.med_class_idx[1]].mean() * 100
        tail_acc = acc_classes[config.tail_class_idx[0]:config.tail_class_idx[1]].mean() * 100

        logger.info('* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))

        cal = calibration(true_class, pred_class, confidence, num_bins=15)
        logger.info('* ECE   {ece:.3f}%.'.format(ece=cal['expected_calibration_error'] * 100))

    return top1.avg, cal['expected_calibration_error'] * 100


def save_checkpoint(state, is_best, model_dir):
    filename = model_dir + '/current.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_dir + '/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate"""
    lr_min = 0
    lr_max = config.lr
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / config.num_epochs * 3.1415926535))

    for idx, param_group in enumerate(optimizer.param_groups):
        if idx == 0:
            param_group['lr'] = config.lr_factor * lr
        else:
            param_group['lr'] = 1.00 * lr


if __name__ == '__main__':
    main()