import os
import torch
import torch.nn as nn

from models import resnet
from models import resnet_places
from models import resnet_cifar
from methods import LearnableWeightScaling
from utils.metrics_selmix import *

# Code for loading models
def load_model(gpu, config, logger, load_lws=True):
    global best_acc1, its_ece
    config.gpu = gpu
#     start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))

    if config.dataset == 'cifar10' or config.dataset == 'cifar100':
        model = getattr(resnet_cifar, config.backbone)()
        classifier = getattr(resnet_cifar, 'Classifier')(feat_in=64, num_classes=config.num_classes)

    elif config.dataset == 'imagenet' or config.dataset == 'ina2018':
        model = getattr(resnet, config.backbone)()
        classifier = getattr(resnet, 'Classifier')(feat_in=2048, num_classes=config.num_classes)

    elif config.dataset == 'places':
        model = getattr(resnet_places, config.backbone)(pretrained=True)
        classifier = getattr(resnet_places, 'Classifier')(feat_in=2048, num_classes=config.num_classes)
        block = getattr(resnet_places, 'Bottleneck')(2048, 512, groups=1, base_width=64,
                                                     dilation=1, norm_layer=nn.BatchNorm2d)

    lws_model = LearnableWeightScaling(num_classes=config.num_classes)

    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')

    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
        classifier = classifier.cuda(config.gpu)
        lws_model = lws_model.cuda(config.gpu)
        if config.dataset == 'places':
            block.cuda(config.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()
        lws_model = torch.nn.DataParallel(lws_model).cuda()
        if config.dataset == 'places':
            block = torch.nn.DataParallel(block).cuda()

    # optionaly resume from a checkpoint (although in stage2 we would expect to resume from a stage1 model)
    
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            if config.gpu is None:
                checkpoint = torch.load(config.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(config.gpu)
                checkpoint = torch.load(config.resume, map_location=loc, weights_only=config.weights_only_model)
            # config.start_epoch = checkpoint['epoch']
            print(checkpoint.keys())
            best_acc1 = checkpoint['best_acc1']
            its_ece = checkpoint['its_ece']
            if config.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(config.gpu)

            # Rename state dict due to torch version diffs
            if config.state_dict_from_old_version:
                renamed_state_dict = {}
                for key in checkpoint['state_dict_model'].keys():
                    new_key = key.replace("module.", "")
                    renamed_state_dict[new_key] = checkpoint['state_dict_model'][key]
                checkpoint['state_dict_model'] = renamed_state_dict

                renamed_state_dict_cls = {}
                for key in checkpoint['state_dict_classifier'].keys():
                    new_key = key.replace("module.", "")
                    renamed_state_dict_cls[new_key] = checkpoint['state_dict_classifier'][key]
                checkpoint['state_dict_classifier'] = renamed_state_dict_cls

                if load_lws:
                    renamed_state_dict_lws = {}
                    for key in checkpoint['state_dict_lws_model'].keys():
                        new_key = key.replace("module.", "")
                        renamed_state_dict_lws[new_key] = checkpoint['state_dict_lws_model'][key]
                    checkpoint['state_dict_lws_model'] = renamed_state_dict_lws
            model.load_state_dict(checkpoint['state_dict_model'])
            classifier.load_state_dict(checkpoint['state_dict_classifier'])
            if load_lws:
                lws_model.load_state_dict(checkpoint['state_dict_lws_model'])
            if config.dataset == 'places':
                block.load_state_dict(checkpoint['state_dict_block'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(config.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    return model, classifier, lws_model