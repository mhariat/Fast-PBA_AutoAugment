import torch

from theconf import Config as C


def adjust_learning_rate_pyramid(optimizer, max_epoch):

    # def __adjust_learning_rate_pyramid(epoch):
    #     lr = (0.1 ** (epoch // (max_epoch * 0.25))) * (0.2 ** (epoch // (max_epoch * 0.5))) * (0.2 ** (epoch // (max_epoch * 0.625))) * (0.2 ** (epoch // (max_epoch * 0.75)))
    #     return lr

    # def __adjust_learning_rate_pyramid(epoch):
    #     if epoch < int(max_epoch*0.25):
    #         lr = 1
    #     elif (int(max_epoch*0.25) <= epoch) & (epoch < int(max_epoch*0.5)):
    #         lr = 0.1
    #     elif (int(max_epoch*0.5) <= epoch) & (epoch < int(max_epoch*0.625)):
    #         lr = 0.02
    #     elif (int(max_epoch*0.625) <= epoch) & (epoch < int(max_epoch*0.75)):
    #         lr = 0.004
    #     else:
    #         lr = 0.0008
    #     return lr

    def __adjust_learning_rate_pyramid(epoch):
        if epoch < int(max_epoch*0.25):
            lr = 1
        elif (int(max_epoch*0.25) <= epoch) & (epoch < int(max_epoch*0.5)):
            lr = 0.1
        elif (int(max_epoch*0.5) <= epoch) & (epoch < int(max_epoch*0.75)):
            lr = 0.004
        else:
            lr = 0.0004
        return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, __adjust_learning_rate_pyramid)


def adjust_learning_rate_resnet(optimizer):
    """
    Sets the learning rate to the initial LR decayed by 10 every [90, 180, 240] epochs
    Ref: AutoAugment
    """

    if C.get()['epoch'] == 90:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 80])
    elif C.get()['epoch'] == 270:   # autoaugment
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, [90, 180, 240])
    else:
        raise ValueError('invalid epoch=%d for resnet scheduler' % C.get()['epoch'])
