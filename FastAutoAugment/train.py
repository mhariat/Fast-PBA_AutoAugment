import itertools
import logging
import math
import os
from collections import OrderedDict
import numpy as np
import sys
import torch
from torch import nn, optim

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser
from torch.autograd import Variable


from FastAutoAugment.common import get_logger
from FastAutoAugment.data import get_dataloaders
from FastAutoAugment.lr_scheduler import adjust_learning_rate_pyramid, adjust_learning_rate_resnet
from FastAutoAugment.metrics import accuracy, Accumulator
from FastAutoAugment.networks import get_model, num_class

from warmup_scheduler import GradualWarmupScheduler
from FastAutoAugment.data import AugmentationPba

logger = get_logger('Fast AutoAugment')
logger.setLevel(logging.INFO)


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count


def run_epoch(model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None):
    model_name = C.get()['model']['type']
    alpha = C.get()['alpha']
    skip_ratios = ListAverageMeter()
    tqdm_disable = bool(os.environ.get('TASK_NAME', ''))
    if verbose:
        loader = tqdm(loader, disable=tqdm_disable)
        loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))

    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    for data, label in loader:
        steps += 1
        data, label = data.cuda(), label.cuda()

        if optimizer:
            optimizer.zero_grad()

        if model_name == 'pyramid_skip':
            if desc_default == '*test':
                with torch.no_grad():
                    preds, masks, gprobs = model(data)
                skips = [mask.data.le(0.5).float().mean() for mask in masks]
                if skip_ratios.len != len(skips):
                    skip_ratios.set_len(len(skips))
                skip_ratios.update(skips, data.size(0))
            else:
                preds, masks, gprobs = model(data)

            sparsity_loss = 0
            for mask in masks:
                sparsity_loss += mask.mean()
            loss1 = loss_fn(preds, label)
            loss2 = alpha * sparsity_loss
            loss = loss1 + loss2
        else:
            preds = model(data)
            loss = loss_fn(preds, label)

        if optimizer:
            loss.backward()
            if getattr(optimizer, "synchronize", None):
                optimizer.skip_synchronize()
            if C.get()['optimizer'].get('clip', 5) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), C.get()['optimizer'].get('clip', 5))

            optimizer.step()

        top1, top5 = accuracy(preds, label, (1, 5))

        if model_name == 'pyramid_skip':
            metrics.add_dict({
                'loss1': loss1.item() * len(data),
                'loss2': loss2.item() * len(data),
                'top1': top1.item() * len(data),
                'top5': top5.item() * len(data),
            })
        else:
            metrics.add_dict({
                'loss': loss.item() * len(data),
                'top1': top1.item() * len(data),
                'top5': top5.item() * len(data),
            })
        cnt += len(data)
        if verbose:
            postfix = metrics / cnt
            if optimizer:
                postfix['lr'] = optimizer.param_groups[0]['lr']
            loader.set_postfix(postfix)

        # if scheduler is not None:
        #     scheduler.step(epoch - 1 + float(steps) / total_steps)

        if model_name == 'pyramid_skip':
            del masks[:], gprobs[:]
        del preds, loss, top1, top5, data, label

    if model_name == 'pyramid_skip':
        if desc_default == '*test':
            skip_summaries = []
            for idx in range(skip_ratios.len):
                skip_summaries.append(1 - skip_ratios.avg[idx])
            cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100

    if tqdm_disable:
        logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)

    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
        if model_name == 'pyramid_skip':
            if desc_default == '*test':
                writer.add_scalar('Computation Percentage', cp, epoch)
    return metrics


def train_and_eval(tag, dataroot, test_ratio=0.0, cv_fold=0, reporter=None, metric='last', save_path=None, only_eval=False, horovod=False):
    if horovod:
        import horovod.torch as hvd
        hvd.init()
        device = torch.device('cuda', hvd.local_rank())
        torch.cuda.set_device(device)

    if not reporter:
        reporter = lambda **kwargs: 0

    max_epoch = C.get()['epoch']
    # trainsampler, trainloader, validloader, testloader_ = get_dataloaders(dataroot, C.get()['batch'], horovod=horovod)
    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(C.get()['dataset'], C.get()['batch'], dataroot, test_ratio, split_idx=cv_fold, horovod=horovod)

    # create a model & an optimizer
    model = get_model(C.get()['model'], num_class(C.get()['dataset']), data_parallel=(not horovod))

    criterion = nn.CrossEntropyLoss()
    if C.get()['optimizer']['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=C.get()['lr'],
            momentum=C.get()['optimizer'].get('momentum', 0.9),
            weight_decay=C.get()['optimizer']['decay'],
            nesterov=C.get()['optimizer']['nesterov']
        )
    else:
        raise ValueError('invalid optimizer type=%s' % C.get()['optimizer']['type'])

    is_master = True
    if horovod:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        optimizer._requires_update = set()  # issue : https://github.com/horovod/horovod/issues/1099
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        if hvd.rank() != 0:
            is_master = False
    logger.debug('is_master=%s' % is_master)

    lr_scheduler_type = C.get()['lr_schedule'].get('type', 'cosine')
    if lr_scheduler_type == 'cosine':
        t_max = C.get()['epoch']
        if C.get()['lr_schedule'].get('warmup', None):
            t_max -= C.get()['lr_schedule']['warmup']['epoch']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.)
    elif lr_scheduler_type == 'resnet':
        scheduler = adjust_learning_rate_resnet(optimizer)
    elif lr_scheduler_type == 'pyramid':
        scheduler = adjust_learning_rate_pyramid(optimizer, C.get()['epoch'])
    else:
        raise ValueError('invalid lr_schduler=%s' % lr_scheduler_type)

    if C.get()['lr_schedule'].get('warmup', None):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
            total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )
    if not tag.strip() or not is_master:
        from FastAutoAugment.metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided, no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter
    writers = [SummaryWriter(log_dir='/app/results/logs/%s/%s' % (tag, x)) for x in ['train', 'valid', 'test']]

    result = OrderedDict()
    epoch_start = 1
    # if save_path and os.path.exists(save_path):
    #     data = torch.load(save_path)
    #     if 'model' in data:
    #         # TODO : patch, horovod trained checkpoint
    #         new_state_dict = {}
    #         for k, v in data['model'].items():
    #             if not horovod and 'module.' not in k:
    #                 new_state_dict['module.' + k] = v
    #             else:
    #                 new_state_dict[k] = v
    #
    #         model.load_state_dict(new_state_dict)
    #         optimizer.load_state_dict(data['optimizer'])
    #         logger.info('ckpt epoch@%d' % data['epoch'])
    #         if data['epoch'] < C.get()['epoch']:
    #             epoch_start = data['epoch']
    #         else:
    #             only_eval = True
    #         logger.info('epoch=%d' % data['epoch'])
    #     else:
    #         model.load_state_dict(data)
    #     del data

    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, None, desc_default='train', epoch=0, writer=writers[0])
        rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid', epoch=0, writer=writers[1])
        rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', epoch=0, writer=writers[2])
        for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result
    # train loop
    best_valid_loss = 10e10

    for epoch in range(epoch_start, max_epoch + 1):
        if horovod:
            trainsampler.set_epoch(epoch)
        model.train()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, optimizer, desc_default='train', epoch=epoch, writer=writers[0], verbose=is_master, scheduler=scheduler)
        AugmentationPba.epoch += 1
        scheduler.step(epoch)
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

        if epoch % (10 if 'cifar' in C.get()['dataset'] else 30) == 0 or epoch == max_epoch:
            rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid', epoch=epoch, writer=writers[1], verbose=is_master)
            rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=is_master)

            if metric == 'last' or rs[metric]['loss'] < best_valid_loss:    # TODO
                if metric != 'last':
                    best_valid_loss = rs[metric]['loss']
                for key, setname in itertools.product(['loss', 'top1', 'top5'], ['train', 'valid', 'test']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                result['epoch'] = epoch

                writers[1].add_scalar('valid_top1/best', rs['valid']['top1'], epoch)
                writers[2].add_scalar('test_top1/best', rs['test']['top1'], epoch)

                reporter(
                    loss_valid=rs['valid']['loss'], top1_valid=rs['valid']['top1'],
                    loss_test=rs['test']['loss'], top1_test=rs['test']['top1']
                )

            # save checkpoint
            if is_master and save_path:
                model_name = C.get()['model']['type']
                if 'skip' in model_name:
                    alpha = int(np.log10(1/C.get()['alpha']))
                    filename = '{}/{}_last_epoch_alpha_{}.pth'.format(save_path, model_name, alpha)
                else:
                    filename = '{}/{}_last_epoch.pth'.format(save_path, model_name)
                logger.info('save model@%d to %s' % (epoch, filename))
                torch.save({
                    'epoch': epoch,
                    'log': {
                        'train': rs['train'].get_dict(),
                        'valid': rs['valid'].get_dict(),
                        'test': rs['test'].get_dict(),
                    },
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict()
                }, filename)

    del model

    return result


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='/usr/share/bind_mount/data/cifar_100')
    parser.add_argument('--save', type=str, default='/app/results/checkpoints')
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--decay', type=float, default=-1)
    parser.add_argument('--horovod', action='store_true')
    parser.add_argument('--only-eval', action='store_true')
    args = parser.parse_args()
    print(args)

    assert not (args.horovod and args.only_eval), 'can not use horovod when evaluation mode is enabled.'
    assert (args.only_eval and not args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    if args.decay > 0:
        logger.info('decay reset=%.8f' % args.decay)
        C.get()['optimizer']['decay'] = args.decay
    if args.save:
        logger.info('checkpoint will be saved at %s', args.save)
        if not os.path.exists(args.save):
            os.makedirs(args.save)

    import time
    t = time.time()
    result = train_and_eval(args.tag, args.dataroot, test_ratio=args.cv_ratio, cv_fold=args.cv, save_path=args.save, horovod=args.horovod)
    elapsed = time.time() - t

    logger.info('training done.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('augmentation: %s' % C.get()['aug'])
    logger.info(result)
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('top1 error in testset: %.4f' % (1. - result['top1_test']))
