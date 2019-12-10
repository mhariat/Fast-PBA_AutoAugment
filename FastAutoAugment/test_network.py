from FastAutoAugment.networks.pyramidnet import *
from FastAutoAugment.data import *
import torch.nn as nn

import gc


def validation(model, use_cuda, val_dataloader):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            validation_loss += criterion(output, target).data.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    torch.cuda.empty_cache()
    gc.collect()
    validation_loss /= len(val_dataloader.dataset)
    validation_accuracy = correct/len(val_dataloader.dataset)
    return validation_accuracy, validation_loss


save_path = '/app/results/checkpoints/backup_2/pyramid_best.pth'

if __name__ == '__main__':
    model = PyramidNet('cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
    print("Model loaded...")
    results = torch.load(save_path, map_location='cpu')
    model_dict = results['model']
    model.load_state_dict(model_dict)
    print("Weights loaded...")
    if torch.cuda.is_available():
        model.cuda()
    _, _, _, testloader = get_dataloaders(dataset='cifar100', batch=64, dataroot='/usr/share/bind_mount/data/cifar_100')
    print("Data loaded...")
    test_acc, test_loss = validation(model, torch.cuda.is_available(), testloader)
    print("Test accuracy", test_acc)


