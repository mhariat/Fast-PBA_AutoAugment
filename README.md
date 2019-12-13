# Comparison of Fast AutoAugment and Population Based Augmentation

The code is an adaptation of the Official [Fast AutoAugment](https://arxiv.org/abs/1905.00397) implementation in
PyTorch available [here](https://github.com/kakaobrain/fast-autoaugment/).

Fast AutoAugment is a simplified version of the well known [AutoAugment](https://arxiv.org/abs/1805.09501) paper of
Google Brain.

- Fast AutoAugment learns augmentation policies using a more efficient search strategy based on density matching.
- Fast AutoAugment speeds up the search time by orders of magnitude while maintaining the comparable performances.

The code is also modified so that it can use the official implementation of the paper
[Population Based Augmentation](https://arxiv.org/abs/1905.05393). This method learns a data-augmentation schedule
policy that needs to be applied during the training process. The underlying intuition being that at the
beginning, there is no need to apply a very elaborated data-augmentation. However, the more the neural network learns, the
better the data-augmentation should be. Data-augmentation schedule policies can be found in the folder *schedules*.

## Table Results


| Model(CIFAR-100)      | Baseline   | Cutout     | AutoAugment | Fast AutoAugment<br/>(transfer/direct) |PBA|
|-----------------------|------------|------------|-------------|------------------|------------------|
| PyramidNet+ShakeDrop  | 14.0       | 12.2       | 10.7        | 11.7      | 11.4     |


## Run

### Prerequisite:
Download CIFAR-100 dataset.


### How to run:
```
cd /usr/share/bind_mount/scripts/MicroNet/Training
pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:/usr/share/bind_mount/scripts/MicroNet/Training"
cd FastAutoAugment
horovodrun -np 4 --cache-capacity 2048 python train.py -c confs/pyramid272a200b.yaml --tag pyramidnet_cifar100 --horovod
```

The argument *np* allows you to choose the number of GPUs you want to use (here 4).

Please see the *confs/pyramid272a200b.yaml* file to get more insights on the parameters used for the training process.
You may notice an argument named *aug*. It allows you to choose which data augmentation policy you want to use.  

The checkpoint weights are regularly saved (every 10 epochs) and put in the directory */app/results/checkpoints*.

