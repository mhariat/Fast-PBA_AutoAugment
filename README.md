# Step 1: Training of PyramidNet

For the Micronet Challenge we chose to start from PyramidNet and then to compress it. Multiple reasons led to choose
such a network. Amongst them:

- High accuracy (see table below)
- Reasonable number of parameters (26.2M) compared to the benchmark Wide-Resnet-28-10 (36.5M)

The code is an adaptation of the Official [Fast AutoAugment](https://arxiv.org/abs/1905.00397) implementation in
PyTorch.

Fast AutoAugment is a simplified version of the well known [AutoAugment](https://arxiv.org/abs/1805.09501) paper of
Google Brain.

- Fast AutoAugment learns augmentation policies using a more efficient search strategy based on density matching.
- Fast AutoAugment speeds up the search time by orders of magnitude while maintaining the comparable performances.

One of the author, *Ildoo Kim*, kindly gave us the **policies obtained with Shake-Shake(26_2x96d) on CIFAR-100**.


## Table Results of Fast-Autoaugment

### CIFAR-10 / 100

Search : **3.5 GPU Hours (1428x faster than AutoAugment)**, WResNet-40x2 on Reduced CIFAR-10

| Model(CIFAR-10)         | Baseline   | Cutout     | AutoAugment | Fast AutoAugment<br/>(transfer/direct) |
|-------------------------|------------|------------|-------------|------------------|
| Wide-ResNet-40-2        | 5.3        | 4.1        | 3.7         | 3.6 / 3.7        |
| Wide-ResNet-28-10       | 3.9        | 3.1        | 2.6         | 2.7 / 2.7        |
| Shake-Shake(26 2x32d)   | 3.6        | 3.0        | 2.5         | 2.7 / 2.5        |
| Shake-Shake(26 2x96d)   | 2.9        | 2.6        | 2.0         | 2.0 / 2.0        |
| Shake-Shake(26 2x112d)  | 2.8        | 2.6        | 1.9         | 2.0 / 1.9        |
| PyramidNet+ShakeDrop    | 2.7        | 2.3        | 1.5         | 1.8 / 1.7        |

| Model(CIFAR-100)      | Baseline   | Cutout     | AutoAugment | Fast AutoAugment<br/>(transfer/direct) |
|-----------------------|------------|------------|-------------|------------------|
| Wide-ResNet-40-2      | 26.0       | 25.2       | 20.7        | 20.6 / 20.6      |
| Wide-ResNet-28-10     | 18.8       | 28.4       | 17.1        | 17.8 / 17.5      |
| Shake-Shake(26 2x96d) | 17.1       | 16.0       | 14.3        | 14.9 / 14.6      |
| PyramidNet+ShakeDrop  | 14.0       | 12.2       | 10.7        | 11.9 / 11.7      |

### ImageNet

Search : **450 GPU Hours (33x faster than AutoAugment)**, ResNet-50 on Reduced ImageNet

| Model      | Baseline   | AutoAugment | Fast AutoAugment |
|------------|------------|-------------|------------------|
| ResNet-50  | 23.7 / 6.9 | 22.4 / 6.2  | **22.4 / 6.3**   |
| ResNet-200 | 21.5 / 5.8 | 20.0 / 5.0  | **19.4 / 4.7**   |



## Run

### Prerequisite:
1. Download CIFAR-100 dataset.
1. Build and Run the Docker Image *fast_autoaugment*. **See instructions** in *MicroNet/Docker/myimages*.


### How to run:
```
cd /usr/share/bind_mount/scripts/MicroNet/Training
pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:/usr/share/bind_mount/scripts/MicroNet/Training"
cd FastAutoAugment
horovodrun -np 4 --cache-capacity 2048 python train.py -c confs/pyramid272a200b.yaml --tag pyramidnet_cifar10 --horovod
```

The argument *np* allows you to choose the number of GPUs you want to use (here 4).

Please see the *confs/pyramid272a200b.yaml* file to get more insights on the parameters used for the training process.
You may notice an argument named *aug*. It allows you to choose which data augmentation policy you want to use.

As said earlier, the policy use here was given by the authors and **obtained using only CIFAR-100**. You may want to
change the batch size according to your computational resources.

The checkpoint weights are regularly saved (every 10 epochs) and put in the directory */app/results/checkpoints*.

