import torch
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from ncsn.utils import TinyImageNet

import os
import sys
path_root = '/path_root'
sys.path.append(path_root)

def importData(dataset, train, shuffle, bsize):
    '''
    dataset: datasets (MNIST, CIFAR10, CIFAR100, SVHN, CELEBA)
    train: True if training set, False if test set
    shuffle: Whether to shuffle or not
    bsize: minibatch size
    '''
    # Set transform
    dataset_list = ["MNIST", "CIFAR10", "FashionMNIST", "CIFAR10C", "CIFAR100", "TinyImageNet"]
    if dataset not in dataset_list:
        sys.exit("Non-handled dataset")

    if dataset=="MNIST":
        path = os.path.join(path_root, "datasets", "MNIST")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = MNIST(path, train=train, download=True, transform=transform)
    elif dataset=="CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        path = os.path.join(path_root, "datasets", "CIFAR10")
        dataset = CIFAR10(path, train=train, download=True, transform=transform)
    elif dataset=="FashionMNIST":
        path = os.path.join(path_root, "datasets", "FashionMNIST")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = FashionMNIST(path, train=train, download=True, transform=transform)
    elif dataset=="CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        path = os.path.join(path_root, "datasets", "CIFAR100")
        dataset = CIFAR100(path, train=train, download=True, transform=transform)
    elif dataset=="TinyImageNet":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        path = os.path.join(path_root,"ncsn")
        if train:
            dataset = TinyImageNet.TinyImageNet(os.path.join(path, 'datasets', 'tiny-imagenet-200', 'train'), train=train)
        else:
            dataset = TinyImageNet.TinyImageNet(os.path.join(path, 'datasets', 'tiny-imagenet-200', 'val'), train=train)
    elif dataset=="CIFAR10C":
        dataloader = []
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        path = os.path.join(path_root, "datasets", "CIFAR10-C")
        file_list = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', \
            'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        label_path = os.path.join(path, "labels.npy")
        lb_file = np.load(label_path) # Size [50000]
        np_y = lb_file[0:10000]
        for i in range(len(file_list)):
            sub_dataloader = []
            np_x = np.load(os.path.join(path, file_list[i]+".npy"))
            np_x = np.transpose(np_x, (0,3,1,2))
            for j in range(5):
                tensor_x = torch.Tensor(np_x[j*10000:(j+1)*10000])
                tensor_y = torch.Tensor(np_y)
                dset = TensorDataset(tensor_x, tensor_y)
                sub_dataloader.append(DataLoader(dset, batch_size=bsize, shuffle=shuffle, num_workers=4))
            dataloader.append(sub_dataloader)
        
    '''
    elif dataset=="CelebA":
        transform = transforms.Compose([
            transforms.CenterCrop(140),
            transforms.Resize(32),
            transforms.ToTensor()
        ])
        path = "./datasets/CelebA"
        dataset = CelebA(path, train=train, download=True, transform=transform)
    '''
    if dataset != "CIFAR10C":
        dataloader = DataLoader(dataset, batch_size=bsize, shuffle=shuffle, num_workers=4)
        return dataloader
    else:
        return dataloader, file_list
