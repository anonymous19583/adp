import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from PIL import Image

import os
import sys

path_root = '/path_root'
sys.path.append(path_root)

def TinyImageNet(path, train):
    wnids = open(os.path.join(path_root, 'ncsn', 'datasets', 'tiny-imagenet-200', 'wnids.txt'), 'r')
    label_list = [line[:-1] for line in wnids]
    if train:
        nData = 100000
    else:
        nData = 10000
    img_np = np.zeros((nData, 3, 64, 64))
    label_np = np.zeros((nData))
    if train: # Training set
        train_fn = os.path.join(path_root, 'ncsn', 'datasets', 'tiny-imagenet-200', 'tinyImageNet_train.npy')
        train_labels = os.path.join(path_root, 'ncsn', 'datasets', 'tiny-imagenet-200', 'tinyImageNet_train_labels.npy')
        if not os.path.isfile(train_fn):
            print("Begin generation")
            for i in range(200): # There are 200 classes
                label = label_list[i]
                training_label_dir = os.path.join(path_root, 'ncsn', 'datasets', 'tiny-imagenet-200', 'train', label, 'images')
                for j in range(500): # Each class has 500 images
                    img_path = os.path.join(training_label_dir, label+'_'+str(j)+'.JPEG')
                    img_given = np.asarray(Image.open(img_path))
                    if len(img_given.shape)==2:
                        img_given = np.tile(img_given, (3, 1, 1))
                    else:
                        img = np.transpose(img_given, (2,0,1))
                    img_np[i*500+j] = img
                    label_np[i*500+j] = i
            np.save(train_fn, img_np/255.)
            np.save(train_labels, label_np)
            print("End generation")
        else:
            img_np = np.load(train_fn)
            label_np = np.load(train_labels)
                
    else: # Validation set
        val_fn = os.path.join(path_root, 'ncsn', 'datasets', 'tiny-imagenet-200', 'tinyImageNet_val.npy')
        val_labels = os.path.join(path_root, 'ncsn', 'datasets', 'tiny-imagenet-200', 'tinyImageNet_val_labels.npy')
        if not os.path.isfile(val_fn):
            validation_label_dir = os.path.join(path_root, 'ncsn', 'datasets', 'tiny-imagenet-200', 'val', 'images')
            val_annotation = open(os.path.join(path_root, 'ncsn', 'datasets', 'tiny-imagenet-200', 'val', 'val_annotations.txt'), 'r')
            for i in range(10000): # There are 10000 validation images
                label = val_annotation.readline().split()[1]
                img_path = os.path.join(validation_label_dir, 'val_'+str(i)+'.JPEG')
                img_given = np.asarray(Image.open(img_path))
                if len(img_given.shape)==2:
                    img_given = np.tile(img_given, (1, 1, 3))
                else:
                    img = np.transpose(img_given, (2,0,1))
                img_np[i] = img
                label_np[i] = label_list.index(label)
            np.save(val_fn, img_np/255.)
            np.save(val_labels, label_np)
        else:
            img_np = np.load(val_fn)
            label_np = np.load(val_labels)

    dset = TensorDataset(torch.Tensor(img_np), torch.Tensor(label_np))
    return dset
