import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tensorboardX
import torch.optim as optim
import shutil
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import sys
path_root = '/path_root'
sys.path.append(path_root)
adv_root = os.path.join(path_root, "adv_training")
from utils.importData import importData
from adv_training.models import *

class transform_raw_to_grid(object):
    def __call__(self, tensor):
        tensor *= 255./256
        tensor += 1./512.
        return tensor
class transform_grid_to_raw(object):
    def __call__(self, tensor):
        tensor *= 256./255.
        tensor -= 1./510.
        return tensor

mean_cifar=(0.4914, 0.4822, 0.4465)
std_cifar=(0.2023, 0.1994, 0.2010)

transform_cifar = transforms.Compose([
    transform_grid_to_raw(),
    transforms.Normalize(mean_cifar, std_cifar)])
inv_transform_cifar = transforms.Compose([
    transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1./0.2023, 1./0.1994, 1./0.2010)),
    transform_raw_to_grid()])

class standard():
    def __init__(self, args, config):
        # configuration part
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer=="Adam":
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)

    def train(self): # Train by adversarial training
        # Data config
        dataLoader = importData(dataset=self.config.data.dataset, train=True, shuffle=True, bsize=self.config.training.batch_size)
        testLoader = importData(dataset=self.config.data.dataset, train=False, shuffle=True, bsize=self.config.training.batch_size)
        testIter = iter(testLoader)

        # tensorboard setting
        tb_path = os.path.join(adv_root, self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        # Network config
        network = eval(self.args.network)().to('cuda')
        network = torch.nn.DataParallel(network)
        optimizer = self.get_optimizer(network.parameters())
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            network.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
        criterion = nn.CrossEntropyLoss()

        # Train
        for epoch in range(self.config.training.n_epochs):
            train_loss_original = 0.0
            train_acc_original = 0.0
            train_loss_adv = 0.0
            train_acc_adv = 0.0
            test_loss_original = 0.0
            test_acc_original = 0.0
            test_loss_adv = 0.0
            test_acc_adv = 0.0
            for i, (x,y) in enumerate(dataLoader):
                x = x.to('cuda')
                y = y.to('cuda')
                x = x/256.*255. + torch.ones_like(x)/512.

                # Update parameters
                x.requires_grad_(True)
                optimizer.zero_grad()
                yhat = network(x)
                loss = criterion(yhat, y)
                loss.backward()
                optimizer.step()

            # Validation
            loss_o, total_o, correct_o = 0, 0, 0
            for i, (x,y) in enumerate(dataLoader):
                # Original data
                x = x.to('cuda')
                y = y.to('cuda')
                x = x/256.*255. + torch.ones_like(x)/512.
                
                yhat = network(x)
                loss_o += criterion(yhat, y).item()
                _, predicted = yhat.max(1)
                total_o += y.size(0)
                correct_o += predicted.eq(y).sum().item()

            train_loss_original = loss_o/(i+1)
            train_acc_original = correct_o/total_o

            loss_o, total_o, correct_o = 0, 0, 0
            for i, (x,y) in enumerate(testLoader):
                # Original data
                x = x.to('cuda')
                y = y.to('cuda')
                x = x/256.*255. + torch.ones_like(x)/512.
                yhat = network(x)
                loss_o += criterion(yhat, y).item()
                _, predicted = yhat.max(1)
                total_o += y.size(0)
                correct_o += predicted.eq(y).sum().item()

            test_loss_original = loss_o/(i+1)
            test_acc_original = correct_o/total_o

            # Update tensorboard loggers
            tb_logger.add_scalar('train_loss_original', train_loss_original, global_step=epoch)
            tb_logger.add_scalar('train_acc_original', train_acc_original, global_step=epoch)
            tb_logger.add_scalar('test_loss_original', test_loss_original, global_step=epoch)
            tb_logger.add_scalar('test_acc_original', test_acc_original, global_step=epoch)

            if epoch%self.config.training.snapshot_freq == 0:
                states = [
                    network.state_dict(),
                    optimizer.state_dict(),
                ]
                torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(epoch)))
                torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

            # Print for screen
            print('Epoch {}\t(Ori) {:.3f}\t|{:.3f}\t|{:.3f}\t|{:.3f}\t|'.format(epoch, train_loss_original, train_acc_original, test_loss_original, test_acc_original))

    def feature(self): # Get features and save
        network = eval(self.args.network)().to('cuda')
        network = torch.nn.DataParallel(network)
        network.load_state_dict(states[0])
        testLoader = importDSata(dataset=self.config.data.dataset, train=True, shuffle=False, bsize=self.config.training.batch_size)
        latents = None
        labels = None
        for i, (x,y) in enumerate(testLoader):
            x = x.to('cuda')
            y = y.to('cuda')
            x = x/256.*255. + torch.ones_like(x)/512.
            x = transform_cifar(x)
            z = network.latent(x).cpu().numpy()
            if latents is None:
                latents = z
            else:
                latents = np.concatenate((latents, z), axis=0)
                print(latents.shape)
            if labels is None:
                labels = y
            else:
                labels = np.concatenate((labels, y), axis=0)
                print(labels.shape)
        np.save('./features_cifar.npy', latents)
        np.save('./labels_cifar.npy', labels)

    def test(self): # Test for adversarially attacked images
        network = eval(self.args.network)().to('cuda')
        network = torch.nn.DataParallel(network)
        optimizer = self.get_optimizer(network.parameters())
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
        network.load_state_dict(states[0])
        optimizer.load_state_dict(states[1])

        testLoader = importData(dataset=self.config.data.dataset, train=False, shuffle=True, bsize=self.config.training.batch_size)
        criterion = nn.CrossEntropyLoss()

        # Get loss and accuracy
        loss, total, correct = 0, 0, 0
        for i, (x,y) in enumerate(testLoader):
            # Original data
            x = x.to('cuda')
            y = y.to('cuda')
            x = x/256.*255. + torch.ones_like(x)/512.

            # Get loss and accuracy
            yhat = network(x)
            loss += criterion(yhat, y).item()
            _, predicted = yhat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            # Get original and adversarial examples and deploy images
            if i==1:
                grid_size = 10
                ax_original = plt.subplot(1, 3, 1)
                ax_original.axis("off")
                view_x = x.view(grid_size**2, x.size(1), x.size(2), x.size(3))
                if x.size(1)==1:
                    grid1 = make_grid(view_x, nrow=grid_size)[0,:,:]
                    ax_original.imshow(grid1.detach().cpu().numpy(), cmap='gray')
                else:
                    grid1 = make_grid(view_x, nrow=grid_size)
                    ax_original.imshow(np.transpose(grid1.detach().cpu().numpy(), (1,2,0)), cmap='gray')
                ax_adv = plt.subplot(1, 3, 2) # attacked image map
                ax_adv.axis("off")
                x_pgd = x
                view_xadv = x_pgd.view(grid_size**2, x_pgd.size(1), x_pgd.size(2), x_pgd.size(3))
                if x_pgd.size(1)==1:
                    grid2 = make_grid(view_xadv, nrow=grid_size)[0,:,:]
                    ax_adv.imshow(grid2.detach().cpu().numpy(), cmap='gray')
                else:
                    grid2 = make_grid(view_xadv, nrow=grid_size)
                    ax_adv.imshow(np.transpose(grid2.detach().cpu().numpy(), (1,2,0)), cmap='gray')
                diff = plt.subplot(1, 3, 3) # difference map
                diff.axis("off")
                x_diff = (x_pgd - x)*0.5
                view_diff = x_diff.view(grid_size**2, x_diff.size(1), x_diff.size(2), x_diff.size(3))
                if x_diff.size(1)==1:
                    grid3 = make_grid(view_diff, nrow=grid_size)[0,:,:]
                    diff.imshow(grid3.detach().cpu().numpy(), cmap=plt.get_cmap('seismic'))
                else:
                    grid3 = make_grid(view_diff, nrow=grid_size)
                    diff.imshow(np.transpose(grid3.detach().cpu().numpy(), (1,2,0)), cmap=plt.get_cmap('seismic'))
                plt.savefig('images.png')
                plt.close()

            criterion = nn.CrossEntropyLoss()
        print('Loss: {}, Accuracy: {}'.format(loss/(i+1), correct/total))
