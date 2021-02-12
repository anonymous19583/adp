import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tensorboardX
import torch.optim as optim
import shutil
from torchvision.utils import make_grid, save_image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import sys
path_root = '/path_root'
sys.path.append(path_root)
adv_root = os.path.join(path_root, "adv_training")
from utils.importData import importData
from adv_training.models import *


class pgd():
    def __init__(self, args, config):
        # configuration part
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer=="Adam":
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)

    def fgsm(self, x, xgrad, alpha):
        # By random start, start from xprime
        if self.args.norm==-1:
            return torch.clamp(x + alpha/256.*xgrad.sign(), min=0.0, max=1.0)
        elif self.args.norm==2: # Not implemented yet
            print("Not implemented for l2 norm yet")
            return x

    def pgd(self, x, xprime, y, network, optimizer, alpha=2., iters=40.):
        xprime = xprime.clone().detach()
        for i in range(iters):
            xprime = xprime.clone().detach()
            xprime.requires_grad_(True)
            yhat = network(xprime)

            optimizer.zero_grad()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(yhat, y)
            loss.backward()

            xprime = x + torch.clamp(self.fgsm(xprime, xprime.grad.data, alpha) - x, min=-self.args.ptb/256., max=self.args.ptb/256.)
        return xprime

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
                if self.config.training.random_start==True: # Random start in epsilon ball
                    delta = torch.rand_like(x)
                    if self.args.norm==-1: # L_inf norm
                        xprime = x + delta*self.args.ptb/256.
                    else: # Now only implemented in L_inf norm
                        xprime = x
                else:
                    xprime = x

                # Make adversarial examples
                x_pgd = self.pgd(x, xprime, y, network, optimizer)

                # Update parameters
                xtrain = x.clone().detach()
                mask = torch.rand(self.config.training.batch_size)
                for j in range(self.config.training.batch_size):
                    if mask[j]<self.config.training.alpha: # alpha: adversarial example weight
                        xtrain[j]=x_pgd[j]
                xtrain.requires_grad_(True)
                optimizer.zero_grad()
                yhat = network(xtrain)
                loss = criterion(yhat, y)
                loss.backward()
                optimizer.step()

            # Validation
            loss_o, total_o, correct_o, loss_a, total_a, correct_a = 0, 0, 0, 0, 0, 0
            for i, (x,y) in enumerate(dataLoader):
                # Original data
                x = x.to('cuda')
                y = y.to('cuda')
                x = x/256.*255. + torch.ones_like(x)/512.
                if self.config.training.random_start==True: # Random start in epsilon ball
                    delta = torch.rand_like(x)
                    if self.args.norm==-1: # L_inf norm
                        xprime = x + delta*self.args.ptb/256.
                    else: # Now only implemented in L_inf norm
                        xprime = x
                else:
                    xprime = x
                yhat = network(x)
                loss_o += criterion(yhat, y).item()
                _, predicted = yhat.max(1)
                total_o += y.size(0)
                correct_o += predicted.eq(y).sum().item()

                # Perturbed data
                x_pgd = self.pgd(x, xprime, y, network, optimizer)
                yhat_a = network(x_pgd)
                loss_a += criterion(yhat_a, y).item()
                _, predicted_a = yhat_a.max(1)
                total_a += y.size(0)
                correct_a += predicted_a.eq(y).sum().item()

            train_loss_original = loss_o/(i+1)
            train_acc_original = correct_o/total_o
            train_loss_adv = loss_a/(i+1)
            train_acc_adv = correct_a/total_a

            loss_o, total_o, correct_o, loss_a, total_a, correct_a = 0, 0, 0, 0, 0, 0
            for i, (x,y) in enumerate(testLoader):
                # Original data
                x = x.to('cuda')
                y = y.to('cuda')
                x = x/256.*255. + torch.ones_like(x)/512.
                if self.config.training.random_start==True: # Random start in epsilon ball
                    delta = torch.rand_like(x)
                    if self.args.norm==-1: # L_inf norm
                        xprime = x + delta*self.args.ptb/256.
                    else: # Now only implemented in L_inf norm
                        xprime = x
                else:
                    xprime = x
                yhat = network(x)
                loss_o += criterion(yhat, y).item()
                _, predicted = yhat.max(1)
                total_o += y.size(0)
                correct_o += predicted.eq(y).sum().item()

                # Perturbed data
                x_pgd = self.pgd(x, xprime, y, network, optimizer)
                yhat = network(x_pgd)
                loss_a += criterion(yhat, y)
                _, predicted = yhat.max(1)
                total_a += y.size(0)
                correct_a += predicted.eq(y).sum().item()

            test_loss_original = loss_o/(i+1)
            test_acc_original = correct_o/total_o
            test_loss_adv = loss_a/(i+1)
            test_acc_adv = correct_a/total_a

            # Update tensorboard loggers
            tb_logger.add_scalar('train_loss_original', train_loss_original, global_step=epoch)
            tb_logger.add_scalar('train_acc_original', train_acc_original, global_step=epoch)
            tb_logger.add_scalar('train_loss_adv', train_loss_adv, global_step=epoch)
            tb_logger.add_scalar('train_acc_adv', train_acc_adv, global_step=epoch)
            tb_logger.add_scalar('test_loss_original', test_loss_original, global_step=epoch)
            tb_logger.add_scalar('test_acc_original', test_acc_original, global_step=epoch)
            tb_logger.add_scalar('test_loss_adv', test_loss_adv, global_step=epoch)
            tb_logger.add_scalar('test_acc_adv', test_acc_adv, global_step=epoch)

            if epoch%self.config.training.snapshot_freq == 0:
                states = [
                    network.state_dict(),
                    optimizer.state_dict(),
                ]
                torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(epoch)))
                torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

            # Print for screen
            print('Epoch {}\t(Ori) {:.3f}\t|{:.3f}\t|{:.3f}\t|{:.3f}\t|'.format(epoch, train_loss_original, train_acc_original, test_loss_original, test_acc_original))
            print('Epoch {}\t(Adv) {:.3f}\t|{:.3f}\t|{:.3f}\t|{:.3f}\t|'.format(epoch, train_loss_adv, train_acc_adv, test_loss_adv, test_acc_adv))

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
            if self.config.training.random_start: # Random start in epsilon ball
                delta = torch.rand_like(x)
                if self.args.norm==-1: # L_inf norm
                    xprime = x + delta*self.args.ptb/256.
                else: # Now only implemented in L_inf norm
                    xprime = x
            else:
                xprime = x

            # Make adversarial examples
            x_pgd = self.pgd(x, xprime, y, network, optimizer)

            # Get loss and accuracy
            yhat = network(x_pgd)
            loss += criterion(yhat, y).item()
            _, predicted = yhat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            # Get original and adversarial examples and deploy images
            if i==1:
                yoriginal = network(xprime)
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
