import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation, ncsnv2_dsm_score_estimation, ncsnv2adv_dsm_score_estimation
from losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets.celeba import CelebA
from models.cond_refinenet_dilated import CondRefineNetDilated
from models.refinenet_dilated_baseline import RefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image
from scipy.fftpack import dct, idct

# Add path
import sys
path_root = '/path_root'
sys.path.append(path_root)
from adv_training.models import *
from transformation.augmix.augment_and_mix import augment_and_mix

__all__ = ['AnnealRunner_v3_dct2']


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

class AnnealRunner_v3_dct2():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.if_cifar = False
        if self.config.data.dataset == 'CIFAR10':
            self.if_cifar = True

    # Attack methods (fgsm is a special case of pgd with single iteration)
    def pgd(self, x, xprime, y, network, optimizer, alpha=2., ptb=8., iters=40):
        xprime_all = []
        for i in range(iters):
            if self.if_cifar:
                xprime = self.transform_cifar(xprime).clone().detach().requires_grad_(True)
            else:
                xprime = xprime.clone().detach().requires_grad_(True)
                
            yhat = network(xprime)

            optimizer.zero_grad()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(yhat, y)
            loss.backward()

            if self.if_cifar:
                xprime_grid = inv_transform_cifar(xprime)
            else:
                xprime_grid = xprime
            xprime = x + torch.clamp(torch.clamp(xprime_grid + xprime.grad.data.sign()*alpha/256., min=0.0, max=1.0) - x, min=-1.0*ptb/256., max=ptb/256.)
            # append xprime to xprime list
            xprime_all.append(xprime)
        return xprime_all

    def augment_and_mix_tensor(self, x, normalize=False):
        x = x.to('cpu').numpy()
        for i in range(x.shape[0]):
            x[i] = augment_and_mix(x[i], normalize=normalize)
        x = torch.Tensor(x).to('cuda')
        return x

    def dct(self, x, th_rate=0.1): # dct part
        x = x.to('cpu').numpy()
        
        # Take DCT to CIFAR-10 Image
        x_dct = dct(dct(x, axis=2, norm='ortho'), axis=3, norm='ortho') # DCT-II (2-D type 2 DCT)

        # Take Thresholding
        threshold = np.random.random(x.shape[0]) * 0.2
        for i in range(x.shape[0]):
            more_positive = np.greater(x[i], threshold[i])
            less_negative = np.less(x[i], -1.0*threshold[i])
            false_ignore = np.logical_or(more_positive, less_negative)
            x_dct[i] *= false_ignore

        threshold_small = np.maximum(threshold - th_rate, 0.)
        x_dct_adv = dct(dct(x, axis=2, norm='ortho'), axis=3, norm='ortho') # DCT-II (2-D type 2 DCT)
        for i in range(x.shape[0]):
            more_positive = np.greater(x[i], threshold_small[i])
            less_negative = np.less(x[i], -1.0*threshold_small[i])
            false_ignore = np.logical_or(more_positive, less_negative)
            x_dct_adv[i] *= false_ignore

        # Take IDCT to DCT features
        x_idct = idct(idct(x_dct, axis=2, norm='ortho'), axis=3, norm='ortho')
        x_idct = torch.Tensor(x_idct).to('cuda')

        x_idct_adv = idct(idct(x_dct_adv, axis=2, norm='ortho'), axis=3, norm='ortho')
        x_idct_adv = torch.Tensor(x_idct_adv).to('cuda')

        return x_idct, x_idct_adv

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=tran_transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True,
                                   transform=test_transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=tran_transform)
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=test_transform)

        # Added FashionMNIST
        elif self.config.data.dataset == 'FashionMNIST':
            dataset = FashionMNIST(os.path.join(self.args.run, 'datasets', 'fashionmnist'), train=True, download=True, transform=tran_transform)
            test_dataset = FashionMNIST(os.path.join(self.args.run, 'datasets', 'fashionmnist_test'), train=False, download=True, transform=test_transform)

        elif self.config.data.dataset == 'CELEBA':
            if self.config.data.random_flip:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                 ]), download=True)
            else:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.ToTensor(),
                                 ]), download=True)

            test_dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba_test'), split='test',
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                  ]), download=True)

        elif self.config.data.dataset == 'SVHN':
            dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                           transform=tran_transform)
            test_dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn_test'), split='test', download=True,
                                transform=test_transform)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=0, drop_last=True)

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        # CondRefineNetDilated: Obsolete by using NCSNv2
        '''
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)
        optimizer = self.get_optimizer(score.parameters())
        '''
        # Apply Technique 3: Use non-conditional RefineNet
        score = RefineNetDilated(self.config).to(self.config.device) #Modified this part
        score = torch.nn.DataParallel(score)
        optimizer = self.get_optimizer(score.parameters())
        
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 145000

        # The sigma distributions, signifcantly modified by Techniques 1, 2, and 4
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)


        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                #step += 1
                score.train()
                X = X.to(self.config.device)
                y = y.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)

                # Other kinds of augmentations
                X_aug, X_aug_adv = self.dct(X, th_rate=self.config.training.th_rate)
                step += 1
                if self.config.training.algo == 'dsm':
                    loss = ncsnv2adv_dsm_score_estimation(score, X_aug_adv, X_aug, labels, sigmas, self.config.training.anneal_power)
                    #loss = ncsnv2dct_dsm_score_estimation(score, X, X_aug, X_th, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X_aug, labels, sigmas,
                                                         n_particles=self.config.training.n_particles)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 1 == 0: # Estimation via data with Gaussian noise
                    score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)

                    with torch.no_grad():
                        #test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas, self.config.training.anneal_power)
                        test_dsm_loss = ncsnv2_dsm_score_estimation(score, test_X, test_labels, sigmas, self.config.training.anneal_power)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

    def Langevin_dynamics(self, x_mod, scorenet, n_steps=200, step_lr=0.00005):
        images = []

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
        labels = labels.long()

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_lr * grad + noise
                x_mod = x_mod
                print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            return images

    def anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
        images = []

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images


    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))

        score.eval()
        grid_size = 5

        imgs = []
        if self.config.data.dataset == 'MNIST' or self.config.data.dataset=='FashionMNIST':
            samples = torch.rand(grid_size ** 2, 1, 28, 28, device=self.config.device)
            #all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)
            all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 10, 3.1e-6)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))


        else: # Here CIFAR10
            samples = torch.rand(grid_size ** 2, 3, 32, 32, device=self.config.device)

            # Version 1 or version 2
            #all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)
            all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 5, 6.3e-6)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)), nrow=10)
                torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))

        imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)

    def anneal_Langevin_dynamics_inpainting(self, x_mod, refer_image, scorenet, sigmas, n_steps_each=100,
                                            step_lr=0.000008):
        images = []

        refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
        refer_image = refer_image.contiguous().view(-1, 3, 32, 32)
        x_mod = x_mod.view(-1, 3, 32 ,32)
        half_refer_image = refer_image[..., :16]
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="annealed Langevin dynamics sampling"):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :16] = corrupted_half_image
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    x_mod[:, :, :, :16] = corrupted_half_image
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images

    def test_inpainting(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))
        score.eval()

        imgs = []
        if self.config.data.dataset == 'CELEBA':
            dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='test',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(self.config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
            refer_image, _ = next(iter(dataloader))

            samples = torch.rand(20, 20, 3, self.config.data.image_size, self.config.data.image_size,
                                 device=self.config.device)

            #all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)
            all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 5, 3.3e-6)
            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(400, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))

        else:
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            if self.config.data.dataset == 'CIFAR10':
                dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                                  transform=transform)
            elif self.config.data.dataset == 'SVHN':
                dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                               transform=transform)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
            data_iter = iter(dataloader)
            refer_image, _ = next(data_iter)

            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))
            samples = torch.rand(20, 20, self.config.data.channels, self.config.data.image_size,
                                 self.config.data.image_size).to(self.config.device)

            if self.config.data.dataset == 'CIFAR10':
                all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 5, 6.30e-6)
            elif self.config.data.dataset == 'MNIST' or self.config.data.dataset == 'FashionMNIST':
                all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 5, 3.10e-6)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(400, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))


        imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)
