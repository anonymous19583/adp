# Software package
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from PIL import Image
import yaml
import sys
from sklearn.manifold import TSNE
from scipy.spatial import Voronoi, voronoi_plot_2d

# Model package
from utils.importData import importData
from adv_training.models import *
from autoattack import AutoAttack

# NCSN package
from ncsn.models.cond_refinenet_dilated import CondRefineNetDilated
from ncsn.models.refinenet_dilated_baseline import RefineNetDilated
from torchvision.utils import save_image, make_grid
import torchvision.transforms  as transforms

# matplotlib package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import tqdm

### Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MNIST', help='Dataset, list: MNIST, FashionMNIST, CIFAR10')
parser.add_argument('--image_folder', default='imgs', help='Image folder path')

# Execution mode flags (can be multiple: runs separately)
parser.add_argument('--TSNE', action='store_true', help='See decision boundary of different attacks with respect to a single data')
parser.add_argument('--TSNE_CLASS', action='store_true', help='See t-SNE of purification with respect to different classes')
parser.add_argument('--HP_TUNING', action='store_true', help='Hyperparameter tuning with 500 validation set')
parser.add_argument('--TEST', action='store_true', help='Default test mode')

# Network parameters (EBM structure in config.yml)
parser.add_argument('--att_log', default='X', help='Attacker model path')
parser.add_argument('--ebm_log', default='X', help='EBM path')
parser.add_argument('--clf_log', default='X', help='Classifier model path')
parser.add_argument('--network', default='X', help='Attacker structure')
parser.add_argument('--classifier', default='X', help='Classifier structure')

# Attack details
parser.add_argument('--attack_method', default='fgsm', help='Attack method: list [fgsm, pgd, bim, pgd_white, bim_white, bpda]')
parser.add_argument('--ptb', type=float, default=8, help='e-ball size: # pixels for l_inf norm / maximum norm for l_1, l_2 norm') # Active for lp attacks
parser.add_argument('--random_start', default=False)
parser.add_argument('--ball_dim', type=int, default=-1, help='norm type of epsilon ball, [-1:l_inf, 1:l_1, 2:l_2]')
parser.add_argument('--pgdwhite_eps', default=2., type=float, help='Learning rate (/256) at one-shot unrolling attack (pgd_white, bim_white)')
parser.add_argument('--attack_start_sigma', default=0.01, type=float, help='Attack step size at 1st stage')
parser.add_argument('--attack_decay', default=0.01, type=float, help='Final attack decayed rate: Last attack step size (attack_start_sigma*attack_decay)')
parser.add_argument('--attack_alpha', default=0.05, type=float, help='Adaptive attack step size strategy')
parser.add_argument('--n_eot', default=1, type=int, help='number of EOT attacks')

# Purification
parser.add_argument('--purify_method', default='projection', help='Purification method, list: [projection, adaptive]')
parser.add_argument('--attack_step_decision', default='projection', help='How to decide attack step size, list: [projection, adaptive]')
parser.add_argument('--rand_smoothing', default=False, type=bool, help='Randomized smoothing after purification')
parser.add_argument('--smoothing_level', default=2., type=float, help='# pixels of randomized smoothing')
parser.add_argument('--init_noise', default=0., type=float, help='Noise before purification')
parser.add_argument('--input_ensemble', default=1, type=int, help='number of noisy inputs')

# Common corruption analysis
parser.add_argument('--CIFARC_CLASS', default=-1, type=int, help='Class of corruption, 1~15')
parser.add_argument('--CIFARC_SEV', default=0, type=int, help='Severity of corruption, 1~5')

# Learning rate parameters: Exponential decay
parser.add_argument('--start_sigma', default=0.01, type=float, help='Purifying step size at 1st stage')
parser.add_argument('--decay', default=0.01, type=float, help='Final decayed rate: Last step size (start_sigma*decay)')
parser.add_argument('--n_stages', default=10, type=int, help='# purification stages')
parser.add_argument('--alpha', default=0.2, type=float, help='Adaptive step size strategy at [adaptive]')
parser.add_argument('--e1', default=0.5, type=float, help='extra e1')
parser.add_argument('--e2', default=0.5, type=float, help='extra e2')

args = parser.parse_args()

if args.att_log=='X':
    sys.exit('Attacker model is not defined')
else:
    print('Attacker model {}'.format(args.att_log))
if args.network=='X':
    sys.exit('Attacker network structure is not specified')
else:
    print('Attacker network {}'.format(args.network))
print('Attack method {}, epsilon {}'.format(args.attack_method, args.ptb))
if args.ebm_log=='X':
    sys.exit('EBM model path is not defined')
else:
    print('EBM model {}'.format(args.ebm_log))
if args.clf_log=='X':
    sys.exit('Classifier model path is not defined')
else:
    print('Classifier model {}'.format(args.clf_log))
if args.classifier=='X':
    sys.exit('Classifier network structure is not specified')
else:
    print('Classifier network {}'.format(args.classifier))

# Get input dataset
if args.dataset == 'CIFAR10C':
    testLoader_list, corruption_list = importData(dataset=args.dataset, train=False, shuffle=True, bsize=100)
    testLoader = testLoader_list[args.CIFARC_CLASS-1][args.CIFARC_SEV-1]
else:
    testLoader = importData(dataset=args.dataset, train=False, shuffle=True, bsize=100)
testIter = iter(testLoader)
if args.dataset=='CIFAR10':
    if_cifar = True
    if_cifarc = False
elif args.dataset=='CIFAR10C':
    if_cifar = True
    if_cifarc = True
else:
    if_cifar = False
    if_cifarc = False
# Make dir(s)
if not os.path.exists(os.path.join("log_images",args.image_folder)):
    os.mkdir(os.path.join("log_images", args.image_folder))

grid_size = 10
# Transforms for cifar-10 dataset
mean_cifar = (0.4914, 0.4822, 0.4465)
std_cifar = (0.2023, 0.1994, 0.2010)
class transform_raw_to_grid(object):
    def __call__(self,tensor):
        tensor *= 255./256.
        tensor += 1./512.
        return tensor
class transform_grid_to_raw(object):
    def __call__(self,tensor):
        tensor *= 256./255.
        tensor -= 1./510.
        return tensor

transform_cifar = transforms.Compose( # Grid image to normalized raw image (cifar classifier/Compute gradient)
    [transform_grid_to_raw(),
     transforms.Normalize(mean_cifar, std_cifar)])
inv_transform_cifar = transforms.Compose( # Normalized raw image to grid image (EBM/attack)
    [transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1./0.2023, 1./0.1994, 1./0.2010)),
     transform_raw_to_grid()])

def one_step_propagation(center, current, move):
    return torch.clamp(center + torch.clamp(current - center + torch.clamp(move, min=-1.0*args.alpha/256., max=args.alpha/256.), \
         min=-1.0*args.ptb/256., max=args.ptb/256.), min=0.0, max=1.0)

# Attack methods
# Ball_dim -1 -> L_inf ball
# BIM: Gradient update from x
def bim(x, xprime, y, network, optimizer, alpha=2., ptb=8., iters=40, ball_dim=args.ball_dim): # Input x and xprime are raw grid data
    for i in range(iters):
        if if_cifar:
            xprime = transform_cifar(xprime).clone().detach().requires_grad_(True)
        else:
            xprime = xprime.clone().detach().requires_grad_(True)
        yhat = network(xprime)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(yhat, y)
        loss.backward()

        if if_cifar: # Move to grid (EBM/attack)
            xprime_grid = inv_transform_cifar(xprime)
        else:
            xprime_grid = xprime
        
        if ball_dim==-1: # l_inf ball
            xprime = one_step_propagation(x, xprime_grid, xprime.grad.data.sign()*alpha/256.)
        else:
            xprime = x + ptb/torch.norm((torch.clamp(xprime_grid + xprime.grad.data.sign()*alpha/256., min=0.0, max=1.0) - x).view(x.shape[0],-1), dim=1, p=ball_dim)[:,None,None,None]*(torch.clamp(xprime_grid + xprime.grad.data.sign()*alpha/256., min=0.0, max=1.0) - x)
    return xprime

# PGD: Gradient update from random data in epsilon ball
def pgd(x, xprime, y, network, optimizer, alpha=2., ptb=8., iters=40, ball_dim=args.ball_dim): # Input x and xprime are raw grid data
    if ball_dim==-1:
        xprime = torch.clamp(x + 2.*ptb/256.*(torch.rand_like(x)-0.5), min=0.0, max=1.0)
    else:
        rand_unif = 2.*(torch.rand_like(x)-0.5)
        rand_unif = ptb/torch.norm(rand_unif.view(x.shape[0],-1), dim=1, p=ball_dim)[:,None,None,None]*rand_unif
        xprime = x + rand_unif
    for i in range(iters):
        if if_cifar:
            xprime = transform_cifar(xprime).clone().detach().requires_grad_(True)
        else:
            xprime = xprime.clone().detach().requires_grad_(True)
        yhat = network(xprime)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(yhat, y)
        loss.backward()

        if if_cifar: # Move to grid (EBM/attack)
            xprime_grid = inv_transform_cifar(xprime)
        else:
            xprime_grid = xprime
        if ball_dim==-1:
            xprime = one_step_propagation(x, xprime_grid, xprime.grad.data.sign()*alpha/256.)
        else:
            xprime = x + ptb/torch.norm((torch.clamp(xprime_grid + xprime.grad.data.sign()*alpha/256., min=0.0, max=1.0) - x).view(x.shape[0],-1), dim=1, p=ball_dim)[:,None,None,None]* \
                (torch.clamp(xprime_grid + xprime.grad.data.sign()*alpha/256., min=0.0, max=1.0) - x)
    return xprime

# BIM_WHITE: One-step unrolling attack from x
def bim_white(x, y, optimizer, scorenet, clfnet, sigmas, alpha=2., ptb=8., iters=40):
    x_start = x.clone().detach()
    for i in range(iters):
        if if_cifar:
            xp = transform_cifar(x).clone().detach().requires_grad_(True)
        else:
            xp = x.clone().detach().requires_grad_(True)
        yhat = full_model(xp, scorenet, clfnet)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(yhat, y)
        loss.backward()
        x = one_step_propagation(x_start, x, xp.grad.data.sign()*alpha/256.)
    return x

# PGD_WHITE: One-step unrolling attack from random data in epsilon ball
def pgd_white(x, y, optimizer, scorenet, clfnet, sigmas, alpha=2., ptb=8., iters=40):
    x_start = torch.clamp(x + (torch.rand_like(x)-0.5)*2.*ptb/256., min=0.0, max=1.0).clone().detach()
    for i in range(iters):
        if if_cifar:
            xp = transform_cifar(x_start).clone().detach().requires_grad_(True)
        else:
            xp = x_start.clone().detach().requires_grad_(True)
        yhat = full_model(xp, scorenet, clfnet)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(yhat, y)
        loss.backward()
        x_start = one_step_propagation(x, x_start, xp.grad.data.sign()*alpha/256.)
    return x_start

# pgd_iter: Iterate making PGD attacked data then purify the data
def pgd_iter(x, y, optimizer, scorenet, clfnet, sigmas, ptb=8., iters=40, purify_iter=10):
    x_list_before = []
    x_list_after = []
    x = x.clone().detach().to('cuda')
    x_start = torch.clamp(x + (torch.rand_like(x)-0.5)*2.*ptb/256., min=0.0, max=1.0).clone().detach().to('cuda')
    x_t = x_start
    for i in range(purify_iter):
        x_pgd = bim(x=x_t.to('cuda'), xprime=x_t.to('cuda'), y=y, network=clfnet, optimizer=optimizer, ptb=ptb)
        if args.attack_step_decision=='projection':
            x_purify = proj_scorenet(x_pgd, scorenet, sigmas)[-1].clone().detach()
        elif args.attack_step_decision=='adaptive':
            x_purify = adaptive_proj_scorenet(x_pgd, scorenet, save_score_norm=True, alpha=args.attack_alpha)[0][-1].clone().detach()
        else:
            sys.exit()
        x_purify = torch.clamp(x + torch.clamp(x_purify.to('cuda')-x, min=-1.0*ptb/256, max=1.0*ptb/256), min=0.0, max=1.0)
        x_list_before.append(x_pgd.clone().detach())
        x_list_after.append(x_purify)
        x_t = x_purify
    return x_list_before, x_list_after, x_list_before[-1]

def bpda(x, y, optimizer, scorenet, clfnet, sigmas, alpha=2., ptb=8., iters=40):
    # x : original data (to modify)
    xp = x.clone().detach().requires_grad_(True)
    for i in range(iters):
        xgrad_sum = torch.zeros_like(xp)
        for j in range(args.n_eot):
            xpt = torch.clamp(xp + torch.randn_like(xp)*args.init_noise/256.,min=0.0,max=1.0).clone().detach().requires_grad_(True)
            if args.attack_step_decision=='projection':
                x_ebm = proj_scorenet(xpt, scorenet, sigmas)
            elif args.attack_step_decision=='adaptive':
                x_ebm = adaptive_proj_scorenet(xpt, scorenet, save_score_norm=True, alpha=args.attack_alpha)[0]
            else:
                sys.exit()
            if if_cifar:
                xprime = transform_cifar(x_ebm[-1]).clone().detach().requires_grad_(True)
            else:
                xprime = x_ebm[-1].clone().detach().requires_grad_(True)
            yhat = network(xprime)
            optimizer.zero_grad()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(yhat, y)
            loss.backward()
            xgrad_sum += xprime.grad.data
        grad_sign = xgrad_sum.sign()
        if ball_dim==-1: # l_inf ball
            xp = x + torch.clamp(torch.clamp(xp + grad_sign.to('cuda')*alpha/256., min=0.0, max=1.0) - x, min=-1.0*ptb/256., max=ptb/256.)
        else:
            xp = x + ptb/torch.norm((torch.clamp(xp + grad_sign.to('cuda')*alpha/256., min=0.0, max=1.0) - x).view(x.shape[0],-1), dim=1, p=ball_dim)[:,None,None,None]*(torch.clamp(xprime_grid + xprime.grad.data.sign()*alpha/256., min=0.0, max=1.0) - x)
    return xp

def pgd_forward_full(x, y, optimizer, scorenet, clfnet, e1, e2, sigmas, alpha=2., ptb=8., iters=40, ball_dim=args.ball_dim):
    xprime = x.clone().detach().requires_grad_(False)
    for i in range(iters):
        if if_cifar:
            xprime_t = transform_cifar(xprime).clone().detach().requires_grad_(True)
        else:
            xprime_t = xprime.clone().detach().requires_grad_(True)
        with torch.no_grad():
            if args.attack_step_decision=='projection':
                score_full = proj_scorenet(xprime, scorenet, sigmas)[-1].clone().detach().to('cuda') - xprime
            elif args.attack_step_decision=='adaptive':
                score_full = adaptive_proj_scorenet(xprime, scorenet, save_score_norm=True, alpha=args.attack_alpha)[0][-1].clone().detach().to('cuda') - xprime
            else:
                sys.exit()
        yhat = clfnet(xprime_t)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(yhat, y)
        loss.backward()

        if ball_dim==-1: # l_inf ball
            xprime = x + torch.clamp(torch.clamp(xprime + (score_full.sign()*e1 + xprime_t.grad.data.sign()*e2)*alpha/256., min=0.0, max=1.0) - x, min=-1.0*ptb/256., max=ptb/256.)
        else:
            return
    
    return xprime

def pgd_forward_one(x, y, optimizer, scorenet, clfnet, e1, e2, sigmas, alpha=2., ptb=8., iters=100, ball_dim=args.ball_dim):
    xprime = x.clone().detach().requires_grad_(False)
    labels = torch.ones(xprime.shape[0], device=xprime.device)*10
    labels = labels.long()
    for i in range(iters):
        if if_cifar:
            xprime_t = transform_cifar(xprime).clone().detach().requires_grad_(True)
        else:
            xprime_t = xprime.clone().detach().requires_grad_(True)
        with torch.no_grad():
            score_full = scorenet(xprime, labels)
        yhat = clfnet(xprime_t)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(yhat, y)
        loss.backward()

        if ball_dim==-1: # l_inf ball
            xprime = x + torch.clamp(torch.clamp(xprime + (score_full.sign()*e1 +  xprime_t.grad.data.sign()*e2)* alpha/256., min=0.0, max=1.0) - x, min=-1.0*ptb/256., max=ptb/256.)
        else:
            return
    
    return xprime

def spsa(x, y, scorenet, clfnet, sigmas, n_samples=32, alpha=2., ptb=8., iters=40, ball_dim=args.ball_dim):
    criterion = nn.CrossEntropyLoss(reduction='none')
    delta = 1./256.
    with torch.no_grad():
        x_init = x.clone().detach().requires_grad_(False)
        for i in range(iters):
            x_grad = torch.zeros_like(x)
            for k in range(args.n_eot):
                x_eot = torch.clamp(x + torch.randn_like(x)*args.init_noise/256., min=0.0, max=1.0).clone().detach().to('cuda')
                if args.attack_step_decision=='projection':
                    xprime_minus = proj_scorenet(x_eot, scorenet, sigmas)[-1].clone().detach().to('cuda')
                elif args.attack_step_decision=='adaptive':
                    xprime_minus = adaptive_proj_scorenet(x_eot, scorenet, save_score_norm=True, alpha=args.attack_alpha)[0][-1].clone().detach().to('cuda')
                else:
                    sys.exit()
                if if_cifar:
                    xprime_minus = transform_cifar(xprime_minus)
                loss_minus = criterion(clfnet(xprime_minus), y)
                for j in range(n_samples):
                    v = (torch.randint(high=2, size=x.shape)*2.0 - 1.0).to('cuda')
                    if args.attack_step_decision=='projection':
                        xprime_plus = proj_scorenet(x_eot+delta*v, scorenet, sigmas)[-1].clone().detach().to('cuda')
                        #xprime_minus = proj_scorenet(x-delta*v, scorenet, sigmas)[-1].clone().detach().to('cuda')
                    elif args.attack_step_decision=='adaptive':
                        xprime_plus = adaptive_proj_scorenet(x_eot+delta*v, scorenet, save_score_norm=True, alpha=args.attack_alpha)[0][-1].clone().detach().to('cuda')
                        #xprime_minus = adaptive_proj_scorenet(x-delta*v, scorenet, save_score_norm=True, alpha=args.attack_alpha)[0][-1].clone().detach().to('cuda')
                    else:
                        sys.exit()
                    if if_cifar:
                        xprime_plus = transform_cifar(xprime_plus)
                        #xprime_minus = transform_cifar(xprime_minus)
                    loss_plus = criterion(clfnet(xprime_plus), y)
                    #loss_minus = criterion(clfnet(xprime_minus), y)
                    loss_diff = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(loss_plus-loss_minus, 1), 2), 3).repeat(1, 3, 32,32)
                    #x_grad += loss_diff * v / (2*delta) 
                    x_grad += loss_diff * v / (delta)
            x_grad /= (n_samples*args.n_eot)
            x = x_init + torch.clamp(torch.clamp(x_grad, min=-1.0*alpha/256., max=alpha/256.) + x - x_init, min=-1.0*ptb/256., max=ptb/256.)
    return x


def spsa_unbounded(x, y, scorenet, clfnet, sigmas, n_samples=32, alpha=2., ptb=8., iters=40, ball_dim=args.ball_dim):
    criterion = nn.CrossEntropyLoss(reduction='none')
    delta = 1./256.
    with torch.no_grad():
        x_init = x.clone().detach().requires_grad_(False)
        for i in range(iters):
            x_grad = torch.zeros_like(x)
            if args.attack_step_decision=='projection':
                xprime_minus = proj_scorenet(x, scorenet, sigmas)[-1].clone().detach().to('cuda')
            elif args.attack_step_decision=='adaptive':
                xprime_minus = adaptive_proj_scorenet(x, scorenet, save_score_norm=True, alpha=args.attack_alpha)[0][-1].clone().detach().to('cuda')
            else:
                sys.exit()
            if if_cifar:
                xprime_minus = transform_cifar(xprime_minus)
            loss_minus = criterion(clfnet(xprime_minus), y)
            for j in range(n_samples):
                v = (torch.randint(high=2, size=x.shape)*2.0 - 1.0).to('cuda')
                if args.attack_step_decision=='projection':
                    xprime_plus = proj_scorenet(x+delta*v, scorenet, sigmas)[-1].clone().detach().to('cuda')
                    #xprime_minus = proj_scorenet(x-delta*v, scorenet, sigmas)[-1].clone().detach().to('cuda')
                elif args.attack_step_decision=='adaptive':
                    xprime_plus = adaptive_proj_scorenet(x+delta*v, scorenet, save_score_norm=True, alpha=args.attack_alpha)[0][-1].clone().detach().to('cuda')
                    #xprime_minus = adaptive_proj_scorenet(x-delta*v, scorenet, save_score_norm=True, alpha=args.attack_alpha)[0][-1].clone().detach().to('cuda')
                else:
                    sys.exit()
                if if_cifar:
                    xprime_plus = transform_cifar(xprime_plus)
                    #xprime_minus = transform_cifar(xprime_minus)
                loss_plus = criterion(clfnet(xprime_plus), y)
                #loss_minus = criterion(clfnet(xprime_minus), y)
                loss_diff = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(loss_plus-loss_minus, 1), 2), 3).repeat(1, 3, 32,32)
                #x_grad += loss_diff * v / (2*delta) 
                x_grad += loss_diff * v / (delta)
            x_grad /= n_samples
            # Print x_grad norm every iteration
            x_grad_norm = torch.norm(x_grad.to('cuda').view(x_grad.shape[0], -1), p=2, dim=1).mean().item()
            print(x_grad_norm)
            x = x_init + torch.clamp(x_grad + x - x_init, min=-1.0*ptb/256., max=ptb/256.)
    return x


def nes(x, y, scorenet, clfnet, sigmas, n_samples=32, alpha=2., ptb=8., iters=40, ball_dim=args.ball_dim):
    criterion = nn.CrossEntropyLoss(reduction='none')
    delta = 1./256.
    with torch.no_grad():
        x_init = x.clone().detach().requires_grad_(False)
        for i in range(iters):
            x_grad = torch.zeros_like(x)
            for j in range(n_samples):
                v = (torch.randn_like(x)).to('cuda')
                if args.attack_step_decision=='projection':
                    xprime_plus = proj_scorenet(x+delta*v, scorenet, sigmas)[-1].clone().detach().to('cuda')
                    xprime_minus = proj_scorenet(x-delta*v, scorenet, sigmas)[-1].clone().detach().to('cuda')
                elif args.attack_step_decision=='adaptive':
                    xprime_plus = adaptive_proj_scorenet(x+delta*v, scorenet, save_score_norm=True, alpha=args.attack_alpha)[0][-1].clone().detach().to('cuda')
                    xprime_minus = adaptive_proj_scorenet(x-delta*v, scorenet, save_score_norm=True, alpha=args.attack_alpha)[0][-1].clone().detach().to('cuda')
                else:
                    sys.exit()
                if if_cifar:
                    xprime_plus = transform_cifar(xprime_plus)
                    xprime_minus = transform_cifar(xprime_minus)
                loss_plus = criterion(clfnet(xprime_plus), y)
                loss_minus = criterion(clfnet(xprime_minus), y)
                loss_diff = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(loss_plus-loss_minus, 1), 2), 3).repeat(1, 3, 32,32)
                x_grad += loss_diff * v / (2*delta) 
            x_grad /= n_samples
            x = x_init + torch.clamp(torch.clamp(x_grad, min=-1.0*alpha/256., max=alpha/256.) + x - x_init, min=-1.0*ptb/256., max=ptb/256.)
    return x

#def squareattack(x, y, scorenet, clfnet, sigmas):
#    # x should be NxCxHxW formatted
#    adversary = AutoAttack(forward_pass, norm='Linf', eps=8./255., version='standard')

def nattack(x, y, scorenet, clfnet, sigmas, n_samples=32, alpha=2., ptb=8., iters=40, ball_dim=args.ball_dim): # Do here
    criterion = nn.CrossEntropyLoss(reduction='none')
    stdev = 0.1
    lr = 0.008
    with torch.no_grad():
        mu = torch.arctanh(2.*x - 1.) + stdev*torch.randn_like(x)
        for i in range(iters):
            losslist = []
            epslist = []
            for j in range(n_samples):
                eps = torch.randn_like(x)
                epslist.append(eps)
                g = (torch.tanh(mu + stdev*eps)+1.)/2.
                g = x + torch.clamp(torch.clamp(g, min=-1.0*alpha/256., max=alpha/256.) - x, min=-1.0*ptb/256., max=ptb/256.)
                gprime = proj_scorenet(g, scorenet, sigmas)[-1].clone().detach().to('cuda')
                if if_cifar:
                    gprime = transform_cifar(gprime)
                loss = criterion(clfnet(gprime), y).cpu().numpy()
                losslist.append(loss)
            loss_array = np.asarray(losslist)
            f_mu = np.mean(loss_array, axis=0)
            f_sigma = np.std(loss_array, axis=0)
            for j in range(n_samples):
                f_norm = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.from_numpy((losslist[j]-f_mu)/(f_sigma+1.0e-6)), 1), 2), 3).to('cuda')
                mu += f_norm.repeat(1, 3, 32, 32).to('cuda') * epslist[j].to('cuda') * (lr/n_samples/stdev)
    return (torch.tanh(mu) + 1.)/2.

def sn_disable_mixed(x, y, optimizer, scorenet, clfnet, sigmas, e1, e2, alpha=2., ptb=8., iters=100, ball_dim=args.ball_dim):
    eps = torch.rand_like(x)*(2.*ptb/256.)-ptb/256.
    if ball_dim==-1:
        xprime = torch.clamp(x + eps, min=0.0, max=1.0).clone().detach().requires_grad_(True)
    else:
        rand_unif = 2.*(torch.rand_like(x)-0.5)
        rand_unif = ptb/torch.norm(rand_unif.view(x.shape[0],-1), dim=1, p=ball_dim)[:,None,None,None]*rand_unif
        xprime = x + rand_unif
    cos = torch.nn.CosineSimilarity()
    criterion = nn.CrossEntropyLoss()
    labels = torch.ones(xprime.shape[0], device=xprime.device)*10
    labels = labels.long()
    for i in range(iters):
        eps = xprime - x
        xprime = xprime.clone().detach().requires_grad_(True)
        score_full = scorenet(xprime, labels)
        optimizer.zero_grad()
        loss = cos(score_full.view(grid_size**2, -1), eps.view(grid_size**2, -1)).mean()
        loss.backward()
        xprime_grad = xprime.grad.data.sign().clone().detach()
        
        xprime_clf = transform_cifar(xprime.clone().detach()).clone().detach().requires_grad_(True)
        yhat = clfnet(xprime_clf)
        loss2 = criterion(yhat, y)
        loss2.backward()
        xprime_clf_grad = xprime_clf.grad.data.sign().clone().detach()
        if ball_dim==-1: # l_inf ball
            xprime = x + torch.clamp(torch.clamp(xprime + (xprime_clf_grad*e1 +  xprime_grad*e2)*alpha/256., min=0.0, max=1.0) - x, min=-1.0*ptb/256., max=ptb/256.)
        else:
            return
    
    return xprime, loss.item()

def sn_disable(x, y, optimizer, scorenet, sigmas, alpha=2., ptb=8., iters=40, ball_dim=args.ball_dim):
    eps = torch.rand_like(x)*(2.*ptb/256.)-ptb/256.
    if ball_dim==-1:
        xprime = torch.clamp(x + eps, min=0.0, max=1.0).clone().detach().requires_grad_(True)
    else:
        rand_unif = 2.*(torch.rand_like(x)-0.5)
        rand_unif = ptb/torch.norm(rand_unif.view(x.shape[0],-1), dim=1, p=ball_dim)[:,None,None,None]*rand_unif
        xprime = x + rand_unif
    cos = torch.nn.CosineSimilarity()
    labels = torch.ones(xprime.shape[0], device=xprime.device)*10
    labels = labels.long()
    for i in range(iters):
        xprime = xprime.clone().detach().requires_grad_(True)
        score_full = scorenet(xprime, labels)
        eps = (xprime - x).clone().detach().requires_grad_(False)
        optimizer.zero_grad()
        loss = -1.0*cos(score_full.view(grid_size**2, -1), eps.view(grid_size**2, -1)).mean()
        loss.backward()
        if ball_dim==-1: # l_inf ball
            xprime = x + torch.clamp(torch.clamp(xprime + xprime.grad.data.sign()*alpha/256., min=0.0, max=1.0) - x, min=-1.0*ptb/256., max=ptb/256.)
        else:
            return
    
    return xprime

        
### Purification methods
# Annealed Langevin dynamics implementation
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas):
    images = []
    with torch.no_grad():
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling', disable=True):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
            noise = torch.randn_like(x_mod) * np.sqrt(sigma * 2)
            grad = scorenet(x_mod, labels)
            x_mod = x_mod + sigma * grad + noise
            x_mod = torch.clamp(x_mod, 0.0, 1.0)
            images.append(torch.clamp(x_mod,0.0,1.0).to('cpu'))
        return images

# proj_scorenet: iterating projection steps
def proj_scorenet(x_mod, scorenet, sigmas, save_score_norm=False):
    images = []
    grads = []
    grad_stds = []
    with torch.no_grad():
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='projection', disable=True):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device)*c
            labels = labels.long()
            images.append(x_mod.to('cpu'))
            grad = scorenet(x_mod, labels)
            if save_score_norm==True:
                grads.append(grad)
                grad_noisys = []
                for j in range(10): # Get score standard deviation
                    x_mod_noisy = torch.clamp(x_mod + torch.randn_like(x_mod)*(2./256.), min=0.0, max=1.0)
                    grad_noisy = scorenet(x_mod_noisy, labels)
                    grad_noisys.append(grad_noisy)
                grad_noisys = torch.stack(grad_noisys, dim=0)
                grad_std = torch.std(grad_noisys, dim=0)
                grad_stds.append(grad_std)
            x_mod = torch.clamp(x_mod+sigma*grad, 0.0, 1.0)
    if not save_score_norm:
        return images
    else:
        return images, grads, grad_stds

# adaptive_scorenet: iterating projection steps, with adaptive rate
def adaptive_proj_scorenet(x_mod, scorenet, min_step_lr=1.0e-5, alpha=0.1, save_score_norm=False):
    images = []
    grads = []
    grad_stds = []
    lr_min = 1.0e-6
    step_sizes = []
    with torch.no_grad():
        for i in range(len(sigmas)):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device)*10
            labels = labels.long()
            images.append(x_mod.to('cpu'))
            grad = scorenet(x_mod, labels)
            x_mod_eps = x_mod+lr_min*grad
            grad_eps = scorenet(x_mod_eps, labels)
            
            z1 = torch.bmm(grad.view(grad.shape[0],1,-1), grad_eps.view(grad_eps.shape[0],-1,1))
            z2 = torch.bmm(grad.view(grad.shape[0],1,-1), grad.view(grad.shape[0],-1,1))
            z = torch.div(z1, z2)
            step_size = torch.clamp(alpha*lr_min/(1.-z), min=min_step_lr, max=min_step_lr*10000.).view(-1)
            x_mod = torch.clamp(x_mod+grad*step_size[:,None,None,None], 0.0, 1.0)
            step_sizes.append(step_size)
            if save_score_norm==True:
                grads.append(grad)
                grad_noisys = []
                for j in range(10): # 10 near data
                    x_mod_noisy = torch.clamp(x_mod + torch.randn_like(x_mod)*(2./256.), min=0.0, max=1.0)
                    grad_noisy = scorenet(x_mod_noisy, labels)
                    grad_noisys.append(grad_noisy)
                grad_noisys = torch.stack(grad_noisys, dim=0)
                grad_std = torch.std(grad_noisys, dim=0)
                grad_stds.append(grad_std)
    if save_score_norm==False:
        return images, step_sizes
    else:
        return images, grads, grad_stds, step_sizes

# anneal_noise
def anneal_noise(x_mod, scorenet, sigmas):
    images = []
    with torch.no_grad():
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealing noise', disable=True):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
            noise = torch.randn_like(x_mod) * np.sqrt(sigma * 2)
            x_mod = x_mod + noise
            x_mod = torch.clamp(x_mod, 0.0, 1.0)
            images.append(torch.clamp(x_mod,0.0,1.0).to('cpu'))
        return images

# full_model: One-step purification + classification
def full_model(x, scorenet, clfnet):
    labels = torch.ones(x.shape[0], device=x.device)*10
    labels = labels.long()
    if if_cifar:
        step_size = args.pgdwhite_eps/256.
    else:
        step_size = args.pgdwhite_eps/256.
    if if_cifar:
        x = inv_transform_cifar(x)
    x = x+step_size*scorenet(x, labels) # Don't clamp to get gradients
    if if_cifar:
        x = transform_cifar(x)
    y = clfnet(x)
    return y

def attack_qual_check(x, scorenet, data_type):
    labels = torch.ones(x.shape[0], device=x.device)*10
    labels = labels.long()
    flags = data_type*np.ones((x.shape[0]), dtype=int)
    lr_min = 1.0e-6
    with torch.no_grad():
        grad = scorenet(x, labels)
        x_eps = x+lr_min*grad
        grad_eps = scorenet(x_eps, labels)
            
    z1 = torch.bmm(grad.view(grad.shape[0],1,-1), grad_eps.view(grad_eps.shape[0],-1,1))
    z2 = torch.bmm(grad.view(grad.shape[0],1,-1), grad.view(grad.shape[0],-1,1))
    z = torch.div(z1, z2)
    step_size = lr_min/torch.clamp(1.-z, min=1.0e-8)

    scorelist = torch.norm(grad.view(100,-1), dim=1).squeeze().detach().cpu().numpy()
    alphalist = np.clip(step_size.cpu().numpy(), a_min=1.0e-7, a_max=None)

    return scorelist, alphalist, flags

# Setting hyperparameters
n_iters = 100 # x100: total number of data
if args.TSNE or args.TSNE_CLASS: # t-SNE deploying mode
    n_iters = 1
if args.HP_TUNING: # Hyperparameter tuning mode
    n_iters = 5
n_stages = args.n_stages # number of purification stages
start_sigma = args.start_sigma
end_sigma = args.start_sigma*args.decay
attack_start_sigma = args.attack_start_sigma
attack_end_sigma = args.attack_start_sigma*args.attack_decay
sigmas = np.exp(np.linspace(np.log(start_sigma), np.log(end_sigma), n_stages))
attack_sigmas = np.exp(np.linspace(np.log(attack_start_sigma), np.log(attack_end_sigma), n_stages))

## metrics
# Accuracy
acc_x = np.zeros((1+len(sigmas))) # Original examples
acc_xadv = np.zeros((1+len(sigmas))) # Adversarially attacked examples
if args.rand_smoothing:
    acc_rand = np.zeros((1)) # ensemble average over 10 randomization (one-hot)
    acc_rand_adv = np.zeros((1))
    acc_rand_logit = np.zeros((1)) # ensemble average over 10 randomization (logit)
    acc_rand_adv_logit = np.zeros((1))
acc_x_ensemble_one_hot = np.zeros((1))
acc_x_ensemble_logit = np.zeros((1))
acc_xadv_ensemble_one_hot = np.zeros((1))
acc_xadv_ensemble_logit = np.zeros((1))

acc_noisy_x_ensemble_one_hot = np.zeros((1))
acc_noisy_x_ensemble_logit = np.zeros((1))
acc_noisy_xadv_ensemble_one_hot = np.zeros((1))
acc_noisy_xadv_ensemble_logit = np.zeros((1))


# l2 distance
l2 = np.zeros((1+len(sigmas))) # Original examples
l2_adv = np.zeros((1+len(sigmas))) # Adversarially attacked examples
l2_diff = np.zeros((1+len(sigmas))) # Average difference between them

# Auxiliary metrics
score_norm = np.zeros((len(sigmas)))
score_std = np.zeros((len(sigmas)))
cos_sim = np.zeros((len(sigmas)-1))
score_norm_adv = np.zeros((len(sigmas)))
score_std_adv = np.zeros((len(sigmas)))
cos_sim_adv = np.zeros((len(sigmas)-1))
if args.attack_method=="pgd_iter":
    acc_before = np.zeros((10))
    acc_after = np.zeros((10))
    acc_purify = np.zeros((10))
    l2_before = np.zeros((10))
    l2_after = np.zeros((10))
    l2_purify = np.zeros((10))

# Checking AUROC and attack power
scorelist_nat = np.zeros((0))
scorelist_adv = np.zeros((0))
scorelist_purnat = np.zeros((0))
scorelist_puradv = np.zeros((0))
alpha_nat = np.zeros((0))
alpha_adv = np.zeros((0))
alpha_purnat = np.zeros((0))
alpha_puradv = np.zeros((0))

## Import networks
network = eval(args.network)().to('cuda')
network = torch.nn.DataParallel(network)
if not if_cifar: # MNIST or FashionMNIST
    states_att = torch.load(os.path.join('adv_training/run/logs', args.att_log, 'checkpoint.pth'), map_location='cuda')
    optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=0., betas=(0.9, 0.999), amsgrad=False)
    network.load_state_dict(states_att[0])
else: # CIFAR10 setting, trained by WideResNet
    states_att = torch.load(os.path.join('adv_training/run/logs', args.att_log, 'checkpoint.t7'), map_location='cuda') # Temporary t7 setting
    optimizer = optim.SGD(network.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    network = states_att['net'].to('cuda')
# Run adversarial purification via EBMs
with open(os.path.join('ncsn/run/logs', args.ebm_log,  'config.yml'), 'r') as f:
    config_ebm = yaml.load(f, Loader=yaml.Loader)
network_ebm = RefineNetDilated(config_ebm).to('cuda')
network_ebm = torch.nn.DataParallel(network_ebm)
states_ebm = torch.load(os.path.join('ncsn/run/logs', args.ebm_log, 'checkpoint.pth'), map_location='cuda')
network_ebm.load_state_dict(states_ebm[0])

# Adversarial purification
for i, (x,y) in enumerate(testLoader):
    if i<n_iters: # Run for (n_iters*100)
        # Get original data
        if args.dataset=='CIFAR10C':
            x/=255.
        y = y.long()
        x = x.to('cuda')
        y = y.to('cuda')
        x = x/256.*255. + torch.ones_like(x)/512. # {0/255,1/255,...,1} to {1/512,3/512,...,511/512} constellation
        if args.TSNE: # Use one data for t-SNE
            x = x[0:1]
            y = y[0:1]
        if args.random_start:
            delta = (torch.rand_like(x)-0.5)*2.
            xprime = x + delta*args.ptb/256.
        else:
            xprime = x

        network_ebm.eval()

        ### t-SNE deployment for one exemplary data (works well only for CIFAR-10)
        # Expected outputs
        # tsne_images.png: Voronoi diagram of data t-SNE (red: False, blue: True)
        # tsne_images_line.png: Voronoi diagram of data t-SNE, including purification procedure
        # tsne_features.png: Voronoi diagram of feature t-SNE (red: False, blue: True)
        # tsne_features_line.png: Voronoi diagram of feature t-SNE, including purification procedure
        if args.TSNE: # t-SNE deployment at a single data
            tsne_ptb = 8.
            white_ptb = 8.
            white_step_lr = 3.0e-4
            white_n_stages = 50 # number of purification stages at pgd_white and bim_white
            sigmas = np.exp(np.linspace(np.log(start_sigma), np.log(end_sigma), n_stages))
            white_sigmas = np.exp(np.linspace(np.log(start_sigma), np.log(end_sigma), white_n_stages))
            # Import all parameters by hand in TSNE cases
            xadv_pgd = pgd(x, xprime, y, network, optimizer, ptb=tsne_ptb)
            xadv_bim = bim(x, xprime, y, network, optimizer, ptb=tsne_ptb)
            xadv_bpda = bpda(x, y, optimizer, network_ebm, network, sigmas, ptb=tsne_ptb)
            network_ebm.train()
            xadv_pgdwhite = pgd_white(x, y, optimizer, network_ebm, network, white_sigmas, ptb=white_ptb)

            x_ebm = proj_scorenet(x, network_ebm, sigmas)
            x_ebm_pgd = proj_scorenet(xadv_pgd, network_ebm, sigmas)
            x_ebm_bim = proj_scorenet(xadv_bim, network_ebm, sigmas)
            x_ebm_bpda = proj_scorenet(xadv_bpda, network_ebm, sigmas)
            x_ebm_pgdwhiteall = proj_scorenet(xadv_pgdwhite, network_ebm, white_sigmas)
            x_ebm_pgdwhite = []
            for k in range(len(x_ebm_pgdwhiteall)):
                if k%5==0:
                    x_ebm_pgdwhite.append(x_ebm_pgdwhiteall[k])
            if if_cifar:
                z_ebm = network.latent(transform_cifar(torch.cat(x_ebm)).to('cuda')).detach().cpu().numpy()
                z_ebm_pgd = network.latent(transform_cifar(torch.cat(x_ebm_pgd)).to('cuda')).detach().cpu().numpy()
                z_ebm_bim = network.latent(transform_cifar(torch.cat(x_ebm_bim)).to('cuda')).detach().cpu().numpy()
                z_ebm_bpda = network.latent(transform_cifar(torch.cat(x_ebm_bpda)).to('cuda')).detach().cpu().numpy()
                z_ebm_pgdwhite = network.latent(transform_cifar(torch.cat(x_ebm_pgdwhite)).to('cuda')).detach().cpu().numpy()
                z_ebm_all = np.concatenate((z_ebm, z_ebm_pgd, z_ebm_bim, z_ebm_bpda, z_ebm_pgdwhite), axis=0)
    
                y_ebm = network(transform_cifar(torch.cat(x_ebm)).to('cuda')).detach().cpu().numpy()
                y_ebm_pgd = network(transform_cifar(torch.cat(x_ebm_pgd)).to('cuda')).detach().cpu().numpy()
                y_ebm_bim = network(transform_cifar(torch.cat(x_ebm_bim)).to('cuda')).detach().cpu().numpy()
                y_ebm_bpda = network(transform_cifar(torch.cat(x_ebm_bpda)).to('cuda')).detach().cpu().numpy()
                y_ebm_pgdwhite = network(transform_cifar(torch.cat(x_ebm_pgdwhite)).to('cuda')).detach().cpu().numpy()
                y_ebm_all = np.concatenate((y_ebm, y_ebm_pgd, y_ebm_bim, y_ebm_bpda, y_ebm_pgdwhite), axis=0)
            else:
                z_ebm = network.module.latent(torch.cat(x_ebm).to('cuda')).detach().cpu().numpy()
                z_ebm_pgd = network.module.latent(torch.cat(x_ebm_pgd).to('cuda')).detach().cpu().numpy()
                z_ebm_bim = network.module.latent(torch.cat(x_ebm_bim).to('cuda')).detach().cpu().numpy()
                z_ebm_bpda = network.module.latent(torch.cat(x_ebm_bpda).to('cuda')).detach().cpu().numpy()
                z_ebm_pgdwhite = network.module.latent(torch.cat(x_ebm_pgdwhite).to('cuda')).detach().cpu().numpy()
                z_ebm_all = np.concatenate((z_ebm, z_ebm_pgd, z_ebm_bim, z_ebm_bpda, z_ebm_pgdwhite), axis=0)
    
                y_ebm = network(torch.cat(x_ebm).to('cuda')).detach().cpu().numpy()
                y_ebm_pgd = network(torch.cat(x_ebm_pgd).to('cuda')).detach().cpu().numpy()
                y_ebm_bim = network(torch.cat(x_ebm_bim).to('cuda')).detach().cpu().numpy()
                y_ebm_bpda = network(torch.cat(x_ebm_bpda).to('cuda')).detach().cpu().numpy()
                y_ebm_pgdwhite = network(torch.cat(x_ebm_pgdwhite).to('cuda')).detach().cpu().numpy()
                y_ebm_all = np.concatenate((y_ebm, y_ebm_pgd, y_ebm_bim, y_ebm_bpda, y_ebm_pgdwhite), axis=0)

            # Predict
            y_argmax = np.argmax(y_ebm_all, axis=1)
            y_predict = np.equal(y_argmax, y.cpu().numpy().astype('int')*np.ones_like(y_argmax))
            y_o = network(x).detach().cpu().numpy()
            x_ebm = torch.cat(x_ebm).numpy().reshape((n_stages, -1))
            x_ebm_pgd = torch.cat(x_ebm_pgd).numpy().reshape((n_stages, -1))
            x_ebm_bim = torch.cat(x_ebm_bim).numpy().reshape((n_stages, -1))
            x_ebm_bpda = torch.cat(x_ebm_bpda).numpy().reshape((n_stages, -1))
            x_ebm_pgdwhite = torch.cat(x_ebm_pgdwhite).numpy().reshape((len(x_ebm_pgdwhite), -1))
            x_ebm_all = np.concatenate((x_ebm, x_ebm_pgd, x_ebm_bim, x_ebm_bpda, x_ebm_pgdwhite), axis=0)

            tsne = TSNE(n_components=2)
            xall_tsne = tsne.fit_transform(x_ebm_all)
            zall_tsne = tsne.fit_transform(z_ebm_all)
            x_ebm_all_bdry = np.append(xall_tsne, [[9999,9999],[9999,-9999],[-9999,9999],[-9999,-9999]], axis=0)
            z_ebm_all_bdry = np.append(zall_tsne, [[9999,9999],[9999,-9999],[-9999,9999],[-9999,-9999]], axis=0)
            voronoi_x = Voronoi(x_ebm_all_bdry)
            voronoi_z = Voronoi(z_ebm_all_bdry)
            xmax = np.amax(xall_tsne, axis=0)
            xmin = np.amin(xall_tsne, axis=0)
            zmax = np.amax(zall_tsne, axis=0)
            zmin = np.amin(zall_tsne, axis=0)

            # Visualize t-SNE
            plt.figure(figsize=(6, 6))
            colors = 'k', 'r', 'b', 'g', 'c', 'k'
            label_list = ['natural', 'PGD', 'BIM', 'BPDA', 'PGD-WHITE']
            y = np.concatenate((np.zeros(n_stages), np.ones(n_stages), 2*np.ones(n_stages), 3*np.ones(n_stages), 4*np.ones(len(x_ebm_pgdwhite))))
            target_ids = range(5)
            for k, c, label in zip(target_ids, colors, label_list):
                plt.scatter(xall_tsne[y==k, 0][0:1], xall_tsne[y==k, 1][0:1], color=c, marker='x')
                plt.scatter(xall_tsne[y==k, 0][1:-1], xall_tsne[y==k, 1][1:-1], color=c, label=label)
                plt.scatter(xall_tsne[y==k, 0][-1], xall_tsne[y==k, 1][-1], color=c, marker='<')
            for k in range(len(voronoi_x.point_region)):
                points = voronoi_x.point_region[k]
                if not -1 in voronoi_x.regions[points]:
                    polygon = [voronoi_x.vertices[j] for j in voronoi_x.regions[points]]
                    if y_predict[k]==True:
                        plt.fill(*zip(*polygon), "b", alpha=0.2)
                    else:
                        plt.fill(*zip(*polygon), "r", alpha=0.2)
            plt.ylim(xmin[1]-1, xmax[1]+1)
            plt.xlim(xmin[0]-1, xmax[0]+1)
            plt.legend()
            plt.savefig(os.path.join("log_images", args.image_folder, "tsne_images.png"), dpi=800)
            plt.close()

            plt.figure(figsize=(6, 6))
            for k, c in zip(target_ids, colors):
                plt.scatter(xall_tsne[y==k, 0][0:1], xall_tsne[y==k, 1][0:1], color=c, marker='x')
                plt.scatter(xall_tsne[y==k, 0][1:-1], xall_tsne[y==k, 1][1:-1], color=c, label=label)
                plt.scatter(xall_tsne[y==k, 0][-1], xall_tsne[y==k, 1][-1], color=c, marker='<')
                if k != 4:
                    n = n_stages
                else:
                    n = len(x_ebm_pgdwhite)
                for j in range(n-1):
                    plt.plot([xall_tsne[y==k,0][j], xall_tsne[y==k,0][j+1]], [xall_tsne[y==k,1][j], xall_tsne[y==k,1][j+1]], color=c)
            for k in range(len(voronoi_x.point_region)):
                points = voronoi_x.point_region[k]
                if not -1 in voronoi_x.regions[points]:
                    polygon = [voronoi_x.vertices[j] for j in voronoi_x.regions[points]]
                    if y_predict[k]==True:
                        plt.fill(*zip(*polygon), "b", alpha=0.2)
                    else:
                        plt.fill(*zip(*polygon), "r", alpha=0.2)
            plt.ylim(xmin[1]-1, xmax[1]+1)
            plt.xlim(xmin[0]-1, xmax[0]+1)
            plt.savefig(os.path.join("log_images", args.image_folder, "tsne_images_line.png"), dpi=800)
            plt.close()

            # Visualize t-SNE
            plt.figure(figsize=(10, 10))
            colors = 'k', 'r', 'b', 'g', 'c', 'k'
            label_list = ['natural', 'PGD', 'BIM', 'BPDA', 'PGD-WHITE']
            y = np.concatenate((np.zeros(n_stages), np.ones(n_stages), 2*np.ones(n_stages), 3*np.ones(n_stages), 4*np.ones(len(x_ebm_pgdwhite))))
            target_ids = range(5)
            for k, c, label in zip(target_ids, colors, label_list):
                plt.scatter(zall_tsne[y==k, 0][0:1], zall_tsne[y==k, 1][0:1], color=c, marker='x')
                plt.scatter(zall_tsne[y==k, 0][1:-1], zall_tsne[y==k, 1][1:-1], color=c, label=label)
                plt.scatter(zall_tsne[y==k, 0][-1], zall_tsne[y==k, 1][-1], color=c, marker='<')
            for k in range(len(voronoi_z.point_region)):
                points = voronoi_z.point_region[k]
                if not -1 in voronoi_z.regions[points]:
                    polygon = [voronoi_z.vertices[j] for j in voronoi_z.regions[points]]
                    if y_predict[k]==True:
                        plt.fill(*zip(*polygon), "b", alpha=0.2)
                    else:
                        plt.fill(*zip(*polygon), "r", alpha=0.2)
            plt.ylim(zmin[1]-1, zmax[1]+1)
            plt.xlim(zmin[0]-1, zmax[0]+1)
            plt.legend()
            plt.savefig(os.path.join("log_images", args.image_folder, "tsne_features.png"), dpi=800)
            plt.close()

            plt.figure(figsize=(10, 10))
            for k, c in zip(target_ids, colors):
                plt.scatter(zall_tsne[y==k, 0][0:1], zall_tsne[y==k, 1][0:1], color=c, marker='x')
                plt.scatter(zall_tsne[y==k, 0][1:-1], zall_tsne[y==k, 1][1:-1], color=c, label=label)
                plt.scatter(zall_tsne[y==k, 0][-1], zall_tsne[y==k, 1][-1], color=c, marker='<')
                if k != 4:
                    n = n_stages
                else:
                    n = len(x_ebm_pgdwhite)
                for j in range(n-1):
                    plt.plot([zall_tsne[y==k,0][j], zall_tsne[y==k,0][j+1]], [zall_tsne[y==k,1][j], zall_tsne[y==k,1][j+1]], color=c)
            for k in range(len(voronoi_z.point_region)):
                points = voronoi_z.point_region[k]
                if not -1 in voronoi_z.regions[points]:
                    polygon = [voronoi_z.vertices[j] for j in voronoi_z.regions[points]]
                    if y_predict[k]==True:
                        plt.fill(*zip(*polygon), "b", alpha=0.2)
                    else:
                        plt.fill(*zip(*polygon), "r", alpha=0.2)
            plt.ylim(zmin[1]-1, zmax[1]+1)
            plt.xlim(zmin[0]-1, zmax[0]+1)
            plt.savefig(os.path.join("log_images", args.image_folder, "tsne_features_line.png"), dpi=800)
            plt.close()

            sys.exit("Finished generating decision boundary")

        ### t-SNE description with 10 classes datasets (works well only for CIFAR-10)
        # Expected outputs
        # tsne_classes.png: t-SNE diagram of feature t-SNE of 100 data, including purification procedure of 3 exemplary data
        if args.TSNE_CLASS:
            tsne_ptb = 8.
            white_ptb = 8.
            white_step_lr = 3.0e-4
            white_n_stages = 50
            sigmas = np.exp(np.linspace(np.log(start_sigma), np.log(end_sigma), n_stages))
            white_sigmas = np.exp(np.linspace(np.log(start_sigma), np.log(end_sigma), white_n_stages))
            # Import all parameters by hand in TSNE cases
            xadv_bpda = bpda(x, y, optimizer, network_ebm, network, sigmas, ptb=tsne_ptb)
            network_ebm.train()
            xadv_pgdwhite = pgd_white(x, y, optimizer, network_ebm, network, white_sigmas, ptb=white_ptb)

            x_ebm = proj_scorenet(x, network_ebm, sigmas)
            x_ebm_bpda = proj_scorenet(xadv_bpda, network_ebm, sigmas)
            x_ebm_pgdwhiteall = proj_scorenet(xadv_pgdwhite, network_ebm, white_sigmas)
            x_ebm_pgdwhite = []

            z_ex = []
            for k in range(100):
                if len(z_ex)==3:
                    break
                if y[k]==0:
                    z_ex.append(k)
            for k in range(len(x_ebm_pgdwhiteall)):
                if k%1==0:
                    x_ebm_pgdwhite.append(x_ebm_pgdwhiteall[k])
            arr_bpda = []
            arr_pgdwhite = []
            for k in range(3): # Choose 3 data to be tracked
                arr_bpdai = []
                arr_pgdwhitei = []
                for l in range(10):
                    arr_bpdai.append(z_ex[k]+l*100) # Track every iteration
                    arr_pgdwhitei.append(z_ex[k]+l*500) # Track every 5 iteration
                arr_bpda.append(arr_bpdai)
                arr_pgdwhite.append(arr_pgdwhitei)
            if if_cifar:
                z_ebm = network.latent(transform_cifar(x_ebm[0]).to('cuda')).detach().cpu().numpy()
                z_ebm_bpda_0 = network.latent(transform_cifar(x_ebm_bpda[0]).to('cuda')).detach().cpu().numpy()
                z_ebm_pgdwhite_0 = network.latent(transform_cifar(x_ebm_pgdwhite[0]).to('cuda')).detach().cpu().numpy()
                z_ebm_bpda_f = network.latent(transform_cifar(x_ebm_bpda[-1]).to('cuda')).detach().cpu().numpy()
                z_ebm_pgdwhite_f = network.latent(transform_cifar(x_ebm_pgdwhite[-1]).to('cuda')).detach().cpu().numpy()
                z_ebm_all = np.concatenate((z_ebm, z_ebm_bpda_0, z_ebm_pgdwhite_0, z_ebm_bpda_f, z_ebm_pgdwhite_f), axis=0)
                #
                z_ebm_bpda1 = network.latent(transform_cifar(torch.cat(x_ebm_bpda)[arr_bpda[0]]).to('cuda')).detach().cpu().numpy()
                z_ebm_pgdwhite1 = network.latent(transform_cifar(torch.cat(x_ebm_pgdwhite)[arr_pgdwhite[0]]).to('cuda')).detach().cpu().numpy()
                z_ebm_bpda2 = network.latent(transform_cifar(torch.cat(x_ebm_bpda)[arr_bpda[1]]).to('cuda')).detach().cpu().numpy()
                z_ebm_pgdwhite2 = network.latent(transform_cifar(torch.cat(x_ebm_pgdwhite)[arr_pgdwhite[1]]).to('cuda')).detach().cpu().numpy()
                z_ebm_bpda3 = network.latent(transform_cifar(torch.cat(x_ebm_bpda)[arr_bpda[2]]).to('cuda')).detach().cpu().numpy()
                z_ebm_pgdwhite3 = network.latent(transform_cifar(torch.cat(x_ebm_pgdwhite)[arr_pgdwhite[2]]).to('cuda')).detach().cpu().numpy()
                
                y_ebm = network(transform_cifar(x_ebm[0]).to('cuda')).detach().cpu().numpy()
                y_ebm_bpda_0 = network(transform_cifar(x_ebm_bpda[0]).to('cuda')).detach().cpu().numpy()
                y_ebm_pgdwhite_0 = network(transform_cifar(x_ebm_pgdwhite[0]).to('cuda')).detach().cpu().numpy()
                y_ebm_bpda_f = network(transform_cifar(x_ebm_bpda[-1]).to('cuda')).detach().cpu().numpy()
                y_ebm_pgdwhite_f = network(transform_cifar(x_ebm_pgdwhite[-1]).to('cuda')).detach().cpu().numpy()
                y_ebm_all = np.concatenate((y_ebm, y_ebm_bpda_0, y_ebm_pgdwhite_0, y_ebm_bpda_f, y_ebm_pgdwhite_f), axis=0)
            else:
                z_ebm = network.module.latent(x_ebm[0].to('cuda')).detach().cpu().numpy()
                z_ebm_bpda_0 = network.module.latent(x_ebm_bpda[0].to('cuda')).detach().cpu().numpy()
                z_ebm_pgdwhite_0 = network.module.latent(x_ebm_pgdwhite[0].to('cuda')).detach().cpu().numpy()
                z_ebm_bpda_f = network.module.latent(x_ebm_bpda[-1].to('cuda')).detach().cpu().numpy()
                z_ebm_pgdwhite_f = network.module.latent(x_ebm_pgdwhite[-1].to('cuda')).detach().cpu().numpy()
                z_ebm_all = np.concatenate((z_ebm, z_ebm_bpda_0, z_ebm_pgdwhite_0, z_ebm_bpda_f, z_ebm_pgdwhite_f), axis=0)
                #
                z_ebm_bpda1 = network.module.latent(torch.cat(x_ebm_bpda)[arr_bpda[0]].to('cuda')).detach().cpu().numpy()
                z_ebm_pgdwhite1 = network.module.latent(torch.cat(x_ebm_pgdwhite)[arr_pgdwhite[0]].to('cuda')).detach().cpu().numpy()
                z_ebm_bpda2 = network.module.latent(torch.cat(x_ebm_bpda)[arr_bpda[1]].to('cuda')).detach().cpu().numpy()
                z_ebm_pgdwhite2 = network.module.latent(torch.cat(x_ebm_pgdwhite)[arr_pgdwhite[1]].to('cuda')).detach().cpu().numpy()
                z_ebm_bpda3 = network.module.latent(torch.cat(x_ebm_bpda)[arr_bpda[2]].to('cuda')).detach().cpu().numpy()
                z_ebm_pgdwhite3 = network.module.latent(torch.cat(x_ebm_pgdwhite)[arr_pgdwhite[2]].to('cuda')).detach().cpu().numpy()
                
                y_ebm = network(x_ebm[0].to('cuda')).detach().cpu().numpy()
                y_ebm_bpda_0 = network(x_ebm_bpda[0].to('cuda')).detach().cpu().numpy()
                y_ebm_pgdwhite_0 = network(x_ebm_pgdwhite[0].to('cuda')).detach().cpu().numpy()
                y_ebm_bpda_f = network(x_ebm_bpda[-1].to('cuda')).detach().cpu().numpy()
                y_ebm_pgdwhite_f = network(x_ebm_pgdwhite[-1].to('cuda')).detach().cpu().numpy()
                y_ebm_all = np.concatenate((y_ebm, y_ebm_bpda_0, y_ebm_pgdwhite_0, y_ebm_bpda_f, y_ebm_pgdwhite_f), axis=0)
            # Predict
            y_argmax = np.argmax(y_ebm_all, axis=1)
            y_predict = np.argmax(y_ebm, axis=1)
            y_cpu = y.cpu().numpy()

            # Generate t-SNE
            tsne = TSNE(n_components=2)
            z_ebm_selected = np.concatenate((z_ebm_all, z_ebm_bpda1, z_ebm_pgdwhite1, z_ebm_bpda2, z_ebm_pgdwhite2, z_ebm_bpda3, z_ebm_pgdwhite3))
            zall_tsne = tsne.fit_transform(z_ebm_selected)
            ztsne = zall_tsne[0:100]
            ztsne_bpda_0 = zall_tsne[100:200]
            ztsne_pgdwhite_0 = zall_tsne[200:300]
            ztsne_bpda_f = zall_tsne[300:400]
            ztsne_pgdwhite_f = zall_tsne[400:500]

            zmax = np.amax(zall_tsne, axis=0)
            zmin = np.amin(zall_tsne, axis=0)
            # Visualize t-SNE
            plt.figure(figsize=(10, 10))
            colors = 'k', 'r', 'b', 'c', 'm', 'y', 'g', 'purple', 'gray', 'darkkhaki'
            if args.dataset=='MNIST':
                label_list = ['0','1','2','3','4','5','6','7','8','9']
            elif args.dataset=='CIFAR10' or args.dataset=='CIFAR10C':
                label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            target_ids = range(10)
            ss = 0
            for k, c, label in zip(target_ids, colors, label_list):
                y_predict1 = y_predict[0:100]
                plt.scatter(ztsne[y_cpu==k, 0], ztsne[y_cpu==k, 1], color=c, label=label)
                if c!=0:
                    continue
                plt.scatter(zall_tsne[500:560, 0], zall_tsne[500:560, 1], color=c, marker='.')
            for k in range(3):
                plt.scatter(zall_tsne[500+k*20,0], zall_tsne[500+k*20,1], color='k', marker='x')
                plt.scatter(zall_tsne[500+k*20+10,0], zall_tsne[500+k*20+10,1], color='k', marker='s')
                plt.scatter(zall_tsne[500+k*20+9,0], zall_tsne[500+k*20+9, 1], color='k', marker='+')
                plt.scatter(zall_tsne[500+k*20+19,0], zall_tsne[500+k*20+19, 1], color='k', marker='*')
                plt.plot([zall_tsne[z_ex[k], 0], zall_tsne[500+k*20, 0]], [zall_tsne[z_ex[k], 1], zall_tsne[500+k*20, 1]], color='k')
                plt.plot([zall_tsne[z_ex[k], 0], zall_tsne[500+k*20+10, 0]], [zall_tsne[z_ex[k], 1], zall_tsne[500+k*20+10, 1]], color='k')
                for l in range(9):
                    plt.plot([zall_tsne[500+k*20+l, 0], zall_tsne[500+k*20+l+1, 0]], [zall_tsne[500+k*20+l, 1], zall_tsne[500+k*20+l+1, 1]], color='k')
                    plt.plot([zall_tsne[500+k*20+l+10, 0], zall_tsne[500+k*20+l+11, 0]], [zall_tsne[500+k*20+l+10, 1], zall_tsne[500+k*20+l+11, 1]], color='k')
            plt.ylim(zmin[1]-1, zmax[1]+1)
            plt.xlim(zmin[0]-1, zmax[0]+1)
            plt.legend()
            plt.savefig(os.path.join("log_images", args.image_folder, "tsne_classes.png"), dpi=800)
            plt.close()

            sys.exit("Finished t-SNE")

        ### Generate adversarial data
        # Expected output in common: x_adv (final adversarial data)
        # Expected output at pgd_iter (best attack discovered): x_adv_list_before (x_adv before each purification), x_adv_list_after (x_adv after each purification)
        if not args.TSNE and not args.TSNE_CLASS:
            if args.dataset=='CIFAR10C':
                x_adv = x.detach() # Data itself is adversarial
            elif args.attack_method=="fgsm":
                x_adv = bim(x, xprime, y, network, optimizer, alpha=args.ptb, ptb=args.ptb, iters=1)
            elif args.attack_method=="pgd":
                x_adv = pgd(x, xprime, y, network, optimizer, ptb=args.ptb)
            elif args.attack_method=="bim":
                x_adv = bim(x, xprime, y, network, optimizer, ptb=args.ptb)
            elif args.attack_method=="bpda":
                x_adv = bpda(x, y, optimizer, network_ebm, network, attack_sigmas, ptb=args.ptb)
            elif args.attack_method=="pgd_white":
                network_ebm.train()
                x_adv = pgd_white(x, y, optimizer, network_ebm, network, attack_sigmas, ptb=args.ptb)
            elif args.attack_method=="bim_white":
                network_ebm.train()
                x_adv = bim_white(x, y, optimizer, network_ebm, network, attack_sigmas, ptb=args.ptb)
            elif args.attack_method=="pgd_iter":
                x_adv_list_before, x_adv_list_after, x_adv = pgd_iter(x, y, optimizer, network_ebm, network, attack_sigmas, ptb=args.ptb)
            elif args.attack_method=="pgd_forward_full":
                x_adv = pgd_forward_full(x, y, optimizer, network_ebm, network, args.e1, args.e2, attack_sigmas, ptb=args.ptb)
            elif args.attack_method=="pgd_forward_one":
                x_adv = pgd_forward_one(x, y, optimizer, network_ebm, network, args.e1, args.e2, attack_sigmas, ptb=args.ptb)
            elif args.attack_method=="spsa":
                x_adv = spsa(x, y, network_ebm, network, attack_sigmas, ptb=args.ptb)
            elif args.attack_method=="nes":
                x_adv = spsa(x, y, network_ebm, network, attack_sigmas, ptb=args.ptb)
            elif args.attack_method=="nattack":
                x_adv = nattack(x, y, network_ebm, network, attack_sigmas, ptb=args.ptb)
            elif args.attack_method=="sn_disable":
                network_ebm.train()
                x_adv = sn_disable(x, y, optimizer, network_ebm, attack_sigmas, ptb=args.ptb)
            elif args.attack_method=="sn_disable_mixed":
                network_ebm.train()
                x_adv, lossitem = sn_disable_mixed(x, y, optimizer, network_ebm, network, attack_sigmas, args.e1, args.e2, ptb=args.ptb)

        ### Purify data
        # Expected output in common: x_ebm, x_adv_ebm
        # Optional output: x_ebm_grad, x_ebm_grad_std, if save_score_norm = True
        # Optional output: x_step_sizes for adaptive step sizes
        network_ebm.eval()
        x_adv_p = torch.clamp(x_adv + torch.randn_like(x_adv)*(args.init_noise/255.), min=0.0, max=1.0)
        x_p = torch.clamp(x + torch.randn_like(x)*(args.init_noise/255.), min=0.0, max=1.0)
        if args.purify_method=="anneal_langevin":
            x_ebm = anneal_Langevin_dynamics(x_p, network_ebm, sigmas)
            x_adv_ebm = anneal_Langevin_dynamics(x_adv_p, network_ebm, sigmas)
        elif args.purify_method=="projection":
            x_ebm, x_ebm_grad, x_ebm_grad_std = proj_scorenet(x_p, network_ebm, sigmas, save_score_norm=True)
            x_adv_ebm, x_adv_ebm_grad, x_adv_ebm_grad_std = proj_scorenet(x_adv_p, network_ebm, sigmas, save_score_norm=True)
        elif args.purify_method=="rand":
            x_ebm = anneal_noise(x_p, network_ebm, sigmas)
            x_adv_ebm = anneal_noise(x_adv_p, network_ebm, sigmas)
        elif args.purify_method=="adaptive":
            x_ebm, x_ebm_grad, x_ebm_grad_std, x_step_sizes = adaptive_proj_scorenet(x_p, network_ebm, save_score_norm=True, alpha=args.alpha)
            x_adv_ebm, x_adv_ebm_grad, x_adv_ebm_grad_std, x_adv_step_sizes = adaptive_proj_scorenet(x_adv_p, network_ebm, save_score_norm=True, alpha=args.alpha)

        ### Given x and x_adv, measure alpha and score norm of them
        nat_scores_t, nat_alpha_t, nat_flags_t = attack_qual_check(x, network_ebm, data_type=0)
        adv_scores_t, adv_alpha_t, adv_flags_t = attack_qual_check(x_adv, network_ebm, data_type=1)
        pur_scores_t1, pur_alpha_t1, pur_flags_t1 = attack_qual_check(x_ebm[1].to('cuda'), network_ebm, data_type=2)
        pur_scores_t2, pur_alpha_t2, pur_flags_t2 = attack_qual_check(x_adv_ebm[1].to('cuda'), network_ebm, data_type=2)

        scorelist_nat = np.append(scorelist_nat, nat_scores_t)
        alpha_nat = np.append(alpha_nat, np.log(nat_alpha_t))
        scorelist_adv = np.append(scorelist_adv, adv_scores_t)
        alpha_adv = np.append(alpha_adv, np.log(adv_alpha_t))
        scorelist_purnat = np.append(scorelist_purnat, pur_scores_t1)
        alpha_purnat = np.append(alpha_purnat, np.log(pur_alpha_t1))
        scorelist_puradv = np.append(scorelist_puradv, pur_scores_t2)
        alpha_puradv = np.append(alpha_puradv, np.log(pur_alpha_t2))
        
        scoremin = np.amin([np.amin(scorelist_nat), np.amin(scorelist_adv), np.amin(scorelist_purnat), np.amin(scorelist_puradv)])
        scoremax = np.amax([np.amax(scorelist_nat), np.amax(scorelist_adv), np.amax(scorelist_purnat), np.amax(scorelist_puradv)])
        alphamin = np.amin([np.amin(alpha_nat), np.amin(alpha_adv), np.amin(alpha_purnat), np.amin(alpha_puradv)])
        alphamax = np.amax([np.amax(alpha_nat), np.amax(alpha_adv), np.amax(alpha_purnat), np.amax(alpha_puradv)])
        scorebins = np.linspace(scoremin, scoremax, 100)
        alphabins = np.linspace(alphamin, alphamax, 100)
        fig, ax = plt.subplots()
        hist_score_nat, bin_edges1, _ = plt.hist(scorelist_nat, scorebins, alpha=0.5, label='nat', density=True)
        hist_score_adv, bin_edges2, _ = plt.hist(scorelist_adv, scorebins, alpha=0.5, label='adv', density=True)
        hist_score_purnat, bin_edges5, _ = plt.hist(scorelist_purnat, scorebins, alpha=0.5, label='pur_n', density=True)
        hist_score_puradv, bin_edges7, _ = plt.hist(scorelist_puradv, scorebins, alpha=0.5, label='pur_a', density=True)
        plt.legend(loc='upper right')    
        plt.savefig(os.path.join("log_images", args.image_folder,"hist_score.png")) 
        plt.close()
        fig, ax = plt.subplots()
        hist_alpha_nat, bin_edges3, _ = plt.hist(alpha_nat, alphabins, alpha=0.5, label='nat', density=True)
        hist_alpha_adv, bin_edges4, _ = plt.hist(alpha_adv, alphabins, alpha=0.5, label='adv', density=True)
        hist_alpha_purnat, bin_edges6, _ = plt.hist(alpha_purnat, alphabins, alpha=0.5, label='pur_n', density=True)
        hist_alpha_puradv, bin_edges8, _ = plt.hist(alpha_puradv, alphabins, alpha=0.5, label='pur_a', density=True)
        plt.legend(loc='upper right')    
        plt.savefig(os.path.join("log_images", args.image_folder,"hist_alpha.png")) 
        plt.close()
        hist_score_nat *= np.diff(bin_edges1)
        hist_score_adv *= np.diff(bin_edges2)
        hist_score_purnat *= np.diff(bin_edges5)
        hist_score_puradv *= np.diff(bin_edges7)
        hist_alpha_nat *= np.diff(bin_edges3)
        hist_alpha_adv *= np.diff(bin_edges4)
        hist_alpha_purnat *= np.diff(bin_edges6)
        hist_alpha_puradv *= np.diff(bin_edges8)
        for j in range(len(hist_score_nat)):
            if j!=0:
                hist_score_nat[j] += hist_score_nat[j-1]
        for j in range(len(hist_score_adv)):
            if j!=0:
                hist_score_adv[j] += hist_score_adv[j-1]
        for j in range(len(hist_alpha_nat)):
            if j!=0:
                hist_alpha_nat[j] += hist_alpha_nat[j-1]
        for j in range(len(hist_alpha_adv)):
            if j!=0:
                hist_alpha_adv[j] += hist_alpha_adv[j-1]

        # calculate AUROC
        auroc_score = 0.0
        auroc_alpha = 0.0
        for j in range(len(hist_score_nat)):
            if j!=0:
                auroc_score += hist_score_nat[j-1]*(hist_score_adv[j]-hist_score_adv[j-1])
        for j in range(len(hist_alpha_nat)):
            if j!=0:
                auroc_alpha += hist_alpha_nat[j-1]*(hist_alpha_adv[j]-hist_alpha_adv[j-1])
        print("{} {} score auroc {:.3f} alpha auroc {:.3f}".format(i+1, args.image_folder,auroc_score, auroc_alpha))
        fig, ax= plt.subplots()
        plt.plot(np.append(0.,hist_score_adv), np.append(0.,hist_score_nat), lw=2, color='r')
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.savefig(os.path.join("log_images", args.image_folder,"auroc_score.png")) 
        plt.close()

        fig, ax= plt.subplots()
        plt.plot(np.append(0., hist_alpha_adv), np.append(0., hist_alpha_nat), lw=2, color='r')
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.savefig(os.path.join("log_images", args.image_folder,"auroc_alpha.png")) 
        plt.close()

        ### Details in pgd_iter: Performance by iteration
        # Expected outputs
        # acc_iterative.png: accuracy by iteration
        # l2_iterative.png: accuracy by iteration
        if args.attack_method=="pgd_iter":
            for j in range(10):
                x_adv_purify = proj_scorenet(x_adv_list_before[j], network_ebm, sigmas)[-1].cuda()

                l2_before[j] += torch.norm((x-x_adv_list_before[j]).to('cuda').view(x.shape[0],-1), p=2, dim=1).mean().item()
                l2_after[j] += torch.norm((x-x_adv_list_after[j]).to('cuda').view(x.shape[0],-1), p=2, dim=1).mean().item()
                l2_purify[j] += torch.norm((x-x_adv_purify).to('cuda').view(x.shape[0],-1), p=2, dim=1).mean().item()
                
                if if_cifar:
                    x_adv_list_before[j] = transform_cifar(x_adv_list_before[j]).to('cuda')
                    x_adv_list_after[j] = transform_cifar(x_adv_list_after[j]).to('cuda')
                    x_adv_purify = transform_cifar(x_adv_purify).to('cuda')
                
                with torch.no_grad():
                    yhat_before = network(x_adv_list_before[j])
                    yhat_after = network(x_adv_list_after[j])
                    yhat_purify = network(x_adv_purify)

                _, predicted_before = yhat_before.max(1)
                _, predicted_after = yhat_after.max(1)
                _, predicted_purify = yhat_purify.max(1)
                acc_before[j] += predicted_before.eq(y).sum().item()
                acc_after[j] += predicted_after.eq(y).sum().item()
                acc_purify[j] += predicted_purify.eq(y).sum().item()

            acc_before_plot = acc_before / ((i+1)*grid_size**2)
            acc_after_plot = acc_after / ((i+1)*grid_size**2)
            acc_purify_plot = acc_purify / ((i+1)*grid_size**2)
            l2_before_plot = l2_before / ((i+1)*grid_size**2)
            l2_after_plot = l2_after / ((i+1)*grid_size**2)
            l2_purify_plot = l2_purify / ((i+1)*grid_size**2)

            # Plot accuracy
            fig, ax = plt.subplots()
            epoch_ls = np.linspace(0, 9, 10)
            ax.plot(epoch_ls, acc_before_plot*100., '-', color='r', label='Attacked')
            ax.plot(epoch_ls, acc_after_plot*100., '-', color='b', label='Purified (bounded)')
            ax.plot(epoch_ls, acc_purify_plot*100., '-', color='g', label='Purified (unbounded)')
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy')
            plt.savefig(os.path.join("log_images", args.image_folder,"acc_iterative.png"))
            plt.close(fig)

            # Plot l2
            fig, ax = plt.subplots()
            epoch_ls = np.linspace(0, 9, 10)
            ax.plot(epoch_ls, l2_before_plot, '-', color='r', label='Attacked')
            ax.plot(epoch_ls, l2_after_plot, '-', color='b', label='Purified (bounded)')
            ax.plot(epoch_ls, l2_purify_plot, '-', color='g', label='Purified (unbounded)')
            plt.xlabel('Iteration')
            plt.ylabel('l2 distance')
            plt.savefig(os.path.join("log_images", args.image_folder,"l2_iterative.png"))
            plt.close(fig)

            # Plot accuracy (except attacked ~ 0%)
            fig, ax = plt.subplots()
            epoch_ls = np.linspace(0, 9, 10)
            ax.plot(epoch_ls, acc_after_plot*100., '-', color='b', label='Purified (bounded)')
            ax.plot(epoch_ls, acc_purify_plot*100., '-', color='g', label='Purified (unbounded)')
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy')
            plt.savefig(os.path.join("log_images", args.image_folder,"acc_iterative_purified.png"))
            plt.close(fig)

            # Plot l2 (except attacked ~ 0%)
            fig, ax = plt.subplots()
            epoch_ls = np.linspace(0, 9, 10)
            ax.plot(epoch_ls, l2_after_plot, '-', color='b', label='Purified (bounded)')
            ax.plot(epoch_ls, l2_purify_plot, '-', color='g', label='Purified (unbounded)')
            plt.xlabel('Iteration')
            plt.ylabel('l2 distance')
            plt.savefig(os.path.join("log_images", args.image_folder,"l2_iterative_purified.png"))
            plt.close(fig)

            # Save logs
            str_acc_iter = ''
            str_l2_iter = ''
            for j in range(10):
                str_acc_iter += '{:.2f} '.format(acc_purify_plot[j]*100.)
                str_l2_iter += '{:.2f} '.format(l2_purify_plot[j]*100.)
            print('pgd_iter acc '+str_acc_iter)
            print('pgd_iter l2 '+str_l2_iter)

        with torch.no_grad():
            ### Get accuracy and l2 distance for natural and adversarial examples
            if if_cifar: # Transform if cifar
                x = transform_cifar(x).to('cuda')
                x_adv = transform_cifar(x_adv).to('cuda')
            yhat = network(x)
            yhat_adv = network(x_adv)
            _, predicted = yhat.max(1)
            _, predicted_adv = yhat_adv.max(1)
            if if_cifar:
                x = inv_transform_cifar(x).to('cuda')
                x_adv = inv_transform_cifar(x_adv).to('cuda')
            acc_x[0] += predicted.eq(y).sum().item()
            acc_xadv[0] += predicted_adv.eq(y).sum().item()
            l2[0] += 0
            l2_adv[0] += torch.norm((x-x_adv).to('cuda').view(x.shape[0],-1), p=2, dim=1).mean().item()
            l2_diff[0] += torch.norm((x-x_adv).to('cuda').view(x.shape[0],-1), p=2, dim=1).mean().item()

            # Ensemble accuracy metrics
            logit_x_sum = np.zeros([100, 10])
            logit_xadv_sum = np.zeros([100, 10])
            logit_x_onehot = []
            logit_xadv_onehot = []
            logit_x_onehot_sum = np.zeros([100, 10])
            logit_xadv_onehot_sum = np.zeros([100, 10])

            imgs = []
            imgs_adv = []

            for j in range(len(sigmas)):
                if if_cifar:
                    x_ebm[j] = transform_cifar(x_ebm[j]).to('cuda')
                    x_adv_ebm[j] = transform_cifar(x_adv_ebm[j]).to('cuda')
                yhat = network(x_ebm[j])
                yhat_adv = network(x_adv_ebm[j])
                _, predicted = yhat.max(1)
                _, predicted_adv = yhat_adv.max(1)
                if if_cifar:
                    x_ebm[j] = inv_transform_cifar(x_ebm[j])
                    x_adv_ebm[j] = inv_transform_cifar(x_adv_ebm[j])
                acc_x[j+1] += predicted.eq(y).sum().item()
                acc_xadv[j+1] += predicted_adv.eq(y).sum().item()
                l2[j+1] += torch.norm((x-x_ebm[j]).to('cuda').view(x.shape[0],-1), p=2, dim=1).mean().item()
                l2_adv[j+1] += torch.norm((x-x_adv_ebm[j]).to('cuda').view(x.shape[0],-1), p=2, dim=1).mean().item()
                l2_diff[j+1] += torch.norm((x_ebm[j]-x_adv_ebm[j]).to('cuda').view(x.shape[0],-1), p=2, dim=1).mean().item()

                # Save ensemble parameters
                logit_x_sum += yhat.detach().to('cpu').numpy()
                logit_xadv_sum += yhat_adv.detach().to('cpu').numpy()
                logit_x_onehot.append(predicted.detach().to('cpu').numpy())
                logit_xadv_onehot.append(predicted_adv.detach().to('cpu').numpy())

                # Measure score metrics for each iterations
                if j != len(sigmas)-1:
                    # Measure cosine similarity
                    cos = nn.CosineSimilarity(dim=1)
                    or1 = x_ebm_grad[j].view(100,-1)
                    or2 = x_ebm_grad[j+1].view(100,-1)
                    adv1 = x_adv_ebm_grad[j].view(100,-1)
                    adv2 = x_adv_ebm_grad[j+1].view(100,-1)
                    cos_sim[j] += cos(or1, or2).mean().item()
                    cos_sim_adv[j] += cos(adv1, adv2).mean().item()
                # measure score norm and stdev
                score_norm[j] += torch.norm(x_ebm_grad[j].view(100,-1), dim=1).mean().item()
                score_norm_adv[j] += torch.norm(x_adv_ebm_grad[j].view(100,-1), dim=1).mean().item()
                score_std[j] += torch.norm(x_ebm_grad_std[j].view(100,-1), dim=1).mean().item()
                score_std_adv[j] += torch.norm(x_adv_ebm_grad_std[j].view(100,-1), dim=1).mean().item()

                ### Deployment at hyperparameter tuning phase (adaptive projection)
                # Expected outputs
                # perturbed_x_score_{}.png: score norm at [x-0.05score(x), x+0.05score(x)]
                # x_to_x_adv_score_norm_{}.png: score norm at [1.5x-0.5x', -0.5x+1.5x']
                # cossim_x_xadv_{}.png: cosine similarity between score at the point and (score(x), score(xadv), (xadv-x))
                # xadv_move_{}.png: score norm at [x-0.5t*score(x), x+1.5t*score(x)], t: Optimal step size at Gaussian approximation
                # best_alpha_{}.png: Best alpha value for each iteration
                # best_alpha_score_norm_{}.png: Score norm at best alpha value for each iteration
                if args.HP_TUNING:
                    if j==0:
                        # Score norm near x
                        perturbed_x_list = []
                        for k in range(-50,51):
                            perturbed_x_list.append(x_ebm[0][0] + (0.001*k)*x_ebm_grad[0][0])
                        perturbed_x = torch.stack(perturbed_x_list, dim=0)
                        with torch.no_grad():
                            labels = torch.ones(101, device='cuda')*10
                            labels = labels.long()
                            perturbed_x_grad = network_ebm(perturbed_x, labels)
                        perturbed_x_norm = torch.norm(perturbed_x_grad.view(101,-1), dim=1)

                        fig, ax = plt.subplots()
                        epoch_ls = np.linspace(-50, 50, 101)
                        ax.plot(epoch_ls, perturbed_x_norm.to('cpu').numpy(), '-', color='r')
                        plt.xlabel('Perturbation')
                        plt.ylabel('score norm')
                        plt.savefig(os.path.join("log_images", args.image_folder, "perturbed_x_score_{}.png".format(i)))
                        plt.close(fig)

                    if j==0:
                        # Score norm between x and xadv
                        perturbed_x_list = []
                        for k in range(-50,151):
                            perturbed_x_list.append((1.-0.01*k)*x_ebm[0][0] + (0.01*k)*x_adv_ebm[0][0])
                        perturbed_x = torch.stack(perturbed_x_list, dim=0)
                        with torch.no_grad():
                            labels = torch.ones(201, device='cuda')*10
                            labels = labels.long()
                            perturbed_x_grad = network_ebm(perturbed_x, labels)
                        perturbed_x_norm = torch.norm(perturbed_x_grad.view(201,-1), dim=1)

                        fig, ax = plt.subplots()
                        epoch_ls = np.linspace(-0.5, 1.5, 201)
                        ax.plot(epoch_ls, perturbed_x_norm.to('cpu').numpy(), '-', color='r')
                        ax.plot(0.0, perturbed_x_norm[50].to('cpu').numpy(), 'o', color='r')
                        ax.plot(1.0, perturbed_x_norm[150].to('cpu').numpy(), 'x', color='r')
                        plt.xlabel('Zero x, One x_adv')
                        plt.ylabel('score norm')
                        plt.savefig(os.path.join("log_images", args.image_folder, "x_to_xadv_score_norm_{}.png".format(i)))
                        plt.close(fig)
                    
                    if j==0:
                        # Cosine similarity between score(x) and score(x+delta)
                        cos = nn.CosineSimilarity(dim=1)
                        cos_from_scorex = np.zeros((201))
                        cos_from_scorexadv = np.zeros((201))
                        cos_from_xadv_to_x = np.zeros((201))
                        for k in range(-50,151):
                            cos_from_scorex[k+50] = cos(perturbed_x_grad[50:51].view(1,-1), perturbed_x_grad[k+50:k+51].view(1,-1)).to('cpu').numpy()
                            cos_from_scorexadv[k+50] = cos(perturbed_x_grad[150:151].view(1,-1), perturbed_x_grad[k+50:k+51].view(1,-1)).to('cpu').numpy()
                            cos_from_xadv_to_x[k+50] = cos((x_adv_ebm[0][0:1]-x_ebm[0][0:1]).view(1,-1), perturbed_x_grad[k+50:k+51].view(1,-1)).to('cpu').numpy()
                        fig, ax = plt.subplots()
                        epoch_ls = np.linspace(-0.5, 1.5, 201)
                        ax.plot(epoch_ls, cos_from_scorex, '-', color='r', label='score(x))')
                        ax.plot(0.0, cos_from_scorex[50], 'o', color='r')
                        ax.plot(1.0, cos_from_scorex[150], 'x', color='r')
                        ax.plot(epoch_ls, cos_from_scorexadv, '-', color='b', label='score(xadv)')
                        ax.plot(0.0, cos_from_scorexadv[50], 'o', color='b')
                        ax.plot(1.0, cos_from_scorexadv[150], 'x', color='b')
                        ax.plot(epoch_ls, cos_from_xadv_to_x, '-', color='k', label='xadv-x')
                        ax.plot(0.0, cos_from_xadv_to_x[50], 'o', color='k')
                        ax.plot(1.0, cos_from_xadv_to_x[150], 'x', color='k')
                        plt.hlines(0.0, -0.5, 1.5, colors='k', alpha=0.2)
                        plt.xlabel('Zero x, One x_adv')
                        plt.ylabel('cosine similarity')
                        plt.legend()
                        plt.ylim(-1.1, 1.1)
                        plt.savefig(os.path.join("log_images", args.image_folder, "cossim_x_xadv_{}.png".format(i)))
                        plt.close(fig)

                    if j==0 and args.purify_method=="adaptive":
                        # Initial movement from xadv
                        next_xadv_list = []
                        for k in range(-50, 151):
                            next_xadv_list.append((1.-0.01*k/args.alpha)*x_adv_ebm[0][0].to('cuda')+(0.01*k/args.alpha)*x_adv_ebm[1][0].to('cuda'))
                        score_norm_xadvs = torch.stack(next_xadv_list, dim=0)
                        with torch.no_grad():
                            labels = torch.ones(201, device='cuda')*10
                            labels = labels.long()
                            next_xadv_grad = network_ebm(score_norm_xadvs, labels)
                        next_xadv_norm = torch.norm(next_xadv_grad.view(201,-1), dim=1).to('cpu').numpy()
                        
                        fig, ax = plt.subplots()
                        epoch_ls = np.linspace(-0.5, 1.5, 201)
                        ax.plot(epoch_ls, next_xadv_norm, '-', color='r')
                        plt.xlabel('Zero xadv, One alpha=1 next')
                        plt.ylabel('Score norm')
                        plt.savefig(os.path.join("log_images", args.image_folder, "xadv_move_{}.png".format(i)))
                        plt.close(fig)
                    
                    if j==0 and args.purify_method=="adaptive":
                        # Get figure for best alpha
                        best_alpha_pars = np.zeros((len(x_adv_ebm)-1)) # Best alpha from -0.5 to 1.5
                        best_alpha_score_norm = np.zeros((len(x_adv_ebm)-1)) # Learning rate at best alpha
                        for l in range(len(x_adv_ebm)-1):
                            next_xadv_list = []
                            for k in range(-50, 151):
                                next_xadv_list.append((1.-0.01*k/args.alpha)*x_adv_ebm[l][0].to('cuda')+(0.01*k/args.alpha)*x_adv_ebm[l+1][0].to('cuda'))
                            score_norm_xadvs = torch.stack(next_xadv_list, dim=0)
                            with torch.no_grad():
                                labels = torch.ones(201, device='cuda')*10
                                labels = labels.long()
                                next_xadv_grad = network_ebm(score_norm_xadvs, labels)
                            next_xadv_norm = torch.norm(next_xadv_grad.view(201,-1), dim=1).to('cpu').numpy()
                            best_alpha_pars[l] = (np.argmin(next_xadv_norm)-50.)/100.
                            best_alpha_score_norm[l] = next_xadv_norm[np.argmin(next_xadv_norm)]

                        fig, ax = plt.subplots()
                        epoch_ls = np.linspace(0, len(x_adv_ebm)-2, len(x_adv_ebm)-1)
                        ax.plot(epoch_ls, best_alpha_pars, '-', color='r')
                        plt.xlabel('Iteration')
                        plt.ylabel('Best alpha')
                        plt.savefig(os.path.join("log_images", args.image_folder, "best_alpha_{}.png".format(i)))
                        plt.close(fig)

                        fig, ax = plt.subplots()
                        epoch_ls = np.linspace(0, len(x_adv_ebm)-2, len(x_adv_ebm)-1)
                        ax.plot(epoch_ls, best_alpha_score_norm, '-', color='r')
                        plt.xlabel('Iteration')
                        plt.ylabel('Score norm at best alpha')
                        plt.savefig(os.path.join("log_images", args.image_folder, "best_alpha_score_norm_{}.png".format(i)))
                        plt.close(fig)

                ### Make image grid list
                if if_cifar: # CIFAR10 case
                    chn = 3
                    sz = 32
                else:
                    chn = 1
                    sz = 28
    
                if j%1==0:
                    x_view = x_ebm[j].view(100, chn, sz, sz)
                    x_adv_view = x_adv_ebm[j].view(100, chn, sz, sz)
                    imgs.append(Image.fromarray(make_grid(x_view, nrow=10).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()))
                    imgs_adv.append(Image.fromarray(make_grid(x_adv_view, nrow=10).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()))
                if j==len(sigmas)-1:
                    x_fin = x_ebm[j].view(100, chn, sz, sz).to('cuda')
                x_adv_fin = x_adv_ebm[j].view(100, chn, sz, sz).to('cuda')
                image_grid = make_grid(x, nrow=grid_size)
                save_image(image_grid, os.path.join("log_images", args.image_folder, 'image_{}.png'.format(i)))

            # Ensemble calculation
            for ind1 in range(len(sigmas)):
                for ind2 in range(100):
                    logit_x_onehot_sum[ind2, logit_x_onehot[ind1][ind2]] += 1
                    logit_xadv_onehot_sum[ind2, logit_xadv_onehot[ind1][ind2]] += 1
            p_max = np.argmax(logit_x_onehot_sum, axis=1)
            p_max_adv = np.argmax(logit_xadv_onehot_sum, axis=1)
            acc_x_ensemble_one_hot += np.sum(np.equal(p_max, y.cpu().numpy()))
            acc_xadv_ensemble_one_hot += np.sum(np.equal(p_max_adv, y.cpu().numpy()))
            p_max_logit = np.argmax(logit_x_sum, axis=1)
            p_max_adv_logit = np.argmax(logit_xadv_sum, axis=1)
            acc_x_ensemble_logit += np.sum(np.equal(p_max_logit, y.cpu().numpy()))
            acc_xadv_ensemble_logit += np.sum(np.equal(p_max_adv_logit, y.cpu().numpy()))

        # If rand_smoothing after all purification
        if args.rand_smoothing:
            sigma_rand = np.array([(args.smoothing_level/256.)**2 / 2.])
            x_r = x_ebm[len(sigmas)-1]
            x_adv_r = x_adv_ebm[len(sigmas)-1]

            predicted_list = []
            predicted_list_adv = []
            logit_sum = np.zeros([100,10])
            logit_sum_adv = np.zeros([100,10])
            with torch.no_grad():
                for k in range(10): # Fix to 10 iterations
                    x_r_t = anneal_noise(x_r, network_ebm, sigma_rand)
                    x_adv_r_t = anneal_noise(x_adv_r, network_ebm, sigma_rand)
                    if if_cifar: # Transform into normalized image for classification
                        x_r_t = transform_cifar(x_r_t[1]).to('cuda')
                        x_adv_r_t = transform_cifar(x_adv_r_t[1]).to('cuda')
                    else:
                        x_r_t = x_r_t[1].to('cuda')
                        x_adv_r_t = x_adv_r_t[1].to('cuda')
                    yhat = network(x_r_t)
                    yhat_adv = network(x_adv_r_t)
                    _, predicted = yhat.max(1)
                    _, predicted_adv = yhat_adv.max(1)
                    predicted_list.append(predicted.detach())
                    predicted_list_adv.append(predicted_adv.detach())
                    logit_sum += yhat.detach().to('cpu').numpy()
                    logit_sum_adv += yhat_adv.detach().to('cpu').numpy()
            predicted_stat = np.zeros([100,10])
            predicted_stat_adv = np.zeros([100,10])
            for k in range(10):
                for l in range(100):
                    predicted_stat[l,predicted_list[k][l]] += 1
                    predicted_stat_adv[l,predicted_list_adv[k][l]] += 1
            p_max = np.argmax(predicted_stat, axis=1)
            p_max_adv = np.argmax(predicted_stat_adv, axis=1)
            acc_rand += np.sum(np.equal(p_max,y.cpu().numpy()))
            acc_rand_adv += np.sum(np.equal(p_max_adv,y.cpu().numpy()))
            p_max_logit = np.argmax(logit_sum, axis=1)
            p_max_adv_logit = np.argmax(logit_sum_adv, axis=1)
            acc_rand_logit += np.sum(np.equal(p_max_logit, y.cpu().numpy()))
            acc_rand_adv_logit += np.sum(np.equal(p_max_adv_logit, y.cpu().numpy()))

        # Get randomized inputs and purify.
        # Input noise ensemble accuracy metrics
        logit_noisy_x_sum = np.zeros([100, 10])
        logit_noisy_xadv_sum = np.zeros([100, 10])
        logit_noisy_x_onehot = []
        logit_noisy_xadv_onehot = []
        logit_noisy_x_onehot_sum = np.zeros([100, 10])
        logit_noisy_xadv_onehot_sum = np.zeros([100, 10])
        for inds in range(args.input_ensemble):
            x_p = torch.clamp(x + torch.randn_like(x)*(args.init_noise/255.), min=0.0, max=1.0)
            x_adv_p = torch.clamp(x_adv + torch.randn_like(x_adv)*(args.init_noise/255.), min=0.0, max=1.0)
            with torch.no_grad():
                if args.purify_method=="anneal_langevin":
                    x_ebm = anneal_Langevin_dynamics(x_p, network_ebm, sigmas)
                    x_adv_ebm = anneal_Langevin_dynamics(x_adv_p, network_ebm, sigmas)
                elif args.purify_method=="projection":
                    x_ebm, x_ebm_grad, x_ebm_grad_std = proj_scorenet(x_p, network_ebm, sigmas, save_score_norm=True)
                    x_adv_ebm, x_adv_ebm_grad, x_adv_ebm_grad_std = proj_scorenet(x_adv_p, network_ebm, sigmas, save_score_norm=True)
                elif args.purify_method=="rand":
                    x_ebm = anneal_noise(x_p, network_ebm, sigmas)
                    x_adv_ebm = anneal_noise(x_adv_p, network_ebm, sigmas)
                elif args.purify_method=="adaptive":
                    x_ebm, x_ebm_grad, x_ebm_grad_std, x_step_sizes = adaptive_proj_scorenet(x_p, network_ebm, save_score_norm=True, alpha=args.alpha)
                    x_adv_ebm, x_adv_ebm_grad, x_adv_ebm_grad_std, x_adv_step_sizes = adaptive_proj_scorenet(x_adv_p, network_ebm, save_score_norm=True, alpha=args.alpha)
            
                if if_cifar:
                    x_ebm[-1] = transform_cifar(x_ebm[-1]).to('cuda')
                    x_adv_ebm[-1] = transform_cifar(x_adv_ebm[-1]).to('cuda')
                yhat = network(x_ebm[-1])
                yhat_adv = network(x_adv_ebm[-1])
                _, predicted = yhat.max(1)
                _, predicted_adv = yhat_adv.max(1)
                logit_noisy_x_sum += yhat.detach().to('cpu').numpy()
                logit_noisy_xadv_sum += yhat_adv.detach().to('cpu').numpy()
                logit_noisy_x_onehot.append(predicted.detach().to('cpu').numpy())
                logit_noisy_xadv_onehot.append(predicted_adv.detach().to('cpu').numpy())

        for ind1 in range(args.input_ensemble):
            for ind2 in range(100):
                logit_noisy_x_onehot_sum[ind2, logit_noisy_x_onehot[ind1][ind2]] += 1
                logit_noisy_xadv_onehot_sum[ind2, logit_noisy_xadv_onehot[ind1][ind2]] += 1
        p_max = np.argmax(logit_noisy_x_onehot_sum, axis=1)
        p_max_adv = np.argmax(logit_noisy_xadv_onehot_sum, axis=1)
        acc_noisy_x_ensemble_one_hot += np.sum(np.equal(p_max, y.cpu().numpy()))
        acc_noisy_xadv_ensemble_one_hot += np.sum(np.equal(p_max_adv, y.cpu().numpy()))
        p_max_logit = np.argmax(logit_noisy_x_sum, axis=1)
        p_max_adv_logit = np.argmax(logit_noisy_xadv_sum, axis=1)
        acc_noisy_x_ensemble_logit += np.sum(np.equal(p_max_logit, y.cpu().numpy()))
        acc_noisy_xadv_ensemble_logit += np.sum(np.equal(p_max_adv_logit, y.cpu().numpy()))

        acc1 = acc_noisy_x_ensemble_one_hot / ((i+1)*grid_size**2) * 100.
        acc2 = acc_noisy_xadv_ensemble_one_hot / ((i+1)*grid_size**2) * 100.
        acc3 = acc_noisy_x_ensemble_logit / ((i+1)*grid_size**2) * 100.
        acc4 = acc_noisy_xadv_ensemble_logit / ((i+1)*grid_size**2) * 100.
        print("{} x_onehot {:.2f} x_logit {:.2f} xadv_onehot {:.2f} xadv_logit {:.2f}".format(args.image_folder, acc1[0], acc2[0], acc3[0], acc4[0]))

        ### Make video and image maps
        # Expected output: movie.gif, movie_adv.gif (Purifying procedure of natural and adversarial images)
        # Expected output (Grayscale): images.png (100 natural, adversarial, purified images and saliency map)
        if i==0:
            imgs[0].save(os.path.join("log_images", args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)
            imgs_adv[0].save(os.path.join("log_images", args.image_folder, "movie_adv.gif"), save_all=True, append_images=imgs_adv[1:], duration=1, loop=0)

            if not if_cifar:
                x_first = x.view(100, 1, 28, 28)
                x_adv_first = x_adv.view(100, 1, 28, 28)
    
                # Make plot with 8 images
                image_grid_xfirst = make_grid(x_first, nrow=grid_size)[0,:,:].detach().cpu().numpy()
                image_grid_xadvfirst = make_grid(x_adv_first, nrow=grid_size)[0,:,:].detach().cpu().numpy()
                image_grid_xfin = make_grid(x_fin, nrow=grid_size)[0,:,:].detach().cpu().numpy()
                image_grid_xadvfin = make_grid(x_adv_fin, nrow=grid_size)[0,:,:].detach().cpu().numpy()
                image_grid_xtoxadv = make_grid(x_adv_first - x_first, nrow=grid_size)[0,:,:].detach().cpu().numpy()
    
                image_grid_pre = make_grid(x_fin - x_first, nrow=grid_size)
                image_grid_xtoxfin = make_grid(x_fin - x_first, nrow=grid_size)[0,:,:].detach().cpu().numpy()
                image_grid_xadvtoxfinadv = make_grid(x_adv_fin - x_adv_first, nrow=grid_size)[0,:,:].detach().cpu().numpy()
                image_grid_xtoxfinadv = make_grid(x_adv_fin - x_first, nrow=grid_size)[0,:,:].detach().cpu().numpy()
    
                ax1 = plt.subplot(2,4,1)
                ax1.axis('off')
                ax1.set_title('Initial',fontsize=6)
                ax1.imshow(image_grid_xfirst, cmap='gray')
                ax2 = plt.subplot(2,4,2)
                ax2.axis('off')
                ax2.set_title('Initial+adv',fontsize=6)
                ax2.imshow(image_grid_xadvfirst, cmap='gray')
                ax3 = plt.subplot(2,4,3)
                ax3.axis('off')
                ax3.set_title('Final',fontsize=6)
                ax3.imshow(image_grid_xfin, cmap='gray')
                ax4 = plt.subplot(2,4,4)
                ax4.axis('off')
                ax4.set_title('Final+adv',fontsize=6)
                ax4.imshow(image_grid_xadvfin, cmap='gray')
                ax5 = plt.subplot(2,4,5)
                ax5.axis('off')
                ax5.set_title('Initial->InitAdv',fontsize=6)
                ax5.imshow(image_grid_xtoxadv, cmap='seismic', vmax=1.0, vmin=-1.0)
                ax6 = plt.subplot(2,4,6)
                ax6.axis('off')
                ax6.set_title('Initial->Final',fontsize=6)
                ax6.imshow(image_grid_xtoxfin, cmap='seismic', vmax=1.0, vmin=-1.0)
                ax7 = plt.subplot(2,4,7)
                ax7.axis('off')
                ax7.set_title('Adv->FinAdv',fontsize=6)
                ax7.imshow(image_grid_xadvtoxfinadv, cmap='seismic', vmax=1.0, vmin=-1.0)
                ax8 = plt.subplot(2,4,8)
                ax8.axis('off')
                ax8.set_title('Init->FinAdv',fontsize=6)
                ax8.imshow(image_grid_xtoxfinadv, cmap='seismic', vmax=1.0, vmin=-1.0)
                plt.savefig(os.path.join("log_images", args.image_folder, "images.png"), dpi=800)
                plt.close()

    else:
        break

    ### Print whole result
    acc_x_current = acc_x / ((i+1)*grid_size**2)
    acc_xadv_current = acc_xadv / ((i+1)*grid_size**2)
    if args.rand_smoothing:
        acc_rand_current = acc_rand / ((i+1)*grid_size**2)
        acc_rand_adv_current = acc_rand_adv / ((i+1)*grid_size**2)
        acc_rand_logit_current = acc_rand_logit / ((i+1)*grid_size**2)
        acc_rand_adv_logit_current = acc_rand_adv_logit / ((i+1)*grid_size**2)

    acc_x_ensemble_one_hot_current = acc_x_ensemble_one_hot / ((i+1)*grid_size**2)
    acc_xadv_ensemble_one_hot_current = acc_xadv_ensemble_one_hot / ((i+1)*grid_size**2)
    acc_x_ensemble_logit_current = acc_x_ensemble_logit / ((i+1)*grid_size**2)
    acc_xadv_ensemble_logit_current = acc_xadv_ensemble_logit / ((i+1)*grid_size**2)
    
    # score norm metrics
    score_norm_current = score_norm / (i+1)
    score_norm_adv_current = score_norm_adv / (i+1)
    score_std_current = score_std / (i+1)
    score_std_adv_current = score_std_adv / (i+1)
    cos_sim_current = cos_sim / (i+1)
    cos_sim_adv_current = cos_sim_adv / (i+1)



    if i==0:
        print("Epoch Folder_name Ori_first Ori_max Ori_last Adv_first Adv_max Adv_last Ori_noise_onehot Ori_noise_logit Adv_noise_onehot Adv_noise_logit Ori_ensemble_onehot Ori_ensemble_logit Adv_ensemble_onnehot Adv_ensemble_logit")
    if args.rand_smoothing:
        print('{} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(i+1, args.image_folder, acc_x_current[0]*100.,np.max(acc_x_current)*100.,  acc_x_current[-1]*100., acc_xadv_current[0]*100., np.max(acc_xadv_current)*100., acc_xadv_current[-1]*100., acc_rand_current[0]*100., acc_rand_logit_current[0]*100., acc_rand_adv_current[0]*100., acc_rand_adv_logit_current[0]*100., acc_x_ensemble_one_hot_current[0]*100., acc_x_ensemble_logit_current[0]*100., acc_xadv_ensemble_one_hot_current[0]*100., acc_xadv_ensemble_logit_current[0]*100.))
    else:
        print('{}\t{}\tOri {:.2f}->{:.2f}\tAdv {:.2f}->{:.2f}\tMax {:.2f}'.format(i+1, args.image_folder, acc_x_current[0]*100., acc_x_current[-1]*100., acc_xadv_current[0]*100., acc_xadv_current[-1]*100., np.max(acc_xadv_current)*100.))

    sys.stdout.flush()

    ### Plot figures every iteration
    # Expected outputs
    # acc.png: Accuracy of natural and adversarial images
    # l2.png: l2 distance between (natural, nat-purified) / (natural, adv-purified) / (nat-purified, Adv-purified)
    # score_norm.png: average score norm
    # score_std.png: average score standard deviation
    # cos_sim.png: average cosine similarity between scores at adjacent purification steps
    # learning_rate.png: average step size, fixed if not adaptive
    fig, ax = plt.subplots()
    epoch_ls = np.linspace(0, len(acc_x)-2, len(acc_x)-1)
    ax.plot(epoch_ls, acc_x_current[1:]*100., '-', color='r', label='Natural')
    ax.plot(epoch_ls, acc_xadv_current[1:]*100., '-', color='b', label='Adversarial')
    if args.rand_smoothing:
        ax.plot(len(acc_x)-2, acc_rand_current*100, 'x', color='r')
        ax.plot(len(acc_xadv)-2, acc_rand_adv_current*100, 'x', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join("log_images", args.image_folder,"acc.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    epoch_ls = np.linspace(0, len(acc_x)-2, len(acc_x)-1)
    ax.plot(epoch_ls, l2[1:], '-', color='r', label='Nat-NatPur')
    ax.plot(epoch_ls, l2_adv[1:], '-', color='b', label='Nat-AdvPur')
    ax.plot(epoch_ls, l2_diff[1:], '-', color='g', label='NatPur-AdvPur')
    plt.xlabel('Iteration')
    plt.ylabel('l2 distance')
    plt.legend()
    plt.savefig(os.path.join("log_images", args.image_folder,"l2.png"))
    plt.close(fig)

    # Show score norm metrics
    fig, ax = plt.subplots()
    epoch_ls = np.linspace(0, len(score_norm)-1, len(score_norm))
    ax.plot(epoch_ls, score_norm_current, '-', color='r', label='Natural')
    ax.plot(epoch_ls, score_norm_adv_current, '-', color='b', label='Adversarial')
    plt.xlabel('Iteration')
    plt.ylabel('score norm')
    plt.legend()
    plt.savefig(os.path.join("log_images", args.image_folder, "score_norm.png"))
    plt.close(fig)

    # Show score std metrics
    fig, ax = plt.subplots()
    epoch_ls = np.linspace(0, len(score_norm)-1, len(score_norm))
    ax.plot(epoch_ls, score_std_current, '-', color='r', label='Natural')
    ax.plot(epoch_ls, score_std_adv_current, '-', color='b', label='Adversarial')
    plt.xlabel('Iteration')
    plt.ylabel('score std')
    plt.legend()
    plt.savefig(os.path.join("log_images", args.image_folder, "score_std.png"))
    plt.close(fig)

    # Show cosine similarity metrics
    fig, ax = plt.subplots()
    epoch_ls = np.linspace(0, len(score_norm)-2, len(score_norm)-1)
    ax.plot(epoch_ls, cos_sim_current, '-', color='r', label='Natural')
    ax.plot(epoch_ls, cos_sim_adv_current, '-', color='b', label='Adversarial')
    plt.ylim((-1.1, 1.1))
    plt.xlabel('Iteration')
    plt.ylabel('cosine similarity: (i)th - (i+1)th')
    plt.legend()
    plt.savefig(os.path.join("log_images", args.image_folder, "cos_sim.png"))
    plt.close(fig)

    # Show learning rates
    if args.purify_method=="adaptive":
        x_step_size_t = torch.stack(x_step_sizes, dim=0)
        x_adv_step_size_t = torch.stack(x_adv_step_sizes, dim=0)
    else:
        x_step_size = sigmas
        x_adv_step_size = sigmas
    fig, ax = plt.subplots()

    if args.purify_method=="adaptive":
        epoch_ls = np.linspace(0, x_step_size_t.shape[0]-1, x_step_size_t.shape[0])
        ax.plot(epoch_ls, torch.mean(x_step_size_t,dim=1).to('cpu').numpy(), '-', color='r', label='Natural')
        ax.plot(epoch_ls, torch.mean(x_adv_step_size_t,dim=1).to('cpu').numpy(), '-', color='b', label='Adversarial')
    else:
        epoch_ls = np.linspace(0, x_step_size.shape[0]-1, x_step_size.shape[0])
        ax.plot(epoch_ls, sigmas, '-', color='r', label='Natural')
        ax.plot(epoch_ls, sigmas, '-', color='b', label='Adversarial')
    plt.xlabel('Iteration')
    plt.ylabel('Learning rate, Clamped at 10^-6')
    plt.legend()
    plt.savefig(os.path.join("log_images", args.image_folder, "learning_rate.png"))
    plt.close(fig)
