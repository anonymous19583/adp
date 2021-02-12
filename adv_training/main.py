import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
from methods import *

def parse_args_and_config():
    # Attack/defense parser
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--mode', type=str, default='defense', help='attack: make adversarial images/defense: adversarial training')
    parser.add_argument('--config', type=str, default='config.yml', help='path to the config file')
    parser.add_argument('--method', type=str, default='fgsm', help='adversarial training or attack method')
    parser.add_argument('--norm', type=int, default=-1, help='lp norm, -1 if linf')
    parser.add_argument('--ptb', type=float, default=8.0, help='perturbation level by pixel levels')
    parser.add_argument('--network', type=str, default='WideResNet', help='network to use')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='0', help='A string for documentation purpose')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('-o', '--image_folder', type=str, default='images', help="The directory of image outputs")

    args = parser.parse_args()
    run_id = str(os.getpid())
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # args.doc = '_'.join([args.doc, run_id, run_time])
    path_root = '/path_root'
    adv_root = os.path.join(path_root, "adv_training")
    args.log = os.path.join(adv_root, args.run, 'logs', args.doc)

    if args.mode=="defense":
        with open(os.path.join(adv_root, 'configs', args.config), 'r') as f:
            config = yaml.load(f)
            new_config = dict2namespace(config)

    elif args.mode=="attack":
        with open(os.path.join(args.log, args.config), 'r') as f:
            config = yaml.load(f)
            new_config = config
    
    elif args.mode=="feature":
        with open(os.path.join(adv_root, 'configs', args.config), 'r') as f:
            config = yaml.load(f)
            new_config = dict2namespace(config)

    # parse config file
    if args.mode=='defense':
        if not args.resume_training:
            if os.path.exists(args.log):
                shutil.rmtree(args.log)
            os.makedirs(args.log)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))
        with open(os.path.join(args.log, args.config), 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    args.device = device
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        method = eval(args.method)(args, config)
        if args.mode=='defense':
            method.train()
        elif args.mode=='attack':
            method.test()
        elif args.mode=='feature':
            method.feature()
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
