#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import os
import numpy as np
from torchvision import datasets, transforms, models
import torch
from torch import nn

from utils.sampling import mnist_iid, mnist_noniid, cifar10_iid, cifar10_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNCifar_glob, ResnetCifar
from models.Fed import FedAvg
from models.test import test_img

import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    trans_mnist = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])

    if args.model == 'resnet':
        trans_cifar_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.Resize([256,256]),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
        trans_cifar_val = transforms.Compose([transforms.Resize([256,256]),
                                                transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
    else:
        trans_cifar_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
        trans_cifar_val = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

    # load dataset and split users
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar_val)
        if args.iid:
            dict_users = cifar10_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar10_noniid(dataset_train, args.num_users)
            # exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar_val)
        if args.iid:
            dict_users = cifar10_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset in ['cifar10', 'cifar100']:
        net_local = CNNCifar(args=args).to(args.device)
        net_glob = CNNCifar_glob(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'resnet' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = ResnetCifar(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    print(net_local)
    net_local.train()
    print(net_glob)
    net_glob.train()

    # generate list of local models for each user
    net_local_list = [net_local]
    for user_ix in range(1, args.num_users):
        net_local_list.append(copy.deepcopy(net_local))

    # pretrain each local model
    pretrain_save_path = 'pretrain/{}/{}_{}/user_{}/ep_{}/'.format(args.model, args.dataset, 'iid' if args.iid else 'noniid',args.num_users, args.local_ep_pretrain)
    if not os.path.exists(pretrain_save_path):
        os.makedirs(pretrain_save_path)

    print("\nPretraining local models...")
    for idx in range(args.num_users):
        print("Local model {}".format(idx))
        net_local = net_local_list[idx]
        net_local_path = os.path.join(pretrain_save_path, '{}.pt'.format(idx))
        if os.path.exists(net_local_path): # check if we have a saved model
            net_local.load_state_dict(torch.load(net_local_path))
        else:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], pretrain=True)
            w_local, loss = local.train(net=net_local.to(args.device))
            print('Train Epoch Loss: {:.4f}'.format(loss))
            torch.save(net_local.state_dict(), net_local_path)

        # net_local.eval()
        # acc_test, loss_test, probs = test_img(net_local, dataset_test, args, return_probs=True)
        # probs_all.append(probs.detach())

    criterion = nn.CrossEntropyLoss()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    for iter in range(args.epochs):
        print("\nRound {}".format(iter))

        w_glob = {}
        loss_locals, grads_local = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            print("Local model {}".format(idx))

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            net_local = net_local_list[idx]

            w_local, loss = local.train(net=net_local.to(args.device))
            loss_locals.append(copy.deepcopy(loss))

            # use grads to calculate a weighted average
            if not args.grad_norm:
                grads = 1.0
            else:
                grads = []
                for grad in [param.grad for param in net_local.parameters()]:
                    if grad is not None:
                        grads.append(grad.view(-1))
                grads = torch.cat(grads).norm().item()
            # print(grads)
            grads_local.append(grads)

            # sum up weights
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(net_glob.state_dict())
                for k in w_glob.keys(): # this depends on the layers being named the same (in Nets.py)
                    w_glob[k] = w_local[k] * grads
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k] * grads

        # get weighted average for global weights
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], sum(grads_local))

        # copy weight to the global model (not really necessary)
        net_glob.load_state_dict(w_glob)

        probs_all = []
        # copy weights to each local model
        for idx in range(args.num_users):
            net_local = net_local_list[idx]
            w_local = net_local.state_dict()
            for k in w_glob.keys():
                w_local[k] = w_glob[k]

            net_local.load_state_dict(w_local)

        if iter % 10 == 0:
            for idx in range(args.num_users):
                net_local.eval()
                _, _, probs = test_img(net_local, dataset_test, args, return_probs=True)
                probs_all.append(probs.detach())

            preds_probs = torch.mean(torch.stack(probs_all), dim=0)
            preds = preds_probs.data.max(1, keepdim=True)[1]
            preds = preds.cpu().numpy().reshape(-1)
            labels = np.array(dataset_test.test_labels)

            loss_test = criterion(preds_probs, torch.tensor(labels).cuda())
            acc_test = (preds == labels).mean()

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)

            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(iter, loss_avg, loss_test, acc_test))
            loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./log/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

# python3 main_lg.py --dataset cifar10 --num_classes 10 --num_channels 3 --model cnn --num_users 100 --epochs 2000 --frac 0.1 --local_ep 5 --local_ep_pretrain 50 --local_bs 50 --bs 50 --lr 0.01 --verbose --print_freq 300 --gpu 0 --iid --grad_norm