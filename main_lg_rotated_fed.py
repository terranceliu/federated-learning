#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import os
import itertools
import numpy as np
from scipy.stats import mode
from torchvision import datasets, transforms, models
import torch
from torch import nn

from utils.sampling import mnist_iid, mnist_noniid, cifar10_iid, cifar10_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, ResnetCifar, AllConvNet
from models.Fed import FedAvg
from models.test import test_img, test_img_local

import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    trans_mnist = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])

    dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
    img_size = dataset_train[0][0].shape

    if args.iid:
        dict_users_train = mnist_iid(dataset_train, args.num_users)
        dict_users_test = mnist_iid(dataset_test, args.num_users)
    else:
        rand_set_all = np.load('save/rotated1/randset_lg_mnist_iidFalse_num100_C0.1.pt.npy')
        dict_users_train, _ = mnist_noniid(dataset_train, args.num_users, num_shards=200, num_imgs=300,
                                                      train=True, rand_set_all=rand_set_all)
        dict_users_test, _ = mnist_noniid(dataset_test, args.num_users, num_shards=200, num_imgs=50, train=False,
                                          rand_set_all=rand_set_all)

    trans_mnist_rotate = transforms.Compose([transforms.RandomVerticalFlip(p=1.0),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])

    trans_mnist_rotate = transforms.Compose([transforms.RandomRotation((90, 90)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])

    dataset_rotated_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist_rotate)
    dataset_rotated_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist_rotate)
    dict_users_rotated_train = mnist_iid(dataset_rotated_train, 20)
    dict_users_rotated_test = mnist_iid(dataset_rotated_test, 20)

    # build model
    if args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')


    print(net_glob)
    net_glob.train()
    total_num_layers = len(net_glob.weight_keys)
    w_glob_keys = net_glob.weight_keys[total_num_layers - args.num_layers_keep:]
    w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    num_param_glob = 0
    num_param_local = 0
    for key in net_glob.state_dict().keys():
        num_param_local += net_glob.state_dict()[key].numel()
        if key in w_glob_keys:
            num_param_glob += net_glob.state_dict()[key].numel()
    percentage_param = 100 * float(num_param_glob) / num_param_local
    print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
        num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    # generate list of local models for each user

    fed_model_path = args.load_fed_name
    net_glob.load_state_dict(torch.load(fed_model_path))

    net_local_list = []
    for user_ix in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))
    net_local_list.append(copy.deepcopy(net_glob)) # for new rotated device

    criterion = nn.CrossEntropyLoss()

    def test_img_local_all():
        acc_test_local = 0
        loss_test_local = 0
        for idx in range(args.num_users):
            net_local = net_local_list[idx]
            net_local.eval()
            a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])

            acc_test_local += a
            loss_test_local += b
        acc_test_local /= args.num_users
        loss_test_local /= args.num_users

        return acc_test_local, loss_test_local

    def test_img_avg_all():
        net_glob_temp = copy.deepcopy(net_glob)
        w_keys_epoch = net_glob.state_dict().keys()
        w_glob_temp = {}
        for idx in range(args.num_users + 1):
            net_local = net_local_list[idx]
            w_local = net_local.state_dict()

            if len(w_glob_temp) == 0:
                w_glob_temp = copy.deepcopy(w_local)
            else:
                for k in w_keys_epoch:
                    w_glob_temp[k] += w_local[k]

        for k in w_keys_epoch:
            w_glob_temp[k] = torch.div(w_glob_temp[k], args.num_users)
        net_glob_temp.load_state_dict(w_glob_temp)
        acc_test_avg, loss_test_avg = test_img(net_glob_temp, dataset_test, args)

        return acc_test_avg, loss_test_avg

    acc_test_local, loss_test_local = test_img_local_all()
    acc_test_avg, loss_test_avg = test_img_avg_all()

    net_local = net_local_list[args.num_users]
    loss_test_local_rotated, acc_test_local_rotated = test_img_local(net_local, dataset_rotated_test, args, user_idx=0, idxs=dict_users_rotated_test[0])

    print('Initial: Loss: {:.3f}, Acc: {:.2f}, Loss (avg): {:.3}, Acc (avg): {:.2f}, Loss (rotated) {:.3f}, Acc: (rotated) {:.2f}, '.format(
        loss_test_local, acc_test_local, loss_test_avg, acc_test_avg, loss_test_local_rotated, acc_test_local_rotated))

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    lr = args.lr
    results = []

    for iter in range(args.epochs):
        w_glob = {}
        loss_locals, grads_local = [], []
        m = int(args.frac * args.num_users)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # w_keys_epoch = net_glob.state_dict().keys() if (iter + 1) % 25 == 0 else w_glob_keys
        w_keys_epoch = w_glob_keys

        if args.verbose:
            print("Round {}: lr: {:.6f}, {}".format(iter, lr, idxs_users))
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = net_local_list[idx]

            w_local, loss = local.train(net=net_local.to(args.device), lr=lr)
            loss_locals.append(copy.deepcopy(loss))

            # sum up weights
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_keys_epoch:
                    w_glob[k] += w_local[k]

        # rotated
        local = LocalUpdate(args=args, dataset=dataset_rotated_train, idxs=dict_users_rotated_train[0])
        net_local = net_local_list[args.num_users]

        w_local, loss = local.train(net=net_local.to(args.device))
        loss_locals.append(copy.deepcopy(loss))

        if len(w_glob) == 0:
            w_glob = copy.deepcopy(w_local)
        else:
            for k in w_glob.keys():
                w_glob[k] += w_local[k]

        if (iter+1) % max(int(args.frac * args.num_users), 1):
            lr *= args.lr_decay

        # get weighted average for global weights
        for k in w_keys_epoch:
            w_glob[k] = torch.div(w_glob[k], m + 1)

        # copy weight to the global model (not really necessary)
        net_glob.load_state_dict(w_glob)

        # copy weights to each local model
        for idx in range(args.num_users + 1):
            net_local = net_local_list[idx]
            w_local = net_local.state_dict()
            for k in w_keys_epoch:
                w_local[k] = w_glob[k]

            net_local.load_state_dict(w_local)


        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # eval
        acc_test_local, loss_test_local = test_img_local_all()
        acc_test_avg, loss_test_avg = test_img_avg_all()

        net_local = net_local_list[args.num_users]
        acc_test_local_rotated, loss_test_local_rotated = test_img_local(net_local, dataset_rotated_test, args,
                                                                         user_idx=0, idxs=dict_users_rotated_test[0])

        print('Epoch {}: Loss: {:.3f}, Acc: {:.2f}, Loss (avg): {:.3}, Acc (avg): {:.2f}, Loss (rotated) {:.3f}, Acc: (rotated) {:.2f}, '.format(
                iter, loss_test_local, acc_test_local, loss_test_avg, acc_test_avg, loss_test_local_rotated,
                acc_test_local_rotated))

        results.append([iter, loss_test_local, acc_test_local, loss_test_avg, acc_test_avg, loss_test_local_rotated,
                acc_test_local_rotated])
        final_results = np.array(results)
        results_save_path = './log/rotated/fedlg_{}_C{}_iid{}.npy'.format(
            args.dataset, args.frac, args.iid)
        np.save(results_save_path, final_results)
