#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms, models
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F

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
        rand_set_all = np.load('save/rotated1/randset_fed_mnist_iidFalse_num100_C0.1.pt.npy')
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

    fed_model_path = args.load_fed_name
    net_glob.load_state_dict(torch.load(fed_model_path))

    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    acc_test_rotated, loss_test_rotated = test_img_local(net_glob, dataset_rotated_test, args, user_idx=0,
                                                         idxs=dict_users_rotated_test[0])

    print('Initial Fed Model, Loss: {}, Acc: {}, Loss Rotated: {}, Acc: {}'.format(
        loss_test, acc_test, loss_test_rotated, acc_test_rotated))

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
        w_glob = None
        loss_locals, grads_local = [], []
        m = int(args.frac * args.num_users)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = copy.deepcopy(net_glob)

            w_local, loss = local.train(net=net_local.to(args.device))
            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]

        # rotated
        local = LocalUpdate(args=args, dataset=dataset_rotated_train, idxs=dict_users_rotated_train[0])
        # local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[0])
        net_local = copy.deepcopy(net_glob)

        w_local, loss = local.train(net=net_local.to(args.device))
        loss_locals.append(copy.deepcopy(loss))

        if w_glob is None:
            w_glob = copy.deepcopy(w_local)
        else:
            for k in w_glob.keys():
                w_glob[k] += w_local[k]

        lr *= args.lr_decay

        # update global weights
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m + 1)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # class DatasetSplit(Dataset):
        #     def __init__(self, dataset, idxs):
        #         self.dataset = dataset
        #         self.idxs = list(idxs)
        #
        #     def __len__(self):
        #         return len(self.idxs)
        #
        #     def __getitem__(self, item):
        #         image, label = self.dataset[self.idxs[item]]
        #         return image, label
        #
        # train_loader = DataLoader(dataset_rotated_train, batch_size=64, shuffle=True)
        # optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
        #
        # for batch_idx, (data, target) in enumerate(train_loader):
        #     data, target = data.to(args.device), target.to(args.device)
        #     optimizer.zero_grad()
        #     output = net_glob(data)
        #     loss = F.cross_entropy(output, target)
        #     loss.backward()
        #     optimizer.step()

        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_test_rotated, loss_test_rotated = test_img(net_glob, dataset_rotated_test, args)

        print('Epoch {}, Loss: {}, Acc: {}, Loss Rotated: {}, Acc Rotated: {}'. format(
            iter, loss_test, acc_test, loss_test_rotated, acc_test_rotated))

        results.append([iter, loss_test, acc_test, loss_test_rotated, acc_test_rotated])
        final_results = np.array(results)
        results_save_path = './log/rotated/fedfed_{}_C{}_iid{}.npy'.format(
            args.dataset, args.frac, args.iid)
        np.save(results_save_path, final_results)