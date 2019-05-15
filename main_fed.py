#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
# from models.Fed import FedAvg
from models.test import test_img

from utils.utils import BufferedDataIterator, NLIIterator
from models.models import MultitaskModel
import pdb

src_vocab_size = 81000
trg_vocab_size = 31000
max_len_src = 90
max_len_trg = 90
save_dir = "data/models/example"

num_users = 50

paths = [{"train_src": "data/corpora/nmt/training/num_users_{}/train.nmt.de-en.en.tok".format(num_users),
            "train_trg": "data/corpora/nmt/training/num_users_{}/train.nmt.de-en.de.tok".format(num_users),
            "val_src": "data/corpora/nmt/training/num_users_{}/dev.nmt.de-en.en.tok".format(num_users),
            "val_trg": "data/corpora/nmt/training/num_users_{}/dev.nmt.de-en.de.tok".format(num_users),
            "taskname": "de-en"
          },
        {"train_src": "data/corpora/nmt/training/num_users_{}/train.nmt.fr-en.en.tok".format(num_users),
            "train_trg": "data/corpora/nmt/training/num_users_{}/train.nmt.fr-en.fr.tok".format(num_users),
            "val_src": "data/corpora/nmt/training/num_users_{}/dev.nmt.fr-en.en.tok".format(num_users),
            "val_trg": "data/corpora/nmt/training/num_users_{}/dev.nmt.fr-en.fr.tok".format(num_users),
            "taskname": "fr-en"
         }]
train_srcs = []
train_trgs = []
val_srcs = []
val_trgs = []
for user_idx in range(num_users):
    train_srcs.append(['{}.{}'.format(path['train_src'], user_idx) for path in paths])
    train_trgs.append(['{}.{}'.format(path['train_trg'], user_idx) for path in paths])
    val_srcs.append(['{}.{}'.format(path['val_src'], user_idx) for path in paths])
    val_trgs.append(['{}.{}'.format(path['val_trg'], user_idx) for path in paths])
num_tasks = len(paths)
tasknames = [item['taskname'] for item in paths]
tasknames.append('NLI')


paths_all = [{"train_src": "data/corpora/nmt/training/train.nmt.de-en.en.tok",
                "train_trg": "data/corpora/nmt/training/train.nmt.de-en.de.tok",
                "val_src": "data/corpora/nmt/training/dev.nmt.de-en.en.tok",
                "val_trg": "data/corpora/nmt/training/dev.nmt.de-en.de.tok",
                "taskname": "de-en"
              },
             {"train_src": "data/corpora/nmt/training/train.nmt.fr-en.en.tok",
                "train_trg": "data/corpora/nmt/training/train.nmt.fr-en.fr.tok",
                "val_src": "data/corpora/nmt/training/dev.nmt.fr-en.en.tok",
                "val_trg": "data/corpora/nmt/training/dev.nmt.fr-en.fr.tok",
                "taskname": "fr-en"
              }]
train_src_all = [path['train_src'] for path in paths_all]
train_trg_all = [path['train_trg'] for path in paths_all]
val_src_all = [path['val_src'] for path in paths_all]
val_trg_all = [path['val_trg'] for path in paths_all]

train_iterators = []
for user_idx in range(num_users):
    train_iterator = BufferedDataIterator(
        train_srcs[user_idx], train_trgs[user_idx],
        train_src_all, train_trg_all,
        src_vocab_size, trg_vocab_size,
        tasknames, save_dir,
        user_idx=user_idx, buffer_size=1e4, lowercase=True
    )
    train_iterators.append(train_iterator)
    print('\n')


path = {"nli_train": "data/corpora/nli/num_users_{}/allnli.train.txt.clean.noblank".format(num_users),
        "nli_dev": "data/corpora/nli/num_users_{}/snli_1.0_dev.txt.clean.noblank".format(num_users),
        "nli_test": "data/corpora/nli/num_users_{}/snli_1.0_test.txt.clean.noblank".format(num_users),
        }
nli_trains = []
nli_devs = []
nli_tests = []
for user_idx in range(num_users):
    nli_trains.append('{}.{}'.format(path['nli_train'], user_idx))
    nli_devs.append('{}.{}'.format(path['nli_dev'], user_idx))
    nli_tests.append('{}.{}'.format(path['nli_test'], user_idx))
nli_iterators = []
for user_idx in range(num_users):
    nli_iterator = NLIIterator(
        train=nli_trains[user_idx],
        dev=nli_devs[user_idx],
        test=nli_tests[user_idx],
        vocab_size=-1,
        vocab=os.path.join(save_dir, 'src_vocab.pkl')
    )
    nli_iterators.append(nli_iterator)


model_config = {"dim_src": 2048,
                "dim_trg": 2048,
                "dim_word_src": 512,
                "dim_word_trg": 512,
                "n_words_src": 80000,
                "n_words_trg": 40000,
                "n_layers_src": 1,
                "bidirectional": True,
                "layernorm": False,
                "dropout": 0.3
                }
paired_tasks = None

model_config = {"dim_src": 1024,
                "dim_trg": 1024,
                "dim_word_src": 256,
                "dim_word_trg": 256,
                "n_words_src": 80000,
                "n_words_trg": 40000,
                "n_layers_src": 1,
                "bidirectional": True,
                "layernorm": False,
                "dropout": 0.3
                }

net_glob = MultitaskModel(src_emb_dim=model_config['dim_word_src'],
                            trg_emb_dim=model_config['dim_word_trg'],
                            src_vocab_size=src_vocab_size,
                            trg_vocab_size=trg_vocab_size,
                            src_hidden_dim=model_config['dim_src'],
                            trg_hidden_dim=model_config['dim_trg'],
                            bidirectional=model_config['bidirectional'],
                            pad_token_src=train_iterator.src[0]['word2id']['<pad>'],
                            pad_token_trg=train_iterator.trg[0]['word2id']['<pad>'],
                            nlayers_src=model_config['n_layers_src'],
                            dropout=model_config['dropout'],
                            num_tasks=len(train_iterator.src),
                            paired_tasks=paired_tasks
                          ).cuda()
print(net_glob)
net_glob.train()

optimizer_global = torch.optim.Adam(net_glob.parameters(), lr=0.0001)
weight_mask = torch.ones(trg_vocab_size).cuda()
weight_mask[train_iterator.trg[0]['word2id']['<pad>']] = 0
loss_criterion = torch.nn.CrossEntropyLoss(weight=weight_mask).cuda()
nli_criterion = torch.nn.CrossEntropyLoss().cuda()

epochs = 1000
epochs_local = 5
batch_size = 36
batches_per_ep = 50 # TODO: figure out how many minibatches per epoch

frac = 0.1
gpu = 0
n_gpus = 1
device = 'cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu'
nli_mbatch_ctr = 0
task_idxs = [[0 for task in tasknames] for i in range(num_users)]

for iter in range(epochs):
    print("Begin epoch {}".format(iter))
    loss_locals = [[] for i in range(num_tasks + 1)]
    w_locals = []
    w_glob = None

    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace=False)

    grads_local = []

    for idx in idxs_users:
        print("Begin training: user {}".format(idx))
        net_local = copy.deepcopy(net_glob)
        optimizer_local = torch.optim.Adam(net_local.parameters(), lr=0.0001)
        train_iterator = train_iterators[idx]
        nli_iterator = nli_iterators[idx]

        num_examples = [0 for i in range(num_users + 1)]

        for iter_local in range(epochs_local):
            epoch_loss = [[] for i in range(num_tasks + 1)]

            for i in range(batches_per_ep):
                if i % 10 == 0:
                    # print('{} NLI {}'.format(idx, nli_mbatch_ctr))

                    minibatch = nli_iterator.get_parallel_minibatch(
                        nli_mbatch_ctr, batch_size * n_gpus
                    )

                    num_examples[num_tasks] += batch_size * n_gpus
                    nli_mbatch_ctr += batch_size * n_gpus
                    if nli_mbatch_ctr >= len(nli_iterator.train_lines):
                        nli_mbatch_ctr = 0

                    optimizer_local.zero_grad()
                    class_logits = net_local(
                        minibatch, -1,
                        return_hidden=False, paired_trg=None
                    )
                    loss = nli_criterion(
                        class_logits.contiguous().view(-1, class_logits.size(1)),
                        minibatch['labels'].contiguous().view(-1)
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(net_local.parameters(), 1.)
                    optimizer_local.step()

                    epoch_loss[num_tasks].append(loss)
                else:
                    task_idx = np.random.randint(low=0, high=num_tasks)
                    # print('{} {} {}'.format(idx, tasknames[task_idx], task_idxs[idx][task_idx]))

                    # Get a minibatch corresponding to the sampled task
                    minibatch = train_iterator.get_parallel_minibatch(
                        task_idx, task_idxs[idx][task_idx], batch_size * n_gpus,
                        max_len_src, max_len_trg
                    )

                    num_examples[task_idx] += batch_size * n_gpus
                    task_idxs[idx][task_idx] += batch_size * n_gpus
                    if task_idxs[idx][task_idx] >= train_iterator.buffer_size:
                        train_iterator.fetch_buffer(task_idx)
                        task_idxs[idx][task_idx] = 0

                    optimizer_local.zero_grad()
                    decoder_logit = net_local(minibatch, task_idx)
                    loss = loss_criterion(
                        decoder_logit.contiguous().view(-1, decoder_logit.size(2)),
                        minibatch['output_trg'].contiguous().view(-1)
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(net_local.parameters(), 1.)
                    optimizer_local.step()

                    epoch_loss[task_idx].append(loss)

        for task_idx in range(num_tasks + 1):
            epoch_loss[task_idx] = sum(epoch_loss[task_idx]) / len(epoch_loss[task_idx])
            print('{}, Num Examples: {} Loss {}'.format(tasknames[task_idx], num_examples[task_idx], epoch_loss[task_idx]))
            loss_locals[task_idx].append(epoch_loss[task_idx])

        grads = []
        for grad in [param.grad for param in net_local.parameters()]:
            if grad is not None:
                grads.append(grad.view(-1))
        grads = torch.cat(grads).norm().item()
        grads_local.append(grads)
        print(grads)

        w_curr = net_local.state_dict()
        if w_glob is None:
            w_glob = w_curr
            for k in w_glob.keys():
                w_glob[k] *= grads
        else:
            for k in w_glob.keys():
                w_glob[k] += w_curr[k] * grads

        # w_curr = net_local.state_dict()
        # if w_glob is None:
        #     w_glob = w_curr
        # else:
        #     for k in w_glob.keys():
        #         w_glob[k] += w_curr[k]

    # update global weights
    # w_glob = FedAvg(w)

    for k in w_glob.keys():
        # w_glob[k] = torch.div(w_glob[k], m)
        w_glob[k] = torch.div(w_glob[k], sum(grads_local))

    # copy weight to net_glob
    net_glob.load_state_dict(w_glob)

    # print loss
    for task_idx in range(num_tasks + 1):
        loss_locals[task_idx] = sum(loss_locals[task_idx]) / len(loss_locals[task_idx])
    print('Round {:3d} Avg Loss, {} {:.3f}, {} {:.3f}, NLI {:.3f}'.format(iter, tasknames[0], loss_locals[0], tasknames[1], loss_locals[1], loss_locals[num_tasks]))


# orig
"""
def main():
    # parse args
    args = args_parser()
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig(
        './log/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
"""
